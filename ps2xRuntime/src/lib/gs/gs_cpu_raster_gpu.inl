// Part of gs_cpu_raster.cpp: the frame and depth buffers kept and drawn by
// the graphics card (GSGpu::Renderer), at whatever size is asked for.
//
// Each (base, width, pixel arrangement, scale) a buffer has been drawn as
// is a surface of the card's. Which surface has what is accounted for a
// VRAM page at a time, in the table the software shadow uses
// (m_shadowSerial, m_shadowKey: see GSCpuBackend::m_shadow): a page of a
// surface is good while the last thing written to the page was a primitive
// drawn into that surface.
//
// With display.renderer = hybrid the raster pool draws every primitive
// into VRAM as ever, and the card draws it again into its surface. A page
// something else wrote (an image, a clear) is filled from VRAM before the
// card next draws there or shows it.
//
// With display.renderer = gpu the card draws alone. VRAM then gets the
// images the game sends and nothing of what is drawn: a page the card
// drew into is VRAM's to catch up on (m_cardAhead), and is brought back
// (GpuDownload) only for something that reads those pixels there. Much of
// what looks like such a reader is not one:
//
// - A game keeps textures and draws by turns in the same memory. Pages
//   that pass between buffers whose pixels are laid out differently (a
//   256-wide frame buffer and a 128-wide one, a frame buffer and a
//   texture) hold nothing either can use of the other's: they change
//   hands as they are (GpuAdopt, GpuUploading, the texture path).
// - What the card alone drew does not count as drawn where VRAM's
//   contents are concerned (MarkWritten's inVram): a texture sent again
//   because the game drew over it is, in VRAM, still there, and the
//   transfer is skipped as any repeated one is.
//
// A primitive's texture is, in order of preference:
//
// - the surface, when the texture is a frame buffer whose pages are good:
//   the frame squeezed into a field, a blur, a billboard drawn a moment
//   ago all stay as sharp as they were drawn;
// - the decoded copy of an uploaded image the software path keeps
//   (DecodedTextureFor), handed to the card the first time;
// - otherwise the texels it can sample, decoded from VRAM and kept for as
//   long as nothing writes them.

#include "runtime/gs/gs_gpu_renderer.h"

struct GSCpuBackend::GpuState
{
    std::unique_ptr<GSGpu::Renderer> renderer;

    struct Surface
    {
        uint32_t id = 0;
        int width = 0, height = 0;
        GSMem::SurfaceLayout layout;
        SurfaceScale scale;
        int depthKind = -1;
        double worst = 0.0; // PS2X_GPU_COMPARE: the largest share of pixels found wrong
    };
    std::unordered_map<uint64_t, Surface> surfaces; // by GpuKey
    // The frame and the depth buffer last asked for: the next primitive's, nearly always.
    struct
    {
        uint64_t key = 0;
        Surface *surface = nullptr;
    } recent[2];
    Surface *last = nullptr;

    // Texels decoded from VRAM.
    struct Texture
    {
        uint32_t id = 0;
        int x0 = 0, y0 = 0, x1 = -1, y1 = -1; // the texels it holds
        uint64_t serial = 0;                  // m_writeSerial when they were read
        uint64_t checkedUpload = 0;           // m_lastUploadSerial when they were last found untouched
        uint64_t hash = 0;
        uint64_t lastUsed = 0;
        uint8_t maxAlpha = 0;
    };
    std::unordered_map<uint64_t, Texture> textures;
    int textureMaxAlpha = 255; // of the texture GpuTexture chose last
    uint64_t clock = 0;

    std::vector<uint32_t> pixels;
    std::vector<float> depths;
    struct Cell
    {
        uint32_t page, row, column;
    };
    std::vector<Cell> cells;

    std::vector<uint32_t> pages;
    // Why pages went back to VRAM, for PS2X_GS_STATS.
    enum Why
    {
        WhyTexture,
        WhyClut,
        WhySoftware,
        WhyOtherSurface,
        WhyImage,
        WhyDisplay,
        WhyEverything,
        WhyCount,
    };
    Why why = WhyEverything;
    uint64_t statWhy[WhyCount]{};

    uint64_t statDraws = 0, statSurfaceTextures = 0, statCopyTextures = 0, statVramTextures = 0, statDecodes = 0, statUploads = 0,
             statFills = 0, statFilledPages = 0, statDownloads = 0, statDownloadedPages = 0, statDropped = 0, statAdopted = 0, statShuffled = 0;
};

void GSCpuBackend::GpuStateDeleter::operator()(GpuState *state) const
{
    delete state;
}

// display.renderer = gpu | hybrid | software (PS2X_GS_RENDERER).
bool GSCpuBackend::GpuSetting()
{
    static const bool wanted = []
    {
        const char *value = ps2x::setting("PS2X_GS_RENDERER", "display.renderer");
        return value != nullptr && (std::strcmp(value, "gpu") == 0 || std::strcmp(value, "hybrid") == 0);
    }();
    return wanted;
}

// The card alone: gpu, unless the two are to be compared.
bool GSCpuBackend::GpuOnlySetting()
{
    static const bool wanted = []
    {
        const char *value = ps2x::setting("PS2X_GS_RENDERER", "display.renderer");
        return value != nullptr && std::strcmp(value, "gpu") == 0 && std::getenv("PS2X_GPU_COMPARE") == nullptr;
    }();
    return wanted;
}

// Whether the card draws the copy; makes the renderer the first time.
bool GSCpuBackend::GpuReady()
{
    if (!m_gpuAsked)
    {
        m_gpuAsked = true;
        if (GpuSetting() && ps2xGsDrawsInBackground(m_vram))
        {
            if (std::unique_ptr<GSGpu::Renderer> renderer = GSGpu::Renderer::Create())
            {
                m_gpu.reset(new GpuState());
                m_gpu->renderer = std::move(renderer);
                m_cardOnly = GpuOnlySetting();
            }
            else
            {
                std::fprintf(stderr, "[gs-gpu] no graphics context for the renderer: drawing in software\n");
            }
        }
    }
    return m_gpu != nullptr;
}

uint64_t GSCpuBackend::GpuKey(const GSMem::SurfaceLayout &layout, SurfaceScale scale, int depthKind) const
{
    const uint64_t key = shadowKeyOf(layout, scale.x, scale.y);
    // Depth as 24 bits and as 32 in the same pages is not the same numbers.
    return depthKind < 0 ? key : (mixKey(key, 0xD0u + static_cast<uint64_t>(depthKind)) | (1ull << 63));
}

// Whether the surface with this key has what VRAM has in a page.
bool GSCpuBackend::PageGood(uint32_t page, uint64_t key)
{
    if (m_shadowKey[page] != key)
        return false;
    if (m_shadowSerial[page] == m_pageSerial[page])
        return true;
    if (m_pageDrawSerial[page] > m_shadowSerial[page])
        return false;
    // An image transfer since, then: one that wrote here, or one beside
    // the page that was only taken to reach into it (an image whose base
    // is not at the start of a page is, for the page after its last)?
    // The blocks know.
    for (uint32_t block = page * 32u; block < page * 32u + 32u; ++block)
        if (m_blockUploadSerial[block] > m_shadowSerial[page])
            return false;
    m_shadowSerial[page] = m_pageSerial[page];
    return true;
}

bool GSCpuBackend::PagesStale(const GSMem::SurfaceLayout &layout, uint64_t key, int x0, int y0, int x1, int y1)
{
    bool stale = false;
    forEachPageOf(layout, x0, y0, x1, y1, [&](uint32_t page, uint32_t, uint32_t)
                  { stale = stale || !PageGood(page, key); });
    return stale;
}

void GSCpuBackend::MarkPages(const GSMem::SurfaceLayout &layout, uint64_t key, int x0, int y0, int x1, int y1)
{
    forEachPageOf(layout, x0, y0, x1, y1, [&](uint32_t page, uint32_t, uint32_t)
                  {
                      m_shadowSerial[page] = m_pageSerial[page];
                      m_shadowKey[page] = key;
                  });
}

// The card's surface for a buffer, with at least `rows` rows.
uint32_t GSCpuBackend::GpuSurface(const GSMem::SurfaceLayout &layout, SurfaceScale scale, int depthKind, int rows)
{
    GpuState &gpu = *m_gpu;
    const int pageRows = 1 << layout.pageHeightShift;
    rows = std::min(((std::max(rows, 1) + pageRows - 1) / pageRows) * pageRows, 2048);
    const int width = static_cast<int>(layout.pagesPerRow) << layout.pageWidthShift;
    const uint64_t key = GpuKey(layout, scale, depthKind);
    auto &recent = gpu.recent[depthKind < 0 ? 0 : 1];
    if (recent.key != key || !recent.surface)
    {
        recent.key = key;
        recent.surface = &gpu.surfaces[key];
    }
    GpuState::Surface &surface = *recent.surface;
    gpu.last = &surface;
    if (surface.id == 0u)
    {
        surface.id = gpu.renderer->NewSurface(depthKind < 0 ? GSGpu::SurfaceKind::Colour : GSGpu::SurfaceKind::Depth, width, rows, scale.x, scale.y);
        surface.width = width;
        surface.height = rows;
        surface.layout = layout;
        surface.scale = scale;
        surface.depthKind = depthKind;
    }
    else if (rows > surface.height)
    {
        gpu.renderer->Resize(surface.id, width, rows);
        surface.height = rows;
    }
    return surface.id;
}

// Brings the stale pages of these pixels of a buffer to its surface from
// VRAM. Nothing queued may still have those pages to write. With
// `overwritten`, the pixels are about to be written whatever they hold: a
// page that lies wholly among them has no need of what VRAM has.
void GSCpuBackend::GpuFill(const GSMem::SurfaceLayout &layout, SurfaceScale scale, int depthKind, int x0, int y0, int x1, int y1,
                           bool overwritten)
{
    if (!layout.table || (layout.unitBits != 32u && layout.unitBits != 16u) || (depthKind < 0 && layout.unitBits != 32u))
        return;
    GpuState &gpu = *m_gpu;
    const uint64_t key = GpuKey(layout, scale, depthKind);
    const int pageWidth = 1 << layout.pageWidthShift, pageHeight = 1 << layout.pageHeightShift;
    gpu.cells.clear();
    uint32_t firstRow = ~0u, lastRow = 0u, firstColumn = ~0u, lastColumn = 0u;
    forEachPageOf(layout, x0, y0, x1, y1, [&](uint32_t page, uint32_t row, uint32_t column)
                  {
                      if (PageGood(page, key))
                          return;
                      const int left = static_cast<int>(column) * pageWidth, top = static_cast<int>(row) * pageHeight;
                      if (overwritten && left >= x0 && top >= y0 && left + pageWidth - 1 <= x1 && top + pageHeight - 1 <= y1)
                      {
                          m_shadowSerial[page] = m_pageSerial[page];
                          m_shadowKey[page] = key;
                          return;
                      }
                      gpu.cells.push_back({page, row, column});
                      firstRow = std::min(firstRow, row);
                      lastRow = std::max(lastRow, row);
                      firstColumn = std::min(firstColumn, column);
                      lastColumn = std::max(lastColumn, column);
                  });
    if (gpu.cells.empty())
        return;
    const uint32_t surface = GpuSurface(layout, scale, depthKind, static_cast<int>(lastRow + 1u) * pageHeight);
    auto fill = [&](int left, int top, int width, int height)
    {
        const size_t count = static_cast<size_t>(width) * static_cast<size_t>(height);
        if (depthKind < 0)
        {
            gpu.pixels.resize(count);
            uint32_t *out = gpu.pixels.data();
            for (int y = top; y < top + height; ++y)
            {
                const FastPipeline::Row row = FastPipeline::rowOf(layout, y);
                for (int x = left; x < left + width; ++x)
                    *out++ = read32(m_vram, FastPipeline::unitIn(layout, row, x));
            }
            gpu.renderer->FillColour(surface, left, top, width, height, gpu.pixels.data());
        }
        else
        {
            gpu.depths.resize(count);
            float *out = gpu.depths.data();
            const bool wide = layout.unitBits == 32u;
            const bool z24 = depthKind == static_cast<int>(DepthKind::Z24);
            for (int y = top; y < top + height; ++y)
            {
                const FastPipeline::Row row = FastPipeline::rowOf(layout, y);
                for (int x = left; x < left + width; ++x)
                {
                    const uint32_t unit = FastPipeline::unitIn(layout, row, x);
                    if (!wide)
                        *out++ = static_cast<float>(read16(m_vram, unit)) * (1.0f / 65536.0f);
                    else if (z24)
                        *out++ = static_cast<float>(read32(m_vram, unit) & 0x00FFFFFFu) * (1.0f / 16777216.0f);
                    else
                        *out++ = static_cast<float>(static_cast<double>(read32(m_vram, unit)) * (1.0 / 4294967296.0));
                }
            }
            gpu.renderer->FillDepth(surface, left, top, width, height, gpu.depths.data());
        }
        ++gpu.statFills;
    };
    // Usually all of a rectangle of pages: one piece then.
    if (gpu.cells.size() == static_cast<size_t>(lastRow - firstRow + 1u) * static_cast<size_t>(lastColumn - firstColumn + 1u))
    {
        fill(static_cast<int>(firstColumn) * pageWidth, static_cast<int>(firstRow) * pageHeight,
             static_cast<int>(lastColumn - firstColumn + 1u) * pageWidth, static_cast<int>(lastRow - firstRow + 1u) * pageHeight);
    }
    else
    {
        for (const GpuState::Cell &cell : gpu.cells)
            fill(static_cast<int>(cell.column) * pageWidth, static_cast<int>(cell.row) * pageHeight, pageWidth, pageHeight);
    }
    for (const GpuState::Cell &cell : gpu.cells)
    {
        m_shadowSerial[cell.page] = m_pageSerial[cell.page];
        m_shadowKey[cell.page] = key;
    }
    gpu.statFilledPages += gpu.cells.size();
}

void GSCpuBackend::SetCardAhead(const GSMem::SurfaceLayout &layout, int x0, int y0, int x1, int y1)
{
    forEachPageOf(layout, x0, y0, x1, y1, [&](uint32_t page, uint32_t, uint32_t)
                  {
                      if (!m_cardAhead[page])
                      {
                          m_cardAhead[page] = 1u;
                          ++m_cardAheadCount;
                      }
                  });
}

namespace
{
    // fn(block) for each 256-byte block of VRAM pixels (x0, y0)..(x1, y1)
    // of a surface have a part in (gs_cpu_backend.cpp's forEachBlock).
    template <typename Fn>
    void forEachImageBlock(const GSMem::SurfaceLayout &layout, int x0, int y0, int x1, int y1, Fn &&fn)
    {
        const uint32_t tileW = layout.unitBits == 32u ? 8u : (layout.unitBits == 4u ? 32u : 16u);
        const uint32_t tileH = layout.unitBits >= 16u ? 8u : 16u;
        const uint32_t firstX = static_cast<uint32_t>(std::max(x0, 0)) & ~(tileW - 1u);
        const uint32_t firstY = static_cast<uint32_t>(std::max(y0, 0)) & ~(tileH - 1u);
        for (uint32_t y = firstY; y <= static_cast<uint32_t>(std::max(y1, 0)); y += tileH)
            for (uint32_t x = firstX; x <= static_cast<uint32_t>(std::max(x1, 0)); x += tileW)
                fn(((layout.unit(x, y) * layout.unitBits) >> 11) & 0x3FFFu);
    }

    // Whether two buffers lay the same kind of pixel out in rows of the same
    // width, starting at the same column: what one holds means something to
    // the other. (A frame buffer and a texture, a narrow buffer and a wide
    // one, or two that start in different columns share pages at
    // different times: each finds the other's pixels scrambled, and the
    // game cannot be counting on them.)
    inline bool sameShape(const GSMem::SurfaceLayout &a, bool aDepth, const GSMem::SurfaceLayout &b, bool bDepth)
    {
        return aDepth == bDepth && a.unitBits == b.unitBits && a.pagesPerRow == b.pagesPerRow && a.pageWidthShift == b.pageWidthShift &&
               a.pageHeightShift == b.pageHeightShift &&
               (a.basePage + kPageCount - b.basePage % kPageCount) % std::max<uint32_t>(a.pagesPerRow, 1u) == 0u;
    }

    // The same, as a number to keep: for telling later whether an image
    // stored in a page was pixels of the kind a buffer there holds.
    inline uint32_t shapeOf(const GSMem::SurfaceLayout &layout, bool depth)
    {
        const uint32_t columns = std::max<uint32_t>(layout.pagesPerRow, 1u);
        return static_cast<uint32_t>(layout.unitBits) | (columns << 8) | ((layout.basePage % columns) << 16) |
               (static_cast<uint32_t>(layout.pageWidthShift) << 24) | (depth ? 0x80000000u : 0u);
    }
}

// Before the card draws into pages of a buffer that are not good for its
// surface: the ones only another surface has. If the primitive writes every
// pixel of a page, or the other buffer's pixels mean nothing to this one,
// the page simply changes hands; otherwise VRAM is brought up to date, for
// GpuFill to take them from.
void GSCpuBackend::GpuAdopt(const GSMem::SurfaceLayout &layout, uint64_t key, int depthKind, int x0, int y0, int x1, int y1, bool overwritten)
{
    GpuState &gpu = *m_gpu;
    const int pageWidth = 1 << layout.pageWidthShift, pageHeight = 1 << layout.pageHeightShift;
    const uint32_t shape = shapeOf(layout, depthKind >= 0);
    gpu.pages.clear();
    forEachPageOf(layout, x0, y0, x1, y1, [&](uint32_t page, uint32_t row, uint32_t column)
                  {
                      if (PageGood(page, key))
                          return;
                      if (!m_cardAhead[page])
                      {
                          // VRAM's, then. If the last thing stored there was
                          // an image of another kind (a texture, where the
                          // game keeps textures and draws by turns), its
                          // bytes are no pixels of this buffer's: it has
                          // what it had.
                          if (m_pageUploadSerial[page] == m_pageSerial[page] && m_pageUploadShape[page] != shape && m_shadowKey[page] != 0u)
                          {
                              m_shadowSerial[page] = m_pageSerial[page];
                              m_shadowKey[page] = key;
                              ++gpu.statAdopted;
                          }
                          return;
                      }
                      const int left = static_cast<int>(column) * pageWidth, top = static_cast<int>(row) * pageHeight;
                      const bool whole = overwritten && left >= x0 && top >= y0 && left + pageWidth - 1 <= x1 && top + pageHeight - 1 <= y1;
                      const auto owner = gpu.surfaces.find(m_shadowKey[page]);
                      if (!whole && owner != gpu.surfaces.end() &&
                          sameShape(owner->second.layout, owner->second.depthKind >= 0, layout, depthKind >= 0))
                      {
                          gpu.pages.push_back(page);
                          return;
                      }
                      m_shadowSerial[page] = m_pageSerial[page];
                      m_shadowKey[page] = key;
                      ++gpu.statAdopted;
                  });
    if (!gpu.pages.empty())
    {
        gpu.why = GpuState::WhyOtherSurface;
        GpuDownloadPages(gpu.pages);
    }
}

// Brings VRAM the pages of these pixels that only the card has (other than
// those of the surface with `exceptKey`).
void GSCpuBackend::GpuDownload(const GSMem::SurfaceLayout &layout, int x0, int y0, int x1, int y1, int why, uint64_t exceptKey,
                               bool sameShapeOnly)
{
    if (!m_gpu || m_cardAheadCount == 0u || !layout.table)
        return;
    GpuState &gpu = *m_gpu;
    gpu.why = static_cast<GpuState::Why>(why);
    gpu.pages.clear();
    // By the block: which pages the pixels are really in, wherever the
    // surface starts.
    uint32_t last = ~0u;
    forEachImageBlock(layout, x0, y0, x1, y1, [&](uint32_t block)
                      {
                          const uint32_t page = (block >> 5) % kPageCount;
                          if (page == last)
                              return;
                          last = page;
                          if (!m_cardAhead[page] || m_shadowKey[page] == exceptKey)
                              return;
                          if (sameShapeOnly)
                          {
                              // For a texture: a buffer of another kind has
                              // nothing in these bytes that is texels.
                              const auto owner = gpu.surfaces.find(m_shadowKey[page]);
                              if (owner != gpu.surfaces.end() && !sameShape(owner->second.layout, false, layout, false))
                                  return;
                          }
                          gpu.pages.push_back(page);
                      });
    if (!gpu.pages.empty())
        GpuDownloadPages(gpu.pages);
}

void GSCpuBackend::GpuDownloadAll()
{
    if (!m_gpu || m_cardAheadCount == 0u)
        return;
    GpuState &gpu = *m_gpu;
    gpu.pages.clear();
    for (uint32_t page = 0; page < kPageCount; ++page)
        if (m_cardAhead[page])
            gpu.pages.push_back(page);
    gpu.why = GpuState::WhyEverything;
    GpuDownloadPages(gpu.pages);
}

// The pages listed, each from the surface that has it: read back at the
// game's size and stored in VRAM the way the buffer lays its pixels out.
// Waits for the card.
void GSCpuBackend::GpuDownloadPages(std::vector<uint32_t> &pages)
{
    GpuState &gpu = *m_gpu;
    std::sort(pages.begin(), pages.end());
    pages.erase(std::unique(pages.begin(), pages.end()), pages.end());
    while (!pages.empty())
    {
        // Everything listed that the same surface has, in one piece.
        const uint64_t key = m_shadowKey[pages.back()];
        const auto found = gpu.surfaces.find(key);
        std::vector<uint32_t> mine;
        for (auto page = pages.begin(); page != pages.end();)
        {
            if (m_shadowKey[*page] == key)
            {
                mine.push_back(*page);
                page = pages.erase(page);
            }
            else
            {
                ++page;
            }
        }
        auto release = [&]
        {
            for (uint32_t page : mine)
            {
                if (m_cardAhead[page])
                {
                    m_cardAhead[page] = 0u;
                    --m_cardAheadCount;
                }
            }
        };
        if (found == gpu.surfaces.end())
        {
            release();
            continue;
        }
        const GpuState::Surface &surface = found->second;
        const GSMem::SurfaceLayout &layout = surface.layout;
        const int pageWidth = 1 << layout.pageWidthShift, pageHeight = 1 << layout.pageHeightShift;
        const uint32_t columns = std::max<uint32_t>(layout.pagesPerRow, 1u);
        uint32_t firstRow = ~0u, lastRow = 0u, firstColumn = ~0u, lastColumn = 0u;
        auto place = [&](uint32_t page, uint32_t &row, uint32_t &column)
        {
            const uint32_t offset = (page + kPageCount - layout.basePage % kPageCount) % kPageCount;
            row = offset / columns;
            column = offset % columns;
        };
        for (uint32_t page : mine)
        {
            uint32_t row, column;
            place(page, row, column);
            firstRow = std::min(firstRow, row);
            lastRow = std::max(lastRow, row);
            firstColumn = std::min(firstColumn, column);
            lastColumn = std::max(lastColumn, column);
        }
        const int left = static_cast<int>(firstColumn) * pageWidth, top = static_cast<int>(firstRow) * pageHeight;
        const int width = static_cast<int>(lastColumn - firstColumn + 1u) * pageWidth;
        const int height = std::min(static_cast<int>(lastRow - firstRow + 1u) * pageHeight, surface.height - top);
        std::vector<uint8_t> got;
        uint32_t gotWidth = 0, gotHeight = 0;
        GSGpu::Reading reading = height > 0 ? gpu.renderer->ReadBack(surface.id, left, top, width, height) : GSGpu::Reading();
        if (reading && reading.Wait(got, gotWidth, gotHeight) && gotWidth == static_cast<uint32_t>(width))
        {
            const bool depth = surface.depthKind >= 0;
            for (uint32_t page : mine)
            {
                uint32_t pageRow, pageColumn;
                place(page, pageRow, pageColumn);
                for (int py = 0; py < pageHeight; ++py)
                {
                    const int y = static_cast<int>(pageRow) * pageHeight + py;
                    if (y - top >= static_cast<int>(gotHeight))
                        break;
                    const FastPipeline::Row row = FastPipeline::rowOf(layout, y);
                    const uint8_t *from = got.data() + (static_cast<size_t>(y - top) * gotWidth +
                                                        static_cast<size_t>(static_cast<int>(pageColumn) * pageWidth - left)) * 4u;
                    for (int px = 0; px < pageWidth; ++px, from += 4)
                    {
                        const uint32_t unit = FastPipeline::unitIn(layout, row, static_cast<int>(pageColumn) * pageWidth + px);
                        if (!depth)
                        {
                            std::memcpy(m_vram + ((unit * 4u) & (kVramMask - 3u)), from, 4u);
                            continue;
                        }
                        float fraction;
                        std::memcpy(&fraction, from, sizeof(fraction));
                        if (layout.unitBits == 16u)
                        {
                            const uint16_t z = static_cast<uint16_t>(std::min(static_cast<double>(fraction) * 65536.0 + 0.5, 65535.0));
                            std::memcpy(m_vram + ((unit * 2u) & (kVramMask - 1u)), &z, sizeof(z));
                        }
                        else if (surface.depthKind == static_cast<int>(DepthKind::Z24))
                        {
                            const uint32_t z = static_cast<uint32_t>(std::min(static_cast<double>(fraction) * 16777216.0 + 0.5, 16777215.0));
                            uint32_t there;
                            std::memcpy(&there, m_vram + ((unit * 4u) & (kVramMask - 3u)), sizeof(there));
                            there = (there & 0xFF000000u) | z;
                            std::memcpy(m_vram + ((unit * 4u) & (kVramMask - 3u)), &there, sizeof(there));
                        }
                        else
                        {
                            const uint32_t z = static_cast<uint32_t>(std::min(static_cast<double>(fraction) * 4294967296.0 + 0.5, 4294967295.0));
                            std::memcpy(m_vram + ((unit * 4u) & (kVramMask - 3u)), &z, sizeof(z));
                        }
                    }
                }
            }
            // VRAM has changed under whatever was read from these pages.
            ++m_writeSerial;
            for (uint32_t page : mine)
                m_pageVramDrawSerial[page] = m_writeSerial;
            ++gpu.statDownloads;
            ++gpu.statWhy[gpu.why];
            gpu.statDownloadedPages += mine.size();
        }
        release();
    }
}

// An image is about to be stored in these pixels of VRAM. A page of them
// that only the card has is brought to VRAM first if the image is pixels
// of the kind its surface holds (part of a frame buffer redrawn by hand),
// and otherwise given up: the game is using that memory for something
// else now.
void GSCpuBackend::GpuUploading(const GSMem::SurfaceLayout &layout, int x0, int y0, int x1, int y1)
{
    if (!m_gpu || m_cardAheadCount == 0u || !layout.table)
        return;
    GpuState &gpu = *m_gpu;
    gpu.pages.clear();
    // A page the card alone has: wanted back first only if the image is
    // pixels of the kind the surface holds and leaves part of the page as
    // it was. One it covers whole, or of another kind, is just given up.
    auto settle = [&](uint32_t page, bool whole)
    {
        if (!m_cardAhead[page])
            return;
        const auto owner = gpu.surfaces.find(m_shadowKey[page]);
        if (!whole && owner != gpu.surfaces.end() && sameShape(owner->second.layout, owner->second.depthKind >= 0, layout, false))
        {
            gpu.pages.push_back(page);
            return;
        }
        m_cardAhead[page] = 0u;
        --m_cardAheadCount;
        ++gpu.statDropped;
    };
    if (!layout.spillsIntoNextPage)
    {
        const int pageWidth = 1 << layout.pageWidthShift, pageHeight = 1 << layout.pageHeightShift;
        forEachPageOf(layout, x0, y0, x1, y1, [&](uint32_t page, uint32_t row, uint32_t column)
                      {
                          const int left = static_cast<int>(column) * pageWidth, top = static_cast<int>(row) * pageHeight;
                          settle(page, left >= x0 && top >= y0 && left + pageWidth - 1 <= x1 && top + pageHeight - 1 <= y1);
                      });
    }
    else
    {
        // By the block, as MarkUploaded does: an image that does not start
        // with a page may be taken to reach one it has nothing in.
        uint32_t last = ~0u;
        forEachImageBlock(layout, x0, y0, x1, y1, [&](uint32_t block)
                          {
                              const uint32_t page = (block >> 5) % kPageCount;
                              if (page != last)
                                  settle(page, false);
                              last = page;
                          });
    }
    if (!gpu.pages.empty())
    {
        gpu.why = GpuState::WhyImage;
        GpuDownloadPages(gpu.pages);
    }
}

// From MarkUploaded, for each block an image is stored in.
void GSCpuBackend::NoteUploadShape(const GSMem::SurfaceLayout &layout, uint32_t block)
{
    const uint32_t page = (block >> 5) % kPageCount;
    m_pageUploadSerial[page] = m_writeSerial;
    m_pageUploadShape[page] = shapeOf(layout, false);
}

void GSCpuBackend::FlushForVram(const char *why)
{
    FlushPrimitives(why);
    GpuDownloadAll();
}

void GSCpuBackend::GpuForgetTexture(uint32_t texture)
{
    if (m_gpu && texture != 0u)
        m_gpu->renderer->Release(texture);
}

// Every texture of the card's made from VRAM or from a decoded copy: the
// copies are going.
void GSCpuBackend::GpuForgetTextures()
{
    if (!m_gpu)
        return;
    for (auto &entry : m_decodedTextures)
    {
        GpuForgetTexture(entry.second.gpuTexture);
        entry.second.gpuTexture = 0u;
    }
    for (auto &entry : m_gpu->textures)
        GpuForgetTexture(entry.second.id);
    m_gpu->textures.clear();
}

void GSCpuBackend::GpuFlush()
{
    if (m_gpu)
        m_gpu->renderer->Flush();
}

// The texels a primitive can sample, decoded from VRAM as it is now.
void GSCpuBackend::GpuTextureFromVram(FastPipeline &p, GSGpu::Pipeline &state)
{
    GpuState &gpu = *m_gpu;
    // The texels the primitive can read: where its coordinates point, or
    // with a region wrap mode where those are taken to. REGION_CLAMP holds
    // them between MIN and MAX, so it is that part of the range; with
    // REGION_REPEAT they are MAX with the bits of MIN free, so from MAX to
    // MAX | MIN. (Not all of the texture: Drakengard's font is a glyph at
    // a time in a texture it calls 1024 by 1024, clamped to the glyph, and
    // a million texels were decoded for each letter of a loading screen.)
    // Whole blocks of 32, so that a neighbouring primitive finds its texels
    // here too.
    auto reach = [](uint8_t wrap, int first, int last, int low, int high, int size, int &from, int &to)
    {
        from = first;
        to = last;
        if (wrap == 2u)
        {
            from = clampInt(first, low, std::max(low, high));
            to = clampInt(last, low, std::max(low, high));
        }
        else if (wrap == 3u)
        {
            from = high;
            to = high | low;
        }
        from = clampInt(from, 0, size - 1);
        to = clampInt(to, from, size - 1);
    };
    int x0, x1, y0, y1;
    reach(p.wrapU, p.sampledU0, p.sampledU1, p.minU, p.maxU, p.texW, x0, x1);
    reach(p.wrapV, p.sampledV0, p.sampledV1, p.minV, p.maxV, p.texH, y0, y1);
    x0 = clampInt(x0, 0, p.texW - 1) & ~31;
    y0 = clampInt(y0, 0, p.texH - 1) & ~31;
    x1 = std::min(clampInt(x1, 0, p.texW - 1) | 31, p.texW - 1);
    y1 = std::min(clampInt(y1, 0, p.texH - 1) | 31, p.texH - 1);
    // With REPEAT, coordinates that run past an end of the texture take
    // texels from the far side: a filtered sprite that starts at the first
    // texel reads the last one beside it. That is a few texels, not the
    // whole axis: they are kept as the coordinates have them, from below 0
    // or past the size, and the shader looks them up the same way. (The
    // font again: 1024 by 1024 by name, a glyph at a time in fact, filtered
    // from its first texel, and decoded whole for every letter.)
    auto window = [](int reach0, int reach1, int size, int &from, int &to)
    {
        if (reach1 < reach0)
            return;
        const int first = reach0 & ~31, last = reach1 | 31;
        if (last - first + 1 < size)
        {
            from = first;
            to = last;
        }
    };
    window(p.reachU0, p.reachU1, p.texW, x0, x1);
    window(p.reachV0, p.reachV1, p.texH, y0, y1);
    // A window as the one to four rectangles of the texture it is.
    const auto parts = [&p](int wx0, int wy0, int wx1, int wy1, auto &&use)
    {
        const auto split = [](int a, int b, int size, int out[2][2])
        {
            if (a >= 0 && b < size)
            {
                out[0][0] = a;
                out[0][1] = b;
                return 1;
            }
            if (b - a + 1 >= size)
            {
                out[0][0] = 0;
                out[0][1] = size - 1;
                return 1;
            }
            const int first = a & (size - 1), last = b & (size - 1);
            if (first <= last)
            {
                out[0][0] = first;
                out[0][1] = last;
                return 1;
            }
            out[0][0] = first;
            out[0][1] = size - 1;
            out[1][0] = 0;
            out[1][1] = last;
            return 2;
        };
        int across[2][2], down[2][2];
        const int columns = split(wx0, wx1, p.texW, across), rows = split(wy0, wy1, p.texH, down);
        for (int row = 0; row < rows; ++row)
            for (int column = 0; column < columns; ++column)
                use(across[column][0], down[row][0], across[column][1], down[row][1]);
    };
    const auto changedSince = [&](const GpuState::Texture &held)
    {
        bool drawn = false, uploaded = false;
        parts(held.x0, held.y0, held.x1, held.y1, [&](int px0, int py0, int px1, int py1)
              { drawn = drawn || DrawnSince(p.texture, px0, py0, px1, py1, held.serial); });
        if (!drawn && held.checkedUpload != m_lastUploadSerial)
            parts(held.x0, held.y0, held.x1, held.y1, [&](int px0, int py0, int px1, int py1)
                  { uploaded = uploaded || UploadedSince(p.texture, px0, py0, px1, py1, held.serial); });
        return drawn || uploaded;
    };

    uint64_t key = mixKey(ConversionKey(p), (static_cast<uint64_t>(p.tbp0) << 32) | (static_cast<uint64_t>(p.tbw) << 8) | p.texPsm);
    key = mixKey(key, (static_cast<uint64_t>(p.texW) << 16) | static_cast<uint64_t>(p.texH));
    GpuState::Texture &texture = gpu.textures[key];
    // The copy has what is wanted if it reaches as far each way, or holds
    // the whole of that axis: a window that starts below 0 is inside a
    // whole texture, whatever its numbers say.
    const auto holds = [](int held0, int held1, int want0, int want1, int size)
    {
        return (held0 <= want0 && held1 >= want1) || (held0 == 0 && held1 == size - 1);
    };
    bool good = texture.id != 0u && holds(texture.x0, texture.x1, x0, x1, p.texW) && holds(texture.y0, texture.y1, y0, y1, p.texH) &&
                !changedSince(texture);
    if (good)
        texture.checkedUpload = m_lastUploadSerial;
    if (!good)
    {
        const bool had = texture.id != 0u;
        const int oldX0 = texture.x0, oldY0 = texture.y0, oldX1 = texture.x1, oldY1 = texture.y1;
        if (had)
        {
            x0 = std::min(x0, texture.x0);
            y0 = std::min(y0, texture.y0);
            x1 = std::max(x1, texture.x1);
            y1 = std::max(y1, texture.y1);
        }
        // (What has grown to the size of the texture is the texture.)
        if (x1 - x0 + 1 >= p.texW)
        {
            x0 = 0;
            x1 = p.texW - 1;
        }
        if (y1 - y0 + 1 >= p.texH)
        {
            y0 = 0;
            y1 = p.texH - 1;
        }
        parts(x0, y0, x1, y1, [&](int px0, int py0, int px1, int py1)
              {
                  FlushPrimitivesTouching(p.texture, px0, py0, px1, py1, false, "texture");
                  GpuDownload(p.texture, px0, py0, px1, py1, GpuState::WhyTexture, 0u, true);
              });
        const int width = x1 - x0 + 1, height = y1 - y0 + 1;
        gpu.pixels.resize(static_cast<size_t>(width) * static_cast<size_t>(height));
        uint32_t *out = gpu.pixels.data();
        const int wrapAcross = p.texW - 1, wrapDown = p.texH - 1;
        for (int v = y0; v <= y1; ++v)
            for (int u = x0; u <= x1; ++u)
                *out++ = p.fetch(u & wrapAcross, v & wrapDown);
        const uint64_t hash = hashBytes(reinterpret_cast<const uint8_t *>(gpu.pixels.data()), gpu.pixels.size() * sizeof(uint32_t),
                                        0x452821E638D01377ull);
        uint32_t most = 0u;
        for (uint32_t pixel : gpu.pixels)
            most = std::max(most, pixel >> 24);
        texture.maxAlpha = static_cast<uint8_t>(most);
        ++gpu.statDecodes;
        // The same texels as before, as often as not: something next to
        // them was written.
        if (!had || hash != texture.hash || x0 != oldX0 || y0 != oldY0 || x1 != oldX1 || y1 != oldY1)
        {
            if (!had)
                texture.id = gpu.renderer->NewSurface(GSGpu::SurfaceKind::Texture, width, height, 1, 1);
            gpu.renderer->SetTexture(texture.id, width, height, gpu.pixels.data());
            ++gpu.statUploads;
        }
        texture.x0 = x0;
        texture.y0 = y0;
        texture.x1 = x1;
        texture.y1 = y1;
        texture.hash = hash;
        texture.serial = m_writeSerial;
        texture.checkedUpload = m_lastUploadSerial;
    }
    texture.lastUsed = ++gpu.clock;
    gpu.textureMaxAlpha = texture.maxAlpha;
    state.texture = texture.id;
    state.textureMode = GSGpu::TextureTexels;
    state.texOrigin[0] = static_cast<int16_t>(texture.x0);
    state.texOrigin[1] = static_cast<int16_t>(texture.y0);
    state.texExtent[0] = static_cast<int16_t>(texture.x1 - texture.x0 + 1);
    state.texExtent[1] = static_cast<int16_t>(texture.y1 - texture.y0 + 1);

    // Each CLUT a texture is drawn with is another of these. The ones not
    // drawn with for longest go once there are a great many.
    if (gpu.textures.size() > 4096u)
    {
        const uint64_t keep = gpu.clock > 200000u ? gpu.clock - 200000u : 0u;
        for (auto entry = gpu.textures.begin(); entry != gpu.textures.end();)
        {
            if (entry->second.lastUsed < keep)
            {
                GpuForgetTexture(entry->second.id);
                entry = gpu.textures.erase(entry);
            }
            else
            {
                ++entry;
            }
        }
    }
}

void GSCpuBackend::GpuTexture(PreparedPrimitive &q, const GSContext &ctx, GSGpu::Pipeline &state)
{
    FastPipeline &p = q.pipeline;
    GpuState &gpu = *m_gpu;
    state.linear = p.linear ? 1u : 0u;
    state.tfx = p.tfx;
    state.tcc = p.tcc ? 1u : 0u;
    state.wrapU = p.wrapU;
    state.wrapV = p.wrapV;
    state.region[0] = static_cast<int16_t>(p.minU);
    state.region[1] = static_cast<int16_t>(p.maxU);
    state.region[2] = static_cast<int16_t>(p.minV);
    state.region[3] = static_cast<int16_t>(p.maxV);
    state.texSize[0] = static_cast<int16_t>(p.texW);
    state.texSize[1] = static_cast<int16_t>(p.texH);

    // A frame buffer the card has, laid out as the texture is, with the
    // pages the primitive samples good.
    // The buffer itself, or a texture that starts some rows down a buffer
    // laid out the same way (a game reads part of its frame so). The
    // surface being drawn into as well: the renderer reads a copy of it.
    if ((p.texKind == TextureKind::Direct32 || p.texKind == TextureKind::Direct24) && (ctx.tex0.tbp0 & 31u) == 0u &&
        !p.texture.spillsIntoNextPage)
    {
        const SurfaceScale from = ScaleOf(ctx.tex0.tbp0 >> 5, ctx.tex0.tbw);
        uint64_t key = GpuKey(p.texture, from, -1);
        const GpuState::Surface *source = nullptr;
        int rowsDown = 0;
        if (const auto found = gpu.surfaces.find(key); found != gpu.surfaces.end())
        {
            source = &found->second;
        }
        else
        {
            for (const auto &entry : gpu.surfaces)
            {
                const GpuState::Surface &surface = entry.second;
                if (surface.depthKind >= 0 || !sameShape(surface.layout, false, p.texture, false))
                    continue;
                const uint32_t pagesDown = (p.texture.basePage + kPageCount - surface.layout.basePage % kPageCount) % kPageCount;
                const int rows = static_cast<int>(pagesDown / std::max<uint32_t>(surface.layout.pagesPerRow, 1u)) << surface.layout.pageHeightShift;
                if (rows > 0 && rows < surface.height)
                {
                    source = &surface;
                    rowsDown = rows;
                    key = entry.first;
                    break;
                }
            }
        }
        if (source)
        {
            const int lastU = std::min(p.sampledU1, source->width - 1), lastV = std::min(p.sampledV1, source->height - 1 - rowsDown);
            if (p.sampledU0 <= lastU && p.sampledV0 <= lastV &&
                !PagesStale(source->layout, key, p.sampledU0, rowsDown + std::max(p.sampledV0 - 2, 0), lastU, rowsDown + lastV))
            {
                state.texture = source->id;
                state.textureMode = GSGpu::TextureSurface;
                state.texOrigin[1] = static_cast<int16_t>(rowsDown);
                state.texAlphaFixed = p.texKind == TextureKind::Direct24 ? 1u : 0u;
                state.ta0 = p.texa.ta0;
                state.aem = p.texa.aem ? 1u : 0u;
                if (q.sprite)
                {
                    // Down to the last row the sprite names and no further:
                    // filtering reaches half a texel past it, and below a
                    // frame buffer's picture the surface may hold anything.
                    // (Not across: a picture copied in strips side by side
                    // has the next strip's texels there, and stopping short
                    // of them leaves a seam at every strip.)
                    state.texExtent[1] = static_cast<int16_t>(std::min(p.texH, std::max(static_cast<int>(std::ceil(std::max(q.tv0, q.tv1))), 1)));
                }
                // A frame buffer's alpha is whatever was drawn; one read as 24 bits has TEXA's.
                gpu.textureMaxAlpha = p.texKind == TextureKind::Direct24 ? p.texa.ta0 : 255;
                ++gpu.statSurfaceTextures;
                return;
            }
        }
    }

    if (p.decodedCopy)
    {
        DecodedTexture &copy = *p.decodedCopy;
        if (copy.gpuTexture == 0u)
        {
            copy.gpuTexture = gpu.renderer->NewSurface(GSGpu::SurfaceKind::Texture, copy.width, copy.height, 1, 1);
            gpu.renderer->SetTexture(copy.gpuTexture, copy.width, copy.height, copy.pixels.data());
            uint32_t most = 0u;
            for (uint32_t pixel : copy.pixels)
                most = std::max(most, pixel >> 24);
            copy.gpuMaxAlpha = static_cast<uint8_t>(most);
            ++gpu.statUploads;
        }
        gpu.textureMaxAlpha = copy.gpuMaxAlpha;
        state.texture = copy.gpuTexture;
        state.textureMode = GSGpu::TextureTexels;
        state.texExtent[0] = static_cast<int16_t>(copy.width);
        state.texExtent[1] = static_cast<int16_t>(copy.height);
        ++gpu.statCopyTextures;
        return;
    }

    GpuTextureFromVram(p, state);
    ++gpu.statVramTextures;
}

// The primitive just prepared for the raster pool, to the card as well.
// Before the pool's copy is queued: decoding a texture from VRAM may have to
// wait for what is queued already.
void GSCpuBackend::GpuDraw(const GSPrimitiveBatch &batch, PreparedPrimitive &q, SurfaceScale scale)
{
    GpuState &gpu = *m_gpu;
    FastPipeline &p = q.pipeline;
    const GSContext &ctx = batch.state.context;
    const bool depthUsed = !p.zmask || p.ztst >= 2u;
    const int rows = std::max(q.maxY + 1, std::min(p.scissorY1 + 1, 512));

    GSGpu::Pipeline state{};
    state.colour = GpuSurface(p.frame, scale, -1, rows);
    const GpuState::Surface &target = *gpu.last;
    if (depthUsed)
        state.depth = GpuSurface(p.depth, scale, static_cast<int>(p.depthKind), rows);
    state.scissor[0] = static_cast<int16_t>(clampInt(p.scissorX0, 0, target.width - 1));
    state.scissor[1] = static_cast<int16_t>(clampInt(p.scissorY0, 0, target.height - 1));
    state.scissor[2] = static_cast<int16_t>(clampInt(p.scissorX1, 0, target.width - 1));
    state.scissor[3] = static_cast<int16_t>(clampInt(p.scissorY1, 0, target.height - 1));

    state.alphaTest = p.alphaTest ? 1u : 0u;
    state.atst = p.atst;
    state.aref = p.aref;
    state.afail = p.afail;
    state.ztst = p.ztst;
    state.zmask = p.zmask ? 1u : 0u;
    state.abe = p.abe ? 1u : 0u;
    state.pabe = p.pabe ? 1u : 0u;
    state.asel = p.asel;
    state.bsel = p.bsel;
    state.csel = p.csel;
    state.dsel = p.dsel;
    state.fix = p.fix;
    state.fba = p.fba ? 1u : 0u;
    state.frameAlphaOne = p.frameKind == FrameKind::C24 ? 1u : 0u;
    state.date = p.date ? 1u : 0u;
    state.datm = p.date && p.datm ? 1u : 0u;
    state.fge = p.fge ? 1u : 0u;
    state.fogR = p.fogR;
    state.fogG = p.fogG;
    state.fogB = p.fogB;
    // FBMSK a channel at a time, which is how games use it.
    state.writeRgb = (p.fbmsk & 0x00FFFFFFu) != 0x00FFFFFFu ? 1u : 0u;
    state.writeAlpha = (p.fbmsk & 0xFF000000u) != 0xFF000000u && p.frameKind != FrameKind::C24 ? 1u : 0u;
    state.shuffle = m_shuffle;
    if (p.tme)
        GpuTexture(q, ctx, state);

    // Depth as a fraction of what the buffer's format can hold. A Z past
    // that (the game's notion of nearest, for what it draws on top) is the
    // most the format holds, as it is once stored.
    const double depthLimit = p.depthKind == DepthKind::Z16 ? 65535.0 : (p.depthKind == DepthKind::Z24 ? 16777215.0 : 4294967295.0);
    const double depthScale = 1.0 / (depthLimit + 1.0);
    auto depthOf = [&](double z)
    {
        return depthUsed ? static_cast<float>(std::min(std::max(z, 0.0), depthLimit) * depthScale) : 0.0f;
    };
    GSGpu::Vertex out[6];
    size_t count = 0;
    auto add = [&](float x, float y, float z, const GSVertex &colour, float fog, float s, float t, float tq)
    {
        GSGpu::Vertex &v = out[count++];
        v.x = x;
        v.y = y;
        v.z = z;
        v.s = s;
        v.t = t;
        v.q = tq;
        v.r = colour.r;
        v.g = colour.g;
        v.b = colour.b;
        v.a = colour.a;
        v.fog = fog;
    };

    if (q.sprite)
    {
        const GSVertex &v1 = q.v[1];
        // Where it is in the game's pixels; one put between them is drawn
        // there, in the surface's own.
        const float x0 = q.placed ? q.sideX0 : static_cast<float>(q.originX), y0 = static_cast<float>(q.originY);
        const float x1 = q.placed ? q.sideX1 : x0 + q.spanX, y1 = y0 + q.spanY;
        float tu0 = q.placed ? q.sideU0 : q.tu0, tv0 = q.tv0, tu1 = q.placed ? q.sideU1 : q.tu1, tv1 = q.tv1;
        // The frame being squeezed into a field (see ScaleUp): the same
        // lines of it every time.
        if (scale.y > scale.x && p.tme && std::fabs(tv1 - tv0) >= q.spanY * 1.5f)
        {
            const float first = std::min(tv0, tv1);
            const float odd = first - 2.0f * std::floor(first * 0.5f);
            tv0 -= odd;
            tv1 -= odd;
        }
        const float z = depthOf(v1.z), fog = static_cast<float>(v1.fog);
        add(x0, y0, z, v1, fog, tu0, tv0, 1.0f);
        add(x1, y0, z, v1, fog, tu1, tv0, 1.0f);
        add(x0, y1, z, v1, fog, tu0, tv1, 1.0f);
        add(x1, y0, z, v1, fog, tu1, tv0, 1.0f);
        add(x1, y1, z, v1, fog, tu1, tv1, 1.0f);
        add(x0, y1, z, v1, fog, tu0, tv1, 1.0f);
    }
    else if (q.line)
    {
        // A pixel wide, from the middle of its first pixel to the middle
        // of its last and half a pixel past each.
        const GSVertex &v0 = q.v[0], &v1 = q.v[1];
        const float ax = static_cast<float>(q.lineX0) + 0.5f, ay = static_cast<float>(q.lineY0) + 0.5f;
        const float bx = static_cast<float>(q.lineX1) + 0.5f, by = static_cast<float>(q.lineY1) + 0.5f;
        float dx = bx - ax, dy = by - ay;
        const float length = std::sqrt(dx * dx + dy * dy);
        if (length > 0.0f)
        {
            dx *= 0.5f / length;
            dy *= 0.5f / length;
        }
        else
        {
            dx = 0.5f;
            dy = 0.0f;
        }
        const float nx = -dy, ny = dx;
        const GSVertex &c0 = q.gouraud ? v0 : v1;
        const float z0 = depthOf(v0.z), z1 = depthOf(v1.z);
        const float f0 = static_cast<float>(v0.fog), f1 = static_cast<float>(v1.fog);
        add(ax - dx + nx, ay - dy + ny, z0, c0, f0, 0.0f, 0.0f, 1.0f);
        add(ax - dx - nx, ay - dy - ny, z0, c0, f0, 0.0f, 0.0f, 1.0f);
        add(bx + dx + nx, by + dy + ny, z1, v1, f1, 0.0f, 0.0f, 1.0f);
        add(ax - dx - nx, ay - dy - ny, z0, c0, f0, 0.0f, 0.0f, 1.0f);
        add(bx + dx - nx, by + dy - ny, z1, v1, f1, 0.0f, 0.0f, 1.0f);
        add(bx + dx + nx, by + dy + ny, z1, v1, f1, 0.0f, 0.0f, 1.0f);
    }
    else
    {
        const float ofx = static_cast<float>(ctx.xyoffset.ofx >> 4), ofy = static_cast<float>(ctx.xyoffset.ofy >> 4);
        const float texW = static_cast<float>(p.texW), texH = static_cast<float>(p.texH);
        for (int i = 0; i < 3; ++i)
        {
            const GSVertex &v = q.v[i];
            float s = 0.0f, t = 0.0f, tq = 1.0f;
            if (p.tme && p.fst)
            {
                s = static_cast<float>(v.u) * 0.0625f;
                t = static_cast<float>(v.v) * 0.0625f;
            }
            else if (p.tme)
            {
                s = v.s * texW;
                t = v.t * texH;
                tq = v.q;
            }
            add(v.x - ofx, v.y - ofy, depthOf(v.z), q.gouraud ? v : q.v[2], static_cast<float>(v.fog), s, t, tq);
        }
        // Q that is nothing would be divided by.
        if (std::fabs(out[0].q) < 1.0e-12f && std::fabs(out[1].q) < 1.0e-12f && std::fabs(out[2].q) < 1.0e-12f)
            out[0].q = out[1].q = out[2].q = 1.0f;
    }
    // Can the alpha the pixels blend with pass 0x80? From the largest the
    // vertices have and the largest in the texture.
    if (p.abe && p.csel == 0u)
    {
        int vertexAlpha = 0;
        for (size_t i = 0; i < count; ++i)
            vertexAlpha = std::max<int>(vertexAlpha, out[i].a);
        int most = vertexAlpha;
        if (p.tme && p.tcc)
        {
            const int texel = gpu.textureMaxAlpha;
            most = p.tfx == 0u ? (texel * vertexAlpha) >> 7 : (p.tfx == 2u ? texel + vertexAlpha : texel);
        }
        state.alphaOver = most > 128 ? 1u : 0u;
    }
    gpu.renderer->Draw(state, out, count);

    static const bool wantStats = []
    {
        const char *value = std::getenv("PS2X_GS_STATS");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    if (wantStats && ++gpu.statDraws % 2000000u == 0u)
        std::fprintf(stderr, "[gs-gpu] %llu primitives: textures %llu from a surface, %llu from a decoded copy, %llu from VRAM (%llu decoded, %llu sent); %llu fills of %llu pages; back to VRAM %llu times, %llu pages (for a texture %llu, a CLUT %llu, software drawing %llu, another surface %llu, an image %llu, the display %llu, everything %llu); %llu pages changed hands, %llu given up to images; %llu pieces of channel shuffles; %zu surfaces, %zu VRAM textures\n",
                     static_cast<unsigned long long>(gpu.statDraws), static_cast<unsigned long long>(gpu.statSurfaceTextures),
                     static_cast<unsigned long long>(gpu.statCopyTextures), static_cast<unsigned long long>(gpu.statVramTextures),
                     static_cast<unsigned long long>(gpu.statDecodes), static_cast<unsigned long long>(gpu.statUploads),
                     static_cast<unsigned long long>(gpu.statFills), static_cast<unsigned long long>(gpu.statFilledPages),
                     static_cast<unsigned long long>(gpu.statDownloads), static_cast<unsigned long long>(gpu.statDownloadedPages),
                     static_cast<unsigned long long>(gpu.statWhy[GpuState::WhyTexture]), static_cast<unsigned long long>(gpu.statWhy[GpuState::WhyClut]),
                     static_cast<unsigned long long>(gpu.statWhy[GpuState::WhySoftware]),
                     static_cast<unsigned long long>(gpu.statWhy[GpuState::WhyOtherSurface]),
                     static_cast<unsigned long long>(gpu.statWhy[GpuState::WhyImage]), static_cast<unsigned long long>(gpu.statWhy[GpuState::WhyDisplay]),
                     static_cast<unsigned long long>(gpu.statWhy[GpuState::WhyEverything]),
                     static_cast<unsigned long long>(gpu.statAdopted), static_cast<unsigned long long>(gpu.statDropped),
                     static_cast<unsigned long long>(gpu.statShuffled), gpu.surfaces.size(), gpu.textures.size());
}

// A channel shuffle. A buffer of 32-bit pixels read as 16-bit ones is twice
// as tall, and in every 16 columns of that view the left 8 are the low
// halves of 8 pixels (their red and green) and the right 8 the high halves
// (blue and alpha) of the same 8. A game that wants one channel where
// another is draws strips of the one view over the other columns, texel for
// texel: a 16-bit frame from a 16-bit texture, 8 columns at a time.
// Drakengard 2 dims the scene behind its text that way, every frame it
// does: the left columns of its frame onto the right columns of a second
// buffer, whose alpha (the frame's green) it then blends the frame by.
//
// The card has both buffers as the 32-bit surfaces they are, and a 16-bit
// view of either is no surface of its: drawn the general way, each strip
// brings its pages back to VRAM for the software path. Here each strip is
// drawn on the card instead, as the pixels it is the halves of: for every
// 8 by 8 block of the 16-bit view it covers, the block of 32-bit pixels
// that is, read from the texture's surface with one half of the texel in
// the place of both and written to that half alone (m_shuffle). Blocks side
// by side in both buffers go as one sprite: a column of the 16-bit view is
// a row of blocks of the other.
//
// False for a sprite that is not that (and it is drawn as before).
bool GSCpuBackend::GpuShuffle(const GSPrimitiveBatch &batch, const PreparedPrimitive &q)
{
    const FastPipeline &p = q.pipeline;
    if (!m_cardOnly || !GpuReady() || !q.sprite || q.placed || !p.tme || !p.fst || p.linear || !p.tcc)
        return false;
    if (p.texKind != TextureKind::Direct16 || p.abe || p.fge || p.date || p.fba || p.fbmsk != 0u || p.ztst != 1u)
        return false;
    // Written whatever is there, and the frame alone.
    const bool frameOnly = p.alphaTest && p.atst == 0u && p.afail == 1u;
    if ((p.alphaTest && p.atst != 1u && !frameOnly) || (!p.zmask && !frameOnly))
        return false;
    // The texel as it is: its colours unchanged, and its one bit of alpha.
    const GSVertex &colour = q.v[1];
    if (p.tfx != 1u && !(p.tfx == 0u && colour.r == 0x80u && colour.g == 0x80u && colour.b == 0x80u && colour.a == 0x80u))
        return false;
    if ((p.texa.ta1 & 0x80u) == 0u || (p.texa.ta0 & 0x80u) != 0u)
        return false;
    if (p.frame.spillsIntoNextPage || p.texture.spillsIntoNextPage || p.frame.unitBits != 16u || p.texture.unitBits != 16u)
        return false;

    // Texel for texel, in whole blocks of 8 by 8.
    const int width = static_cast<int>(q.spanX), height = static_cast<int>(q.spanY);
    const int u0 = static_cast<int>(std::lround(q.tu0)), v0 = static_cast<int>(std::lround(q.tv0));
    if (std::fabs(q.tu0 - static_cast<float>(u0)) > 0.01f || std::fabs(q.tv0 - static_cast<float>(v0)) > 0.01f ||
        std::fabs(q.tu1 - q.tu0 - q.spanX) > 0.01f || std::fabs(q.tv1 - q.tv0 - q.spanY) > 0.01f)
        return false;
    const int x0 = q.minX, x1 = q.maxX + 1, y0 = q.minY, y1 = q.maxY + 1;
    const int du = u0 - q.originX, dv = v0 - q.originY;
    if (x0 < q.originX || x1 > q.originX + width || y0 < q.originY || y1 > q.originY + height)
        return false;
    if (((x0 | x1 | y0 | y1 | du | dv) & 7) != 0 || x0 + du < 0 || y0 + dv < 0 || x1 + du > p.texW || y1 + dv > p.texH)
        return false;

    const GSContext &ctx = batch.state.context;
    const GSMem::SurfaceLayout frame32 = GSMem::LayoutOf(GS_PSM_CT32, ctx.frame.fbp * 32u, ctx.frame.fbw);
    const GSMem::SurfaceLayout texture32 = GSMem::LayoutOf(GS_PSM_CT32, p.tbp0, p.tbw);
    if (!frame32.table || !texture32.table || frame32.spillsIntoNextPage || texture32.spillsIntoNextPage)
        return false;

    // Where each 32-bit word of a page is in the page, as 32-bit pixels.
    static const std::array<uint16_t, 2048> wordAt = []
    {
        std::array<uint16_t, 2048> at{};
        const GSMem::SurfaceLayout layout = GSMem::LayoutOf(GS_PSM_CT32, 0u, 1u);
        for (uint32_t y = 0; y < 32u; ++y)
            for (uint32_t x = 0; x < 64u; ++x)
                at[layout.unit(x, y) & 2047u] = static_cast<uint16_t>((y << 6) | x);
        return at;
    }();
    // The 32-bit pixel a pixel of a 16-bit view is half of, in `as32`'s
    // buffer, and which half.
    struct Half
    {
        int x, y;
        bool high;
    };
    const auto halfOf = [](const GSMem::SurfaceLayout &as16, const GSMem::SurfaceLayout &as32, int x, int y)
    {
        const uint32_t unit = as16.unit(static_cast<uint32_t>(x), static_cast<uint32_t>(y));
        const uint32_t word = unit >> 1, page = (word >> 11) - as32.basePage, at = wordAt[word & 2047u];
        return Half{static_cast<int>((page % as32.pagesPerRow) * 64u + (at & 63u)),
                    static_cast<int>((page / as32.pagesPerRow) * 32u + (at >> 6)), (unit & 1u) != 0u};
    };

    struct Piece
    {
        int x, y, sourceX, sourceY, width;
        bool high, sourceHigh;
    };
    std::vector<Piece> pieces;
    pieces.reserve(static_cast<size_t>((y1 - y0) / 64 + 2) * static_cast<size_t>((x1 - x0) / 8));
    for (int x = x0; x < x1; x += 8)
    {
        for (int y = y0; y < y1; y += 8)
        {
            const Half to = halfOf(p.frame, frame32, x, y), from = halfOf(p.texture, texture32, x + du, y + dv);
            // A block is a block in both views, or this is not the layout
            // it is taken for.
            const Half toEnd = halfOf(p.frame, frame32, x + 7, y + 7), fromEnd = halfOf(p.texture, texture32, x + du + 7, y + dv + 7);
            if (toEnd.x != to.x + 7 || toEnd.y != to.y + 7 || toEnd.high != to.high || fromEnd.x != from.x + 7 ||
                fromEnd.y != from.y + 7 || fromEnd.high != from.high || ((to.x | to.y | from.x | from.y) & 7) != 0)
                return false;
            if (!pieces.empty())
            {
                Piece &last = pieces.back();
                if (last.y == to.y && last.x + last.width == to.x && last.sourceY == from.y &&
                    last.sourceX + last.width == from.x && last.high == to.high && last.sourceHigh == from.high)
                {
                    last.width += 8;
                    continue;
                }
            }
            pieces.push_back({to.x, to.y, from.x, from.y, 8, to.high, from.high});
        }
    }

    // Each piece as a sprite between the two buffers as 32-bit ones.
    GSPrimitiveBatch piece = batch;
    GSDrawState &state = piece.state;
    state.context.frame.psm = GS_PSM_CT32;
    state.context.frame.fbmsk = 0u;
    state.context.zbuf.zmask = true;
    state.context.tex0.psm = GS_PSM_CT32;
    state.context.tex0.tw = 10u;
    state.context.tex0.th = 10u;
    state.context.tex0.tcc = 1u;
    state.context.tex0.tfx = 1u;
    state.context.tex1 = 0u;
    state.context.clamp = 0u;
    state.context.alpha = 0u;
    state.context.fba = 0u;
    state.context.test = 0x30000u; // depth: always
    state.context.xyoffset = {};
    state.context.scissor.x0 = 0u;
    state.context.scissor.y0 = 0u;
    state.context.scissor.x1 = static_cast<uint16_t>(std::max<uint32_t>(ctx.frame.fbw * 64u, 1u) - 1u);
    state.context.scissor.y1 = 2047u;
    state.prim.abe = false;
    state.prim.fge = false;
    state.prim.tme = true;
    state.prim.fst = true;
    state.pabe = false;
    state.textureWidth = 1024u;
    state.textureHeight = 1024u;
    state.linearFilter = false;
    piece.vertexCount = 2u;
    bool drawn = true;
    if (m_gpu->statShuffled == 0u && std::getenv("PS2X_GS_STATS"))
        std::fprintf(stderr, "[gs-gpu] a channel shuffle: page %u from page %u, both %u wide\n", ctx.frame.fbp, p.tbp0 / 32u,
                     ctx.frame.fbw * 64u);
    m_gpu->statShuffled += pieces.size();
    for (const Piece &part : pieces)
    {
        GSVertex &a = piece.vertices[0], &b = piece.vertices[1];
        a.x = static_cast<float>(part.x);
        a.y = static_cast<float>(part.y);
        a.u = static_cast<uint16_t>(part.sourceX << 4);
        a.v = static_cast<uint16_t>(part.sourceY << 4);
        b.x = static_cast<float>(part.x + part.width);
        b.y = static_cast<float>(part.y + 8);
        b.u = static_cast<uint16_t>((part.sourceX + part.width) << 4);
        b.v = static_cast<uint16_t>((part.sourceY + 8) << 4);
        a.z = b.z = 0.0;
        m_shuffle = static_cast<uint8_t>(1u + (part.sourceHigh ? 1u : 0u) + (part.high ? 2u : 0u));
        drawn = DrawFast(piece, FastKind::Sprite) && drawn;
    }
    m_shuffle = 0u;
    return drawn;
}

// PS2X_GPU_COMPARE=<n> (with a resolution of 1): every n-th frame shown,
// each frame buffer the card has is read back and its good pages held
// against VRAM, which the raster pool drew from the same primitives. Says
// how far apart they are, and writes both as PPM files
// (PS2X_GPU_COMPARE_DIR, or the working directory) whenever a buffer is
// further off than it has been before.
void GSCpuBackend::GpuCompare()
{
    GpuState &gpu = *m_gpu;
    FlushPrimitives("compare");
    static uint64_t round = 0;
    ++round;
    for (auto &entry : gpu.surfaces)
    {
        GpuState::Surface &surface = entry.second;
        if (surface.depthKind >= 0 || surface.scale.x != 1 || surface.scale.y != 1 || surface.layout.unitBits != 32u)
            continue;
        const GSMem::SurfaceLayout &layout = surface.layout;
        gpu.cells.clear();
        forEachPageOf(layout, 0, 0, surface.width - 1, surface.height - 1, [&](uint32_t page, uint32_t row, uint32_t column)
                      {
                          if (PageGood(page, entry.first))
                              gpu.cells.push_back({page, row, column});
                      });
        if (gpu.cells.empty())
            continue;
        std::vector<uint8_t> card;
        uint32_t gotWidth = 0, gotHeight = 0;
        GSGpu::Reading reading = gpu.renderer->Read(surface.id, 0, 0, surface.width, surface.height);
        if (!reading.Wait(card, gotWidth, gotHeight))
            continue;
        const int pageWidth = 1 << layout.pageWidthShift, pageHeight = 1 << layout.pageHeightShift;
        uint64_t sum = 0, sumAlpha = 0, pixels = 0, wrong = 0, wrongAlpha = 0;
        int largest = 0;
        for (const GpuState::Cell &cell : gpu.cells)
        {
            for (int py = 0; py < pageHeight; ++py)
            {
                const int y = static_cast<int>(cell.row) * pageHeight + py;
                const FastPipeline::Row row = FastPipeline::rowOf(layout, y);
                for (int px = 0; px < pageWidth; ++px)
                {
                    const int x = static_cast<int>(cell.column) * pageWidth + px;
                    const uint32_t there = read32(m_vram, FastPipeline::unitIn(layout, row, x));
                    const uint8_t *got = card.data() + (static_cast<size_t>(y) * gotWidth + static_cast<size_t>(x)) * 4u;
                    int difference = 0;
                    for (int c = 0; c < 3; ++c)
                        difference = std::max(difference, std::abs(static_cast<int>(got[c]) - static_cast<int>((there >> (c * 8)) & 0xFFu)));
                    const int alpha = std::abs(static_cast<int>(got[3]) - static_cast<int>(there >> 24));
                    sum += static_cast<uint64_t>(difference);
                    sumAlpha += static_cast<uint64_t>(alpha);
                    largest = std::max(largest, difference);
                    wrong += difference > 16 ? 1u : 0u;
                    wrongAlpha += alpha > 16 ? 1u : 0u;
                    ++pixels;
                }
            }
        }
        const double share = 100.0 * static_cast<double>(wrong) / static_cast<double>(pixels);
        const double shareAlpha = 100.0 * static_cast<double>(wrongAlpha) / static_cast<double>(pixels);
        std::fprintf(stderr, "[gpu-compare] %llu fbp=%u fbw=%u pages=%zu: colour mean %.2f max %d wrong %.3f%%; alpha mean %.2f wrong %.3f%%\n",
                     static_cast<unsigned long long>(round), layout.basePage, layout.pagesPerRow, gpu.cells.size(),
                     static_cast<double>(sum) / static_cast<double>(pixels), largest, share,
                     static_cast<double>(sumAlpha) / static_cast<double>(pixels), shareAlpha);
        if (share > surface.worst + 0.05)
        {
            surface.worst = share;
            const char *directory = std::getenv("PS2X_GPU_COMPARE_DIR");
            for (int which = 0; which < 2; ++which)
            {
                char name[512];
                std::snprintf(name, sizeof(name), "%s/gpucmp_%u_%u_%s.ppm", directory ? directory : ".", layout.basePage, layout.pagesPerRow,
                              which == 0 ? "card" : "vram");
                FILE *out = std::fopen(name, "wb");
                if (!out)
                    continue;
                std::fprintf(out, "P6\n%d %d\n255\n", surface.width, surface.height);
                for (int y = 0; y < surface.height; ++y)
                {
                    const FastPipeline::Row row = FastPipeline::rowOf(layout, y);
                    for (int x = 0; x < surface.width; ++x)
                    {
                        uint8_t rgb[3];
                        if (which == 0)
                        {
                            std::memcpy(rgb, card.data() + (static_cast<size_t>(y) * gotWidth + static_cast<size_t>(x)) * 4u, 3);
                        }
                        else
                        {
                            const uint32_t there = read32(m_vram, FastPipeline::unitIn(layout, row, x));
                            rgb[0] = static_cast<uint8_t>(there);
                            rgb[1] = static_cast<uint8_t>(there >> 8);
                            rgb[2] = static_cast<uint8_t>(there >> 16);
                        }
                        std::fwrite(rgb, 1, 3, out);
                    }
                }
                std::fclose(out);
            }
        }
    }
}

// Before the display is made from VRAM after all: the pages of these
// frames that only the card has. True if there were any.
bool GSCpuBackend::GpuDownloadFrames(const GSFrameReg *frames, const uint32_t *heights, const bool *baseIsPages, size_t count)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!m_gpu || m_cardAheadCount == 0u)
        return false;
    const uint64_t before = m_gpu->statDownloads;
    for (size_t i = 0; i < count; ++i)
    {
        const uint32_t fbw = std::max<uint32_t>(frames[i].fbw, 1u);
        const GSMem::SurfaceLayout layout = GSMem::LayoutOf(frames[i].psm, baseIsPages[i] ? framePageBaseToBlock(frames[i].fbp) : frames[i].fbp, fbw);
        if (layout.table && heights[i] != 0u)
            GpuDownload(layout, 0, 0, static_cast<int>(fbw * 64u) - 1, static_cast<int>(heights[i]) - 1, GpuState::WhyDisplay);
    }
    return m_gpu->statDownloads != before;
}

// CopyScaledFrame from the card: the pixels of the surface, when the frame has one.
bool GSCpuBackend::GpuCopyFrame(const GSFrameReg &frame, SurfaceScale scale, uint32_t originX, uint32_t originY, uint32_t width,
                                uint32_t height, std::vector<uint8_t> &outPixels)
{
    const uint32_t fbw = std::max<uint32_t>(frame.fbw, 1u);
    const GSMem::SurfaceLayout layout = GSMem::LayoutOf(frame.psm, framePageBaseToBlock(frame.fbp), fbw);
    if (!layout.table || layout.spillsIntoNextPage || layout.unitBits != 32u || width == 0u || height == 0u)
        return false;
    const int x0 = static_cast<int>(originX), y0 = static_cast<int>(originY);
    const int x1 = x0 + static_cast<int>(width) - 1, y1 = y0 + static_cast<int>(height) - 1;
    // The window draws the picture from a texture the card fills
    // (GSGpu::LatestShown) unless its pixels are asked for here
    // (PS2X_GPU_READBACK=1): what goes back then stands for it, with its
    // size.
    static const bool direct = []
    {
        const char *value = std::getenv("PS2X_GPU_READBACK");
        return value == nullptr || value[0] == '\0' || value[0] == '0';
    }();
    GSGpu::Reading reading;
    for (int attempt = 0; !reading; ++attempt)
    {
        uint64_t waitFor = 0;
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            if (!m_gpu)
                return false;
            const uint64_t key = GpuKey(layout, scale, -1);
            const auto found = m_gpu->surfaces.find(key);
            if (found == m_gpu->surfaces.end())
                return false;
            // Pages something else wrote since come from VRAM, once the
            // pool has drawn what it still has for them.
            if (PagesStale(layout, key, x0, y0, x1, y1))
            {
                waitFor = LastBatchTouching(layout, x0, y0, x1, y1, false, "present");
                if (waitFor != 0u && attempt >= 3)
                {
                    WaitForBatch(waitFor);
                    waitFor = 0u;
                }
                if (waitFor == 0u)
                {
                    GpuDownload(layout, x0, y0, x1, y1, GpuState::WhyDisplay, key);
                    GpuFill(layout, scale, -1, x0, y0, x1, y1, false);
                }
            }
            if (waitFor == 0u)
            {
                static const int compareEvery = []
                {
                    const char *value = std::getenv("PS2X_GPU_COMPARE");
                    return value ? std::max(std::atoi(value), 1) : 0;
                }();
                static int shown = 0;
                if (compareEvery != 0 && ++shown % compareEvery == 0)
                    GpuCompare();
                reading = direct ? m_gpu->renderer->Show(found->second.id, x0, y0, static_cast<int>(width), static_cast<int>(height))
                                 : m_gpu->renderer->Read(found->second.id, x0, y0, static_cast<int>(width), static_cast<int>(height));
            }
        }
        if (waitFor != 0u)
            WaitForBatch(waitFor);
    }
    uint32_t gotWidth = 0u, gotHeight = 0u;
    if (!reading.Wait(outPixels, gotWidth, gotHeight) || gotWidth != width * static_cast<uint32_t>(scale.x) ||
        gotHeight != height * static_cast<uint32_t>(scale.y))
        return false;
    if (direct)
        outPixels.assign(4u, 255u);
    for (size_t i = 3; i < outPixels.size(); i += 4u)
        outPixels[i] = 255u;
    return true;
}
