#pragma once

#include "runtime/gs/gs_backend.h"
#include "runtime/gs/gs_texture_page_cache.h"

#include <array>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace GSMem
{
    struct SurfaceLayout;
}

namespace GSGpu
{
    struct Pipeline;
}

class GSCpuBackend final : public GSRasterBackend
{
public:
    GSCpuBackend();

    void Initialize(uint8_t *vram, uint32_t vramSize) override;
    void Reset() override;

    void Submit(const GSPrimitiveBatch &batch) override;
    void LoadClut(const GSTex0Reg &tex0, const GSTexClutReg &texclut) override;
    void BeginTransfer(const GSTransferCommand &command) override;
    void UploadImage(const uint8_t *data, uint32_t sizeBytes) override;

    void Flush() override;
    void TextureFlush() override;
    void Sync(GSSyncReason reason) override;
    PresentationFrame Present(const GSPresentationRequest &request) override;

    bool ClearFramebuffer(const GSContext &context, uint32_t rgba) override;
    uint32_t ConsumeLocalToHostBytes(uint8_t *dst, uint32_t maxBytes) override;

    uint32_t ReadVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y) const override;
    void WriteVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y, uint32_t value) override;
    void SnapshotVram(std::vector<uint8_t> &out) const override;
    GSTransferSnapshot GetTransferSnapshot() const override;

private:
    void ResetUnlocked();
    void LoadClutUnlocked(const GSTex0Reg &tex0, const GSTexClutReg &texclut);
    uint32_t ReadVramUnlocked(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y) const;
    uint32_t ReadTextureVramUnlocked(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y);
    void WriteVramUnlocked(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y, uint32_t value);

    // gs_cpu_raster.cpp: the same pixels as DrawTriangle/DrawSprite from a
    // pipeline resolved once per primitive. Return false to fall back.
    struct FastPipeline;
    // An uploaded image as RGBA, keyed by what was uploaded rather than where
    // (see DecodedTextureFor). It is decoded once and never again, so nothing
    // has to wait for the primitives that use it.
    struct DecodedTexture
    {
        uint64_t lastUsed = 0;
        int width = 0;  // of a row: the texture's
        int height = 0; // rows it has: the image's
        std::vector<uint32_t> pixels; // RGBA, row major
        uint32_t gpuTexture = 0; // the same as a texture of the graphics card's, once it has drawn with it
    };
    struct PreparedPrimitive;
    // What a Rasterise call covered, for PS2X_GS_STATS.
    struct RasterTally
    {
        uint64_t pixels = 0, rows = 0, blocks = 0;
    };
    struct RasterQueue;
    struct RasterQueueDeleter
    {
        void operator()(RasterQueue *queue) const;
    };
    static bool UseReferenceRasteriser();
    enum class FastKind
    {
        Triangle,
        Sprite,
        Line,
    };
    bool DrawFast(const GSPrimitiveBatch &batch, FastKind kind);
    bool PrepareTriangle(const GSPrimitiveBatch &batch, PreparedPrimitive &primitive);
    bool PrepareSprite(const GSPrimitiveBatch &batch, PreparedPrimitive &primitive);
    bool PrepareLine(const GSPrimitiveBatch &batch, PreparedPrimitive &primitive);
    static void Rasterise(const PreparedPrimitive &primitive, int yFirst, int yLast);
    static void RasteriseRows(const PreparedPrimitive &primitive, int yFirst, int yLast, RasterTally *tally);
    static void RasteriseRowsScalar(const PreparedPrimitive &primitive, int yFirst, int yLast, RasterTally *tally);
    // Draw what is queued: all of it, or only if it involves these pixels of
    // a surface (anything that reads or writes VRAM outside the queue asks).
    void FlushPrimitives(const char *why = "sync");
    void SubmitPrimitives();
    void QueueImage(const GSMem::SurfaceLayout &layout, uint8_t dpsm, uint32_t streamBits, uint32_t dsax, uint32_t dsay,
                    uint32_t rrw, uint32_t rrh, const uint8_t *data, size_t bytes);
    static uint32_t StoreImagePixels(uint8_t *vram, const GSMem::SurfaceLayout &layout, uint8_t dpsm, uint32_t streamBits,
                                     uint32_t dsax, uint32_t dsay, uint32_t rrw, uint32_t pixel, uint32_t endPixel,
                                     const uint8_t *data, size_t sizeBytes, uint32_t streamPixel);
    uint64_t LastBatchTouching(const GSMem::SurfaceLayout &layout, int x0, int y0, int x1, int y1, bool writes, const char *why);
    void WaitForBatch(uint64_t batch);
    void SubmitPrimitivesWriting(const GSMem::SurfaceLayout &layout, int x0, int y0, int x1, int y1, const char *why);
    void FlushPrimitivesTouching(const GSMem::SurfaceLayout &layout, int x0, int y0, int x1, int y1, bool writes,
                                 const char *why = "upload");
    bool HasQueuedPrimitives() const;
    const DecodedTexture *DecodedTextureFor(FastPipeline &pipeline);
    void ForgetOldDecodedTextures();
    uint64_t ConversionKey(const FastPipeline &pipeline);
    const uint32_t *ClutSnapshotFor(FastPipeline &pipeline);
    bool PagesWrittenSince(const GSMem::SurfaceLayout &layout, int x0, int y0, int x1, int y1, uint64_t serial) const;

    // Texture cache bookkeeping: every writer says which VRAM pages it touched.
    static constexpr uint32_t kVramPages = 512u;
    void MarkWritten(const GSMem::SurfaceLayout &layout, int x0, int y0, int x1, int y1);
    void MarkUploaded(const GSMem::SurfaceLayout &layout, int x0, int y0, int x1, int y1);
    bool DrawnSince(const GSMem::SurfaceLayout &layout, int x0, int y0, int x1, int y1, uint64_t serial) const;
    bool UploadedSince(const GSMem::SurfaceLayout &layout, int x0, int y0, int x1, int y1, uint64_t serial) const;
    void MarkAllWritten();

    void DrawPrimitive(const GSPrimitiveBatch &batch);
    void DrawSprite(const GSPrimitiveBatch &batch);
    void DrawTriangle(const GSPrimitiveBatch &batch);
    void DrawLine(const GSPrimitiveBatch &batch);
    void WritePixel(const GSDrawState &state, int x, int y, int z, uint8_t r, uint8_t g, uint8_t b, uint8_t a, uint8_t fog);
    uint32_t SampleTexture(const GSDrawState &state, float s, float t, float q, uint16_t u, uint16_t v);
    uint32_t LookupCLUT(const GSDrawState &state, uint8_t index, uint8_t cpsm, uint8_t csm, uint8_t csa, uint8_t sourcePsm);

    void PerformLocalToLocalTransfer();
    void PerformLocalToHostTransfer();
    PresentationFrame PresentFromLocalMemory(const GSPresentationRequest &request);
    void SnapshotFrames(const GSFrameReg *frames, const uint32_t *heights, const bool *baseIsPages, size_t count,
                        std::vector<uint8_t> &out);
    bool CopyFrameToHostRgba(const GSFrameReg &frame,
                             uint32_t width,
                             uint32_t height,
                             std::vector<uint8_t> &outPixels,
                             bool preserveAlpha,
                             bool useLocalMemoryLayout,
                             bool frameBaseIsPages,
                             uint32_t sourceOriginX,
                             uint32_t sourceOriginY) const;

    using WriteVramFunc = void (*)(uint8_t *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);
    using ReadVramFunc = uint32_t (*)(uint8_t *, uint32_t, uint32_t, uint32_t, uint32_t);

    static constexpr size_t kPsmHandlerCount = 1u << 6u;
    mutable std::mutex m_mutex;
    uint8_t *m_vram = nullptr;
    uint32_t m_vramSize = 0;
    std::array<ReadVramFunc, kPsmHandlerCount> m_readVramFuncs{};
    std::array<WriteVramFunc, kPsmHandlerCount> m_writeVramFuncs{};
    std::array<uint16_t, 512> m_clut{};
    std::array<uint32_t, 2> m_clutCbp{};
    // CLUT entries converted to RGBA for the fast paths, filled on first use.
    // A stamp other than m_fastClutKey marks an entry stale.
    // What each sixteen-entry stretch of the CLUT buffer was last loaded with
    // (the load's parameters) and when (m_writeSerial).
    struct ClutHeld
    {
        uint64_t key = 0;
        uint64_t serial = 0;
    };
    std::array<ClutHeld, 32> m_clutHeld{};
    uint64_t m_clutLoads = 0;
    uint64_t m_fastClutLoad = ~0ull;
    uint64_t m_fastClutConversion = ~0ull;
    uint32_t m_fastClutKey = 1;
    std::array<uint32_t, 256> m_fastClut{};
    std::array<uint32_t, 256> m_fastClutStamp{};
    uint64_t m_writeSerial = 1;
    std::array<uint64_t, kVramPages> m_pageSerial{};     // last write of any kind
    std::array<uint64_t, kVramPages> m_pageDrawSerial{}; // ...other than an image transfer
    // Image transfers, by 256-byte block.
    std::array<uint64_t, 16384> m_blockUploadSerial{};
    uint64_t m_lastUploadSerial = 0;
    // The last image each host-to-local transfer wrote, to recognise one
    // that would write what VRAM already holds.
    struct UploadRecord
    {
        uint64_t hash = 0;
        uint64_t bytes = 0;
        uint64_t rect = 0;   // position and size
        uint64_t target = 0; // base, width and format
        uint64_t serial = 0; // m_writeSerial once it was written
        uint64_t checkedUpload = 0; // m_lastUploadSerial when its blocks were last found untouched
        // The image itself when it is small enough to be a CLUT, so that
        // loading the CLUT need not wait for it to reach VRAM.
        std::vector<uint8_t> pixels;
    };
    std::unordered_map<uint64_t, UploadRecord> m_uploads;
    // The latest whole image written at each base pointer, from (0, 0).
    std::unordered_map<uint64_t, UploadRecord> m_uploadAt;
    std::unordered_map<uint64_t, std::unique_ptr<std::array<uint32_t, 256>>> m_clutSnapshots;
    uint64_t m_clutHashLoad = ~0ull; // m_clutLoads the hash below is of
    uint64_t m_clutHash = 0;
    uint64_t m_decodedBytes = 0;
    std::unordered_map<uint64_t, DecodedTexture> m_decodedTextures;
    uint64_t m_decodedUseClock = 0;
    std::unique_ptr<RasterQueue, RasterQueueDeleter> m_rasterQueue;
    GSMem::TexturePageCache m_texturePageCache;

    // Set while presenting from a snapshot that lacks the context frames.
    bool m_presentWantsContextFrames = false;

    // A larger copy of the frame and depth buffers, drawn alongside VRAM
    // and shown in its place. Two things use it:
    //
    // - A higher internal resolution (display.resolution in the settings
    //   file, PS2X_GS_SCALE): everything is drawn m_scale times as large.
    // - A display buffer the game fills a field at a time (m_fieldFbp). A
    //   game made for an interlaced television may draw each frame whole
    //   and then squeeze it into a buffer half as high, a different half
    //   of its lines each field, which is what the display shows. Shown
    //   field after field on a screen that is not interlaced that is half
    //   the detail, and it shakes. Whatever is drawn into such a buffer is
    //   drawn twice as tall as well, so the copy holds the whole frame the
    //   field was made of, and what the game then draws over the field
    //   (its subtitles).
    //
    // Every primitive the fast paths draw into a surface with a copy is
    // drawn a second time, scaled, into m_shadow (see DrawFast). VRAM
    // itself is drawn as ever, so whatever reads it (a texture taken from
    // the frame buffer, a transfer back to the game) sees what it always
    // did.
    //
    // The shadow is laid out like VRAM with room for the largest copy of
    // every page: a surface's base page is multiplied by that and its
    // width in pages by the scale across, and the same page table then
    // places the pixels (see ShadowLayout). A page of it is good while the
    // last thing written to the VRAM page was a primitive drawn into both;
    // anything else that writes the page (an image, a clear, a move, the
    // reference rasteriser) leaves m_pageSerial ahead of m_shadowSerial,
    // and the page is filled from VRAM again, each pixel repeated, before
    // it is next drawn into or shown.
    struct SurfaceScale
    {
        int x = 1, y = 1;
        bool larger() const { return x > 1 || y > 1; }
    };
    int m_scale = 0; // 0 until the first primitive asks
    uint32_t m_fieldFbp = ~0u, m_fieldFbw = 0; // from Present
    std::unique_ptr<uint8_t[]> m_shadow;
    uint32_t m_shadowMask = 0;
    std::array<uint64_t, kVramPages> m_shadowSerial{};
    std::array<uint64_t, kVramPages> m_shadowKey{}; // which surface the page was filled as, and how large
    static int ScaleSetting();
    static bool WholeFramesSetting();
    SurfaceScale ScaleOf(uint32_t fbp, uint32_t fbw) const;
    GSMem::SurfaceLayout ShadowLayout(const GSMem::SurfaceLayout &layout, SurfaceScale scale) const;
    bool ShadowStale(const GSMem::SurfaceLayout &layout, SurfaceScale scale, int x0, int y0, int x1, int y1) const;
    void FillShadow(const GSMem::SurfaceLayout &layout, SurfaceScale scale, int x0, int y0, int x1, int y1);
    void MarkShadowDrawn(const GSMem::SurfaceLayout &layout, SurfaceScale scale, int x0, int y0, int x1, int y1);
    void ScaleUp(PreparedPrimitive &primitive, SurfaceScale scale) const;
    bool PrepareBlit(PreparedPrimitive &scaled, const GSContext &context);
    static void BlitRows(const PreparedPrimitive &primitive, int yFirst, int yLast);
    bool CopyScaledFrame(const GSFrameReg &frame, SurfaceScale scale, uint32_t originX, uint32_t originY, uint32_t width, uint32_t height,
                         std::vector<uint8_t> &outPixels);
    bool PageGood(uint32_t page, uint64_t key);
    bool PagesStale(const GSMem::SurfaceLayout &layout, uint64_t key, int x0, int y0, int x1, int y1);
    void MarkPages(const GSMem::SurfaceLayout &layout, uint64_t key, int x0, int y0, int x1, int y1);

    // The larger copy kept by the graphics card (display.renderer in the
    // settings file, PS2X_GS_RENDERER) rather than in m_shadow, and drawn
    // by it: gs_cpu_raster_gpu.inl. The pages are accounted for in the same
    // way, with m_shadowSerial and m_shadowKey; a surface of the card's
    // stands for each frame or depth buffer where the shadow has room for
    // every page. `depthKind` is -1 for a frame buffer.
    struct GpuState;
    struct GpuStateDeleter
    {
        void operator()(GpuState *state) const;
    };
    std::unique_ptr<GpuState, GpuStateDeleter> m_gpu;
    bool m_gpuAsked = false;
    static bool GpuSetting();
    bool GpuReady();
    uint64_t GpuKey(const GSMem::SurfaceLayout &layout, SurfaceScale scale, int depthKind) const;
    uint32_t GpuSurface(const GSMem::SurfaceLayout &layout, SurfaceScale scale, int depthKind, int rows);
    void GpuFill(const GSMem::SurfaceLayout &layout, SurfaceScale scale, int depthKind, int x0, int y0, int x1, int y1, bool overwritten);
    void GpuDraw(const GSPrimitiveBatch &batch, PreparedPrimitive &primitive, SurfaceScale scale);
    void GpuTexture(PreparedPrimitive &primitive, const GSContext &context, GSGpu::Pipeline &state);
    void GpuTextureFromVram(FastPipeline &pipeline, GSGpu::Pipeline &state);
    void GpuForgetTexture(uint32_t texture);
    void GpuForgetTextures();
    void GpuFlush();
    void GpuCompare();
    bool GpuCopyFrame(const GSFrameReg &frame, SurfaceScale scale, uint32_t originX, uint32_t originY, uint32_t width, uint32_t height,
                      std::vector<uint8_t> &outPixels);

    GSTransferCommand m_transfer{};
    GSTransferSnapshot m_transferState{};
    std::vector<uint8_t> m_localToHostBuffer;
    size_t m_localToHostReadPos = 0;
};

// A GS on its own draws each primitive before the call that sent it returns:
// whoever holds its memory can look at it right away. The runtime, which
// reads that memory only through Sync and Present, names its GS's memory
// here to let the raster pool draw in the background (nullptr: no GS does).
void ps2xSetGsBackgroundDrawing(const uint8_t *vram);
// Whether the backend that draws into this VRAM is that one.
bool ps2xGsDrawsInBackground(const uint8_t *vram);
