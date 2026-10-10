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
    struct DecodedTexture
    {
        uint64_t contentHash = 0;   // of the VRAM pages it was decoded from
        uint64_t checkedSerial = 0; // m_writeSerial when it was last known current
        uint64_t lastUsed = 0;
        uint64_t lastDecoded = 0;   // lookup clock at the last content change
        uint64_t directUntil = 0;   // sample VRAM directly until this lookup
        int width = 0;
        int height = 0;
        std::vector<uint32_t> pages;
        std::vector<uint32_t> pixels; // RGBA, row major
        // Keyed by what was uploaded rather than where: never decoded again,
        // so a transfer to its pages need not wait for primitives using it.
        bool immutable = false;
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
    bool DrawFast(const GSPrimitiveBatch &batch, bool sprite);
    bool PrepareTriangle(const GSPrimitiveBatch &batch, PreparedPrimitive &primitive);
    bool PrepareSprite(const GSPrimitiveBatch &batch, PreparedPrimitive &primitive);
    static void Rasterise(const PreparedPrimitive &primitive, int yFirst, int yLast);
    static void RasteriseRows(const PreparedPrimitive &primitive, int yFirst, int yLast, RasterTally *tally);
    static void RasteriseRowsScalar(const PreparedPrimitive &primitive, int yFirst, int yLast, RasterTally *tally);
    // Draw what is queued: all of it, or only if it involves these pixels of
    // a surface (anything that reads or writes VRAM outside the queue asks).
    void FlushPrimitives();
    void FlushPrimitivesTouching(const GSMem::SurfaceLayout &layout, int x0, int y0, int x1, int y1, bool writes,
                                 const char *why = "upload");
    bool HasQueuedPrimitives() const;
    const DecodedTexture *DecodedTextureFor(FastPipeline &pipeline);
    uint64_t HashPages(const std::vector<uint32_t> &pages) const;
    void ForgetOldDecodedTextures();
    uint64_t BorderHash(FastPipeline &p, uint64_t conversion, int imageW, int imageH, int width, int height);
    bool PagesWrittenSince(const GSMem::SurfaceLayout &layout, int x0, int y0, int x1, int y1, uint64_t serial) const;

    // Texture cache bookkeeping: every writer says which VRAM pages it touched.
    static constexpr uint32_t kVramPages = 512u;
    void MarkWritten(const GSMem::SurfaceLayout &layout, int x0, int y0, int x1, int y1);
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
    uint64_t m_clutLoads = 0;
    uint64_t m_fastClutLoad = ~0ull;
    uint64_t m_fastClutConversion = ~0ull;
    uint32_t m_fastClutKey = 1;
    std::array<uint32_t, 256> m_fastClut{};
    std::array<uint32_t, 256> m_fastClutStamp{};
    uint64_t m_writeSerial = 1;
    std::array<uint64_t, kVramPages> m_pageSerial{};
    // The last image each host-to-local transfer wrote, to recognise one
    // that would write what VRAM already holds.
    struct UploadRecord
    {
        uint64_t hash = 0;
        uint64_t bytes = 0;
        uint64_t rect = 0;   // position and size
        uint64_t target = 0; // base, width and format
        uint64_t serial = 0; // m_writeSerial once it was written
    };
    std::unordered_map<uint64_t, UploadRecord> m_uploads;
    // The latest whole image written at each base pointer, from (0, 0).
    std::unordered_map<uint64_t, UploadRecord> m_uploadAt;
    struct BorderRecord
    {
        uint64_t hash = 0;
        uint64_t serial = 0; // m_writeSerial when it was read; 0 = never
    };
    std::unordered_map<uint64_t, BorderRecord> m_borderHashes;
    uint64_t m_clutHashLoad = ~0ull; // m_clutLoads the hash below is of
    uint64_t m_clutHash = 0;
    uint64_t m_decodedBytes = 0;
    std::unordered_map<uint64_t, DecodedTexture> m_decodedTextures;
    uint64_t m_decodedUseClock = 0;
    std::unique_ptr<RasterQueue, RasterQueueDeleter> m_rasterQueue;
    GSMem::TexturePageCache m_texturePageCache;

    GSTransferCommand m_transfer{};
    GSTransferSnapshot m_transferState{};
    std::vector<uint8_t> m_localToHostBuffer;
    size_t m_localToHostReadPos = 0;
};
