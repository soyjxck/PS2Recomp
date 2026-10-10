#pragma once

// The GS's frame and depth buffers drawn by the graphics card, at any size.
//
// A Renderer keeps surfaces (a frame buffer, a depth buffer or a texture,
// each a GL texture) and draws triangles into them the way the GS draws
// its primitives: the texture functions, fog, the alpha test and alpha
// blending are a fragment shader and blend state. It knows nothing of GS
// memory: whoever uses it (GSCpuBackend) says which surface is which and
// hands it the pixels of whatever was not drawn here.
//
// Everything is queued and carried out on the renderer's own thread, which
// has its own GL context, so no call here waits for the graphics card
// except Reading::Wait. Calls on one Renderer must not overlap.
//
// Coordinates are the game's pixels throughout: a surface made `scale`
// times larger is drawn into and read from in the same coordinates as one
// that is not.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace GSGpu
{
    // To be called once on the thread that owns the window, after the
    // window has its GL context: makes the context the renderer's thread
    // will use. False (and no Renderer can be made) without a window, or
    // on a platform this is not written for.
    bool PrepareContext();
    bool ContextReady();

    // A picture the renderer has put where the window can draw it without
    // its pixels leaving the card: a texture, shared with the window's
    // context, with the picture's first row first (as an uploaded image
    // has it) and opaque.
    struct Shown
    {
        uint32_t texture = 0;
        uint32_t width = 0, height = 0;
        uint64_t serial = 0; // goes up with every picture shown
    };
    // The latest (for the window's thread). False if nothing has been shown.
    bool LatestShown(Shown &shown);

    struct Vertex
    {
        float x, y; // pixels of the surface
        float z;    // depth, 0 to 1
        float s, t; // texel coordinates times q
        float q;
        uint8_t r, g, b, a;
        float fog; // 0 to 255
    };

    enum class SurfaceKind : uint8_t
    {
        Colour,
        Depth,
        Texture,
    };

    enum : uint8_t
    {
        TextureNone = 0,
        TextureTexels = 1,  // an image of texels, read exactly as the GS reads them
        TextureSurface = 2, // a Colour surface, filtered at whatever size it is (the one drawn
                            // into as well: a copy of it, as it was before the primitive)
    };

    // Everything a primitive is drawn with. Primitives are drawn together
    // while this stays the same, byte for byte: make one with Pipeline{}
    // and set its fields, never copy one field by field into another.
    struct Pipeline
    {
        uint32_t colour = 0;  // the surface drawn into
        uint32_t depth = 0;   // 0 when depth is neither tested nor written
        uint32_t texture = 0; // 0 when textureMode is TextureNone
        int16_t scissor[4]{}; // x0, y0, x1, y1, inclusive

        int16_t texSize[2]{};   // what TEX0 says the texture's size is
        int16_t texOrigin[2]{}; // TextureTexels: the texel the image starts at
                                // TextureSurface: where in the surface the texture starts
        int16_t texExtent[2]{}; // TextureTexels: how many texels it holds each way
                                // TextureSurface: texels past these are not read (0: no limit)
        int16_t region[4]{};    // CLAMP's MINU, MAXU, MINV, MAXV
        uint8_t textureMode = TextureNone;
        uint8_t linear = 0;
        uint8_t tfx = 0, tcc = 0;
        uint8_t wrapU = 0, wrapV = 0;
        uint8_t texAlphaFixed = 0; // the texture has no alpha: TEXA's TA0, or 0 for black with AEM
        uint8_t ta0 = 0, aem = 0;

        uint8_t alphaTest = 0, atst = 0, aref = 0, afail = 0;
        uint8_t ztst = 1, zmask = 1;

        uint8_t abe = 0, pabe = 0;
        uint8_t asel = 0, bsel = 0, csel = 0, dsel = 0, fix = 0;
        uint8_t fba = 0;
        uint8_t frameAlphaOne = 0; // a 24-bit frame: its alpha reads as 0x80

        uint8_t fge = 0, fogR = 0, fogG = 0, fogB = 0;
        uint8_t writeRgb = 1, writeAlpha = 1;
        uint8_t date = 0, datm = 0; // the destination alpha test, and the bit it wants
        uint8_t alphaOver = 0;      // a pixel's alpha may be more than 0x80, the GS's one
        uint8_t unused[3]{};
    };

    // Pixels asked for with Renderer::Read, when the graphics card gets to them.
    class Reading
    {
    public:
        // Waits for them: four bytes a pixel, top row first, `width` by
        // `height`. From Read that is RGBA of the surface's own pixels (its
        // scale times what was asked for); from ReadBack, the game's pixels:
        // RGBA of a Colour surface, a float from 0 to 1 of a Depth surface.
        // False if the surface was not there.
        bool Wait(std::vector<uint8_t> &pixels, uint32_t &width, uint32_t &height);
        explicit operator bool() const { return m_state != nullptr; }

        struct State;

    private:
        friend class Renderer;
        std::shared_ptr<State> m_state;
    };

    class Renderer
    {
    public:
        // nullptr unless PrepareContext succeeded.
        static std::unique_ptr<Renderer> Create();
        ~Renderer();

        // A surface `width` by `height` of the game's pixels, each of them
        // scaleX by scaleY of its own. A Texture has no scale.
        uint32_t NewSurface(SurfaceKind kind, int width, int height, int scaleX, int scaleY);
        // A different size, keeping what it holds.
        void Resize(uint32_t surface, int width, int height);
        void Release(uint32_t surface);

        // A Texture's pixels, RGBA, replacing all it had.
        void SetTexture(uint32_t surface, int width, int height, const uint32_t *rgba);
        // Pixels of a Colour surface (RGBA as the GS stores them) or a
        // Depth surface (0 to 1) from elsewhere, each made as many of the
        // surface's as its scale says.
        void FillColour(uint32_t surface, int x, int y, int width, int height, const uint32_t *rgba);
        void FillDepth(uint32_t surface, int x, int y, int width, int height, const float *depth);

        // Triangles: three vertices each.
        void Draw(const Pipeline &pipeline, const Vertex *vertices, size_t count);

        // Hands what is queued to the graphics thread.
        void Flush();
        // The pixels of part of a Colour surface, after everything queued so far.
        Reading Read(uint32_t surface, int x, int y, int width, int height);
        // Part of a Colour or Depth surface as the game's pixels again, each
        // from however many of the surface's stand for it.
        Reading ReadBack(uint32_t surface, int x, int y, int width, int height);
        // Part of a Colour surface made the picture LatestShown gives. Wait
        // on what is returned says when it is there; no pixels come back.
        Reading Show(uint32_t surface, int x, int y, int width, int height);

        struct Impl;

    private:
        Renderer();
        std::unique_ptr<Impl> m_impl;
    };
}
