// The GS's buffers drawn by the graphics card: see gs_gpu_renderer.h.
//
// OpenGL 3.3 core, through the loader and the window library raylib is
// built with. raylib's window owns a context on the main thread; a second
// one, sharing with it, belongs to the thread here. Calls on a Renderer
// only fill a list of commands (and one array of vertices for all the
// draws in it), which Flush hands to that thread.
//
// How a GS primitive becomes a draw call:
//
// - Positions are the game's pixels; the viewport is the whole surface, so
//   a larger surface is the same picture with more pixels in it. Texture
//   coordinates arrive multiplied by Q and are divided per pixel, as the
//   GS does (it has no perspective for anything else).
// - The fragment shader works in the GS's own 0..255 numbers and rounds
//   where it does (see gs_cpu_raster.cpp, FastPipeline::combine and
//   write), so that a surface no larger than the game's comes out all but
//   the same as the software one.
// - Blending: the GS computes (A - B) * alpha + D with A, B and D each the
//   pixel, what is there or nothing. That is always (pixel * X) + (there *
//   Y) with X and Y one of 0, 1, alpha, 1 - alpha, 1 + alpha or -alpha. The
//   shader multiplies by X and passes Y as a second output for the blend
//   state to multiply by (dual source blending); a negative one turns
//   the blend equation into a subtraction. Alpha from the frame buffer
//   and Y = 1 + alpha have no such form and are drawn in two passes. So
//   is Y = 1 - alpha where alpha may pass one (0x80; smoke does): the
//   pixels under it as a sum, the pixels over it as a difference.
// - An alpha test that keeps part of a failing pixel (its colour but not
//   its depth, say) is two passes as well: the pixels that pass, then the
//   ones that fail with less written.
// - The destination alpha test (draw only where the frame's alpha has, or
//   has not, its top bit: a mask a game draws first) needs the frame's
//   alpha in the shader, which may not read what it is drawing into: the
//   pixels the primitive can touch are copied to a texture of their own
//   first, and each such primitive is drawn by itself.

#include "runtime/gs/gs_gpu_renderer.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <thread>
#include <type_traits>
#include <unordered_map>

static_assert(sizeof(GSGpu::Vertex) == 32, "the vertex layout below");
static_assert(std::has_unique_object_representations_v<GSGpu::Pipeline>, "pipelines are compared as bytes");

// Nothing but refusals where there is no desktop GL to draw with, and in
// builds without a window (PS2X_GS_NO_GPU).
#if defined(PLATFORM_VITA) || defined(__ANDROID__) || defined(PS2X_GS_NO_GPU)

namespace GSGpu
{
    struct Reading::State
    {
    };
    struct Renderer::Impl
    {
    };
    bool PrepareContext() { return false; }
    bool ContextReady() { return false; }
    bool Reading::Wait(std::vector<uint8_t> &, uint32_t &, uint32_t &) { return false; }
    std::unique_ptr<Renderer> Renderer::Create() { return nullptr; }
    Renderer::Renderer() = default;
    Renderer::~Renderer() = default;
    uint32_t Renderer::NewSurface(SurfaceKind, int, int, int, int) { return 0; }
    void Renderer::Resize(uint32_t, int, int) {}
    void Renderer::Release(uint32_t) {}
    void Renderer::SetTexture(uint32_t, int, int, const uint32_t *) {}
    void Renderer::FillColour(uint32_t, int, int, int, int, const uint32_t *) {}
    void Renderer::FillDepth(uint32_t, int, int, int, int, const float *) {}
    void Renderer::Draw(const Pipeline &, const Vertex *, size_t) {}
    void Renderer::Flush() {}
    Reading Renderer::Read(uint32_t, int, int, int, int) { return {}; }
    Reading Renderer::ReadBack(uint32_t, int, int, int, int) { return {}; }
    Reading Renderer::Show(uint32_t, int, int, int, int) { return {}; }
    bool LatestShown(Shown &) { return false; }
}

#else

#define GLFW_INCLUDE_NONE
#include "external/glfw/include/GLFW/glfw3.h"
#include "external/glad.h"

namespace GSGpu
{
    namespace
    {
        std::atomic<GLFWwindow *> g_context{nullptr};

        // What LatestShown gives: the renderer's thread writes one of two
        // textures while the window may still be drawing the other.
        std::mutex g_shownMutex;
        Shown g_shown;

        const char *const kVertexShader = R"(#version 330 core
layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aTexture;
layout(location = 2) in vec4 aColour;
layout(location = 3) in float aFog;
uniform vec4 uToClip;
noperspective out vec3 vTexture;
noperspective out vec4 vColour;
noperspective out float vFog;
invariant gl_Position;
void main()
{
    gl_Position = vec4(aPosition.xy * uToClip.xy + uToClip.zw, aPosition.z * 2.0 - 1.0, 1.0);
    vTexture = aTexture;
    vColour = aColour;
    vFog = aFog;
}
)";

        const char *const kFragmentShader = R"(#version 330 core
noperspective in vec3 vTexture;
noperspective in vec4 vColour;
noperspective in float vFog;
layout(location = 0, index = 0) out vec4 oColour;
layout(location = 0, index = 1) out vec4 oFactor;

uniform sampler2D uTexture;
uniform int uTextureMode;   // 0 none, 1 texels, 2 surface
uniform int uLinear;
uniform int uTfx;
uniform int uTcc;
uniform ivec2 uWrap;
uniform ivec4 uRegion;      // MINU, MAXU, MINV, MAXV
uniform ivec2 uTexSize;
uniform ivec2 uTexOrigin;
uniform ivec2 uTexExtent;
uniform vec2 uSurfaceTexel; // one over the surface's size
uniform int uTexAlphaFixed;
uniform int uShuffle;       // 1 both halves of the texel are its red and green, 2 its blue and alpha
uniform float uTa0;
uniform int uAem;
uniform int uFog;
uniform vec3 uFogColour;
uniform int uAlphaTest;     // 0 none, 1 keep what passes, 2 keep what fails
uniform int uAtst;
uniform float uAref;
uniform int uOutput;        // 0 the colour, 1 white
uniform vec2 uBlendSource;  // the colour is multiplied by x + y * alpha
uniform vec2 uBlendDest;    // and what is there by x + y * alpha
uniform int uBlendAlpha;    // 0 the pixel's alpha, 2 uFix
uniform float uFix;
uniform int uPabe;
uniform int uFba;
uniform int uAlphaSplit;    // 0 all, 1 only pixels whose blending alpha is at most one, 2 only the others
uniform sampler2D uFrame;   // a copy of what is being drawn into, for uDate
uniform int uDate;          // 0 none, 1 draw where the frame's alpha is below 0x80, 2 where it is not

int wrapTexel(int c, int mode, int size, int low, int high)
{
    if (mode == 0)
        return c & (size - 1);
    if (mode == 1)
        return clamp(c, 0, size - 1);
    if (mode == 2)
        return clamp(c, low, high);
    return (c & low) | high;
}

vec4 texelAt(ivec2 c)
{
    c = ivec2(wrapTexel(c.x, uWrap.x, uTexSize.x, uRegion.x, uRegion.y),
              wrapTexel(c.y, uWrap.y, uTexSize.y, uRegion.z, uRegion.w));
    c = clamp(c - uTexOrigin, ivec2(0), uTexExtent - ivec2(1));
    return floor(texelFetch(uTexture, c, 0) * 255.0 + 0.5);
}

float surfaceCoordinate(float c, int mode, float size, float low, float high, float limit)
{
    float edge = uLinear != 0 ? 0.5 : 0.001;
    if (mode == 1)
        c = clamp(c, edge, size - edge);
    else if (mode == 2)
        c = clamp(c, low + edge, high + 1.0 - edge);
    else
        c = mod(c, size);
    if (limit > 0.0)
        c = min(c, limit - edge);
    return c;
}

void main()
{
    vec4 vertex = floor(vColour * 255.0 + 0.02);
    vec3 rgb = vertex.rgb;
    float a = vertex.a;
    if (uTextureMode != 0)
    {
        vec2 uv = vTexture.xy / vTexture.z;
        vec4 t;
        if (uTextureMode == 1)
        {
            if (uLinear != 0)
            {
                vec2 st = uv - 0.5;
                vec2 first = floor(st);
                vec2 f = floor((st - first) * 256.0 + 0.5) / 256.0;
                ivec2 i = ivec2(first);
                vec4 upper = floor(mix(texelAt(i), texelAt(i + ivec2(1, 0)), f.x) + 0.5);
                vec4 lower = floor(mix(texelAt(i + ivec2(0, 1)), texelAt(i + ivec2(1, 1)), f.x) + 0.5);
                t = floor(mix(upper, lower, f.y) + 0.5);
            }
            else
            {
                t = texelAt(ivec2(uv));
            }
        }
        else
        {
            uv = vec2(surfaceCoordinate(uv.x, uWrap.x, float(uTexSize.x), float(uRegion.x), float(uRegion.y), float(uTexExtent.x)),
                      surfaceCoordinate(uv.y, uWrap.y, float(uTexSize.y), float(uRegion.z), float(uRegion.w), float(uTexExtent.y)));
            t = floor(texture(uTexture, (uv + vec2(uTexOrigin)) * uSurfaceTexel) * 255.0 + 0.5);
        }
        if (uTexAlphaFixed != 0)
            t.a = (uAem != 0 && t.rgb == vec3(0.0)) ? 0.0 : uTa0;
        if (uShuffle == 1)
            t = t.rgrg;
        else if (uShuffle == 2)
            t = t.baba;
        float va = a;
        if (uTfx == 0)
        {
            rgb = min(floor(t.rgb * rgb / 128.0), 255.0);
            a = uTcc != 0 ? min(floor(t.a * va / 128.0), 255.0) : va;
        }
        else if (uTfx == 1)
        {
            rgb = t.rgb;
            a = uTcc != 0 ? t.a : va;
        }
        else
        {
            rgb = min(floor(t.rgb * rgb / 128.0) + va, 255.0);
            a = uTcc != 0 ? (uTfx == 2 ? min(t.a + va, 255.0) : t.a) : va;
        }
    }
    if (uFog != 0)
    {
        float f = floor(vFog + 0.02);
        rgb = floor(f * rgb / 256.0) + floor((255.0 - f) * uFogColour / 256.0);
    }
    if (uDate != 0)
    {
        bool set = texelFetch(uFrame, ivec2(gl_FragCoord.xy), 0).a >= 0.5;
        if (set != (uDate == 2))
            discard;
    }
    if (uAlphaTest != 0)
    {
        bool pass = uAtst == 2 ? a < uAref : uAtst == 3 ? a <= uAref : uAtst == 4 ? a == uAref
                  : uAtst == 5 ? a >= uAref : uAtst == 6 ? a > uAref : a != uAref;
        if (pass != (uAlphaTest == 1))
            discard;
    }
    float alpha = (uBlendAlpha == 0 ? a : uFix) / 128.0;
    if (uAlphaSplit != 0 && (alpha > 1.0) != (uAlphaSplit == 2))
        discard;
    float source = max(uBlendSource.x + uBlendSource.y * alpha, 0.0);
    float dest = clamp(uBlendDest.x + uBlendDest.y * alpha, 0.0, 1.0);
    if (uPabe != 0 && a < 128.0)
    {
        source = 1.0;
        dest = 0.0;
    }
    float stored = (uFba != 0 && a < 128.0) ? a + 128.0 : a;
    if (uOutput == 1)
        oColour = vec4(1.0, 1.0, 1.0, stored / 255.0);
    else
        oColour = vec4(rgb * source / 255.0, stored / 255.0);
    oFactor = vec4(dest);
}
)";

        // A rectangle of a surface from a texture of pixels brought from
        // elsewhere, each repeated: colour, or depth from its red.
        const char *const kFillVertexShader = R"(#version 330 core
uniform vec4 uRect;
uniform vec2 uSize;
noperspective out vec2 vTexel;
void main()
{
    vec2 corner = vec2(float(gl_VertexID & 1), float((gl_VertexID >> 1) & 1));
    gl_Position = vec4(mix(uRect.xy, uRect.zw, corner), 0.0, 1.0);
    vTexel = corner * uSize;
}
)";

        const char *const kFillFragmentShader = R"(#version 330 core
noperspective in vec2 vTexel;
out vec4 oColour;
uniform sampler2D uSource;
uniform int uDepth;
void main()
{
    vec4 t = texelFetch(uSource, ivec2(vTexel), 0);
    oColour = t;
    gl_FragDepth = uDepth != 0 ? t.r : 0.0;
}
)";

        constexpr int kScratchSize = 1024;

        struct Command
        {
            enum Type : uint8_t
            {
                NewSurface,
                Resize,
                Release,
                SetTexture,
                FillColour,
                FillDepth,
                Draw,
                Read,
                ReadBack,
                Show,
            };
            Type type = Draw;
            SurfaceKind kind = SurfaceKind::Colour;
            uint32_t surface = 0;
            int x = 0, y = 0, width = 0, height = 0;
            int scaleX = 1, scaleY = 1;
            size_t first = 0; // of its vertices, or of its bytes in the blob
            size_t count = 0;
            uint32_t pipeline = 0;
            // A draw that samples the surface it draws into: the texels it can reach.
            float sampled[4] = {1.0e9f, 1.0e9f, -1.0e9f, -1.0e9f};
            std::shared_ptr<Reading::State> reading;
        };

        struct Batch
        {
            std::vector<Command> commands;
            std::vector<Vertex> vertices;
            std::vector<Pipeline> pipelines;
            std::vector<uint8_t> blob;

            bool empty() const { return commands.empty(); }
            void clear()
            {
                commands.clear();
                vertices.clear();
                pipelines.clear();
                blob.clear();
            }
        };
    }

    struct Reading::State
    {
        std::mutex mutex;
        std::condition_variable ready;
        bool done = false;
        bool good = false;
        uint32_t width = 0, height = 0;
        std::vector<uint8_t> pixels;
    };

    bool Reading::Wait(std::vector<uint8_t> &pixels, uint32_t &width, uint32_t &height)
    {
        if (!m_state)
            return false;
        std::unique_lock<std::mutex> lock(m_state->mutex);
        m_state->ready.wait(lock, [this]
                            { return m_state->done; });
        if (!m_state->good)
            return false;
        pixels.swap(m_state->pixels);
        width = m_state->width;
        height = m_state->height;
        return true;
    }

    bool PrepareContext()
    {
        if (g_context.load())
            return true;
        GLFWwindow *shown = glfwGetCurrentContext();
        if (!shown)
            return false;
        glfwDefaultWindowHints();
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#if defined(__APPLE__)
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
#endif
        GLFWwindow *hidden = glfwCreateWindow(16, 16, "gs", nullptr, shown);
        glfwDefaultWindowHints();
        glfwMakeContextCurrent(shown);
        if (!hidden)
        {
            std::fprintf(stderr, "[gs-gpu] no second GL context: drawing in software\n");
            return false;
        }
        g_context.store(hidden);
        return true;
    }

    bool ContextReady()
    {
        return g_context.load() != nullptr;
    }

    bool LatestShown(Shown &shown)
    {
        std::lock_guard<std::mutex> lock(g_shownMutex);
        shown = g_shown;
        return shown.texture != 0u;
    }

    struct Renderer::Impl
    {
        // What the caller is putting together.
        Batch pending;
        bool drawing = false; // the last command is a Draw that more vertices may join
        uint32_t nextSurface = 1;

        // Handed over.
        std::mutex mutex;
        std::condition_variable wake, taken;
        std::deque<Batch> queue;
        std::vector<Batch> spare;
        bool stopping = false;
        std::thread thread;

        // The graphics thread's.
        struct Surface
        {
            GLuint texture = 0;
            SurfaceKind kind = SurfaceKind::Colour;
            int width = 0, height = 0;
            int scaleX = 1, scaleY = 1;
        };
        std::unordered_map<uint32_t, Surface> surfaces;
        std::unordered_map<uint64_t, GLuint> framebuffers;
        bool failed = false;
        GLuint program = 0, fillProgram = 0;
        GLuint vertexArray = 0, vertexBuffer = 0;
        GLuint nearest = 0, linear = 0;
        GLuint scratchColour = 0, scratchDepth = 0, scratchFramebuffer = 0;
        std::vector<float> readDepth;
        struct
        {
            GLuint texture = 0, framebuffer = 0;
            int width = 0, height = 0;
        } shownAs[2], frameCopy;
        int shownNext = 0;
        struct
        {
            GLint toClip, texture, textureMode, linear, tfx, tcc, wrap, region, texSize, texOrigin, texExtent, surfaceTexel,
                texAlphaFixed, ta0, aem, fog, fogColour, alphaTest, atst, aref, output, blendSource, blendDest, blendAlpha, fix,
                pabe, fba, frame, date, alphaSplit, shuffle;
        } at{};
        struct
        {
            GLint rect, size, source, depth;
        } fillAt{};
        uint64_t draws = 0, passes = 0, batches = 0;

        void Run();
        bool Start();
        void Execute(Batch &batch);
        GLuint Compile(const char *vertex, const char *fragment, const char *name);
        Surface *Find(uint32_t id);
        GLuint FramebufferFor(uint32_t colour, uint32_t depth);
        void ForgetFramebuffersOf(uint32_t surface);
        void Allocate(Surface &surface);
        void DoResize(const Command &command);
        void DoFill(const Command &command, const Batch &batch);
        void DoDraw(const Pipeline &pipeline, size_t first, size_t count, const Command &command);
        void DoRead(const Command &command);
        void DoReadBack(const Command &command);
        void DoShow(const Command &command);

        Command &Add(Command::Type type)
        {
            drawing = false;
            pending.commands.emplace_back();
            pending.commands.back().type = type;
            return pending.commands.back();
        }
    };

    std::unique_ptr<Renderer> Renderer::Create()
    {
        if (!ContextReady())
            return nullptr;
        std::unique_ptr<Renderer> renderer(new Renderer());
        renderer->m_impl->thread = std::thread([impl = renderer->m_impl.get()]
                                               { impl->Run(); });
        return renderer;
    }

    Renderer::Renderer() : m_impl(new Impl())
    {
    }

    Renderer::~Renderer()
    {
        {
            std::lock_guard<std::mutex> lock(m_impl->mutex);
            m_impl->stopping = true;
        }
        m_impl->wake.notify_all();
        if (m_impl->thread.joinable())
            m_impl->thread.join();
    }

    uint32_t Renderer::NewSurface(SurfaceKind kind, int width, int height, int scaleX, int scaleY)
    {
        Command &command = m_impl->Add(Command::NewSurface);
        command.kind = kind;
        command.surface = m_impl->nextSurface++;
        command.width = width;
        command.height = height;
        command.scaleX = kind == SurfaceKind::Texture ? 1 : scaleX;
        command.scaleY = kind == SurfaceKind::Texture ? 1 : scaleY;
        return command.surface;
    }

    void Renderer::Resize(uint32_t surface, int width, int height)
    {
        Command &command = m_impl->Add(Command::Resize);
        command.surface = surface;
        command.width = width;
        command.height = height;
    }

    void Renderer::Release(uint32_t surface)
    {
        m_impl->Add(Command::Release).surface = surface;
    }

    void Renderer::SetTexture(uint32_t surface, int width, int height, const uint32_t *rgba)
    {
        Command &command = m_impl->Add(Command::SetTexture);
        command.surface = surface;
        command.width = width;
        command.height = height;
        command.first = m_impl->pending.blob.size();
        command.count = static_cast<size_t>(width) * static_cast<size_t>(height) * 4u;
        const uint8_t *bytes = reinterpret_cast<const uint8_t *>(rgba);
        m_impl->pending.blob.insert(m_impl->pending.blob.end(), bytes, bytes + command.count);
    }

    void Renderer::FillColour(uint32_t surface, int x, int y, int width, int height, const uint32_t *rgba)
    {
        Command &command = m_impl->Add(Command::FillColour);
        command.surface = surface;
        command.x = x;
        command.y = y;
        command.width = width;
        command.height = height;
        command.first = m_impl->pending.blob.size();
        command.count = static_cast<size_t>(width) * static_cast<size_t>(height) * 4u;
        const uint8_t *bytes = reinterpret_cast<const uint8_t *>(rgba);
        m_impl->pending.blob.insert(m_impl->pending.blob.end(), bytes, bytes + command.count);
    }

    void Renderer::FillDepth(uint32_t surface, int x, int y, int width, int height, const float *depth)
    {
        Command &command = m_impl->Add(Command::FillDepth);
        command.surface = surface;
        command.x = x;
        command.y = y;
        command.width = width;
        command.height = height;
        command.first = m_impl->pending.blob.size();
        command.count = static_cast<size_t>(width) * static_cast<size_t>(height) * 4u;
        const uint8_t *bytes = reinterpret_cast<const uint8_t *>(depth);
        m_impl->pending.blob.insert(m_impl->pending.blob.end(), bytes, bytes + command.count);
    }

    void Renderer::Draw(const Pipeline &pipeline, const Vertex *vertices, size_t count)
    {
        Impl &impl = *m_impl;
        if (count == 0u)
            return;
        Batch &batch = impl.pending;
        // What is there times (1 + alpha) takes two passes that only come
        // out right a primitive at a time.
        // So does a destination alpha test, which sees the frame as it
        // was before the primitive, and a texture that is the surface
        // drawn into, which is the surface as the primitive before left it.
        const bool ownTexture = pipeline.textureMode == TextureSurface && pipeline.texture == pipeline.colour;
        const bool alone = (pipeline.abe && pipeline.dsel == 1u && pipeline.asel == 1u && pipeline.bsel != 1u) || pipeline.date || ownTexture;
        if (!impl.drawing || alone || std::memcmp(&batch.pipelines.back(), &pipeline, sizeof(Pipeline)) != 0)
        {
            Command &command = impl.Add(Command::Draw);
            command.first = batch.vertices.size();
            command.pipeline = static_cast<uint32_t>(batch.pipelines.size());
            batch.pipelines.push_back(pipeline);
            impl.drawing = true;
            if (pipeline.date)
            {
                // The pixels it can touch, for the copy of the frame.
                float x0 = vertices[0].x, y0 = vertices[0].y, x1 = x0, y1 = y0;
                for (size_t i = 1; i < count; ++i)
                {
                    x0 = std::min(x0, vertices[i].x);
                    y0 = std::min(y0, vertices[i].y);
                    x1 = std::max(x1, vertices[i].x);
                    y1 = std::max(y1, vertices[i].y);
                }
                command.x = std::max(static_cast<int>(std::floor(x0)), static_cast<int>(pipeline.scissor[0]));
                command.y = std::max(static_cast<int>(std::floor(y0)), static_cast<int>(pipeline.scissor[1]));
                command.width = std::min(static_cast<int>(std::ceil(x1)), pipeline.scissor[2] + 1) - command.x;
                command.height = std::min(static_cast<int>(std::ceil(y1)), pipeline.scissor[3] + 1) - command.y;
            }
        }
        Command &drawn = batch.commands.back();
        drawn.count += count;
        if (ownTexture)
        {
            for (size_t i = 0; i < count; ++i)
            {
                const float q = vertices[i].q != 0.0f ? vertices[i].q : 1.0f;
                const float u = vertices[i].s / q + static_cast<float>(pipeline.texOrigin[0]);
                const float v = vertices[i].t / q + static_cast<float>(pipeline.texOrigin[1]);
                drawn.sampled[0] = std::min(drawn.sampled[0], u);
                drawn.sampled[1] = std::min(drawn.sampled[1], v);
                drawn.sampled[2] = std::max(drawn.sampled[2], u);
                drawn.sampled[3] = std::max(drawn.sampled[3], v);
            }
        }
        batch.vertices.insert(batch.vertices.end(), vertices, vertices + count);
        if (batch.vertices.size() >= 96000u)
            Flush();
    }

    void Renderer::Flush()
    {
        Impl &impl = *m_impl;
        if (impl.pending.empty())
            return;
        impl.drawing = false;
        {
            std::unique_lock<std::mutex> lock(impl.mutex);
            // The graphics thread far behind is a reason to wait for it
            // rather than to pile up frames of work.
            impl.taken.wait(lock, [&impl]
                            { return impl.queue.size() < 6u || impl.stopping; });
            impl.queue.push_back(std::move(impl.pending));
            if (!impl.spare.empty())
            {
                impl.pending = std::move(impl.spare.back());
                impl.spare.pop_back();
            }
            else
            {
                impl.pending = Batch{};
            }
        }
        impl.pending.clear();
        impl.wake.notify_one();
    }

    Reading Renderer::Read(uint32_t surface, int x, int y, int width, int height)
    {
        Reading reading;
        reading.m_state = std::make_shared<Reading::State>();
        Command &command = m_impl->Add(Command::Read);
        command.surface = surface;
        command.x = x;
        command.y = y;
        command.width = width;
        command.height = height;
        command.reading = reading.m_state;
        Flush();
        return reading;
    }

    Reading Renderer::Show(uint32_t surface, int x, int y, int width, int height)
    {
        Reading reading;
        reading.m_state = std::make_shared<Reading::State>();
        Command &command = m_impl->Add(Command::Show);
        command.surface = surface;
        command.x = x;
        command.y = y;
        command.width = width;
        command.height = height;
        command.reading = reading.m_state;
        Flush();
        return reading;
    }

    Reading Renderer::ReadBack(uint32_t surface, int x, int y, int width, int height)
    {
        Reading reading;
        reading.m_state = std::make_shared<Reading::State>();
        Command &command = m_impl->Add(Command::ReadBack);
        command.surface = surface;
        command.x = x;
        command.y = y;
        command.width = width;
        command.height = height;
        command.reading = reading.m_state;
        Flush();
        return reading;
    }

    void Renderer::Impl::Run()
    {
        glfwMakeContextCurrent(g_context.load());
        failed = !Start();
        if (failed)
            std::fprintf(stderr, "[gs-gpu] the renderer could not start: nothing will be drawn by it\n");
        for (;;)
        {
            Batch batch;
            {
                std::unique_lock<std::mutex> lock(mutex);
                wake.wait(lock, [this]
                          { return stopping || !queue.empty(); });
                if (queue.empty())
                    break;
                batch = std::move(queue.front());
                queue.pop_front();
            }
            taken.notify_all();
            Execute(batch);
            batch.clear();
            std::lock_guard<std::mutex> lock(mutex);
            if (spare.size() < 4u)
                spare.push_back(std::move(batch));
        }
        glfwMakeContextCurrent(nullptr);
    }

    GLuint Renderer::Impl::Compile(const char *vertex, const char *fragment, const char *name)
    {
        auto stage = [name](GLenum kind, const char *source) -> GLuint
        {
            const GLuint shader = glCreateShader(kind);
            glShaderSource(shader, 1, &source, nullptr);
            glCompileShader(shader);
            GLint good = 0;
            glGetShaderiv(shader, GL_COMPILE_STATUS, &good);
            if (!good)
            {
                char log[2048] = "";
                glGetShaderInfoLog(shader, sizeof(log), nullptr, log);
                std::fprintf(stderr, "[gs-gpu] %s %s shader: %s\n", name, kind == GL_VERTEX_SHADER ? "vertex" : "fragment", log);
                glDeleteShader(shader);
                return 0;
            }
            return shader;
        };
        const GLuint vs = stage(GL_VERTEX_SHADER, vertex);
        const GLuint fs = stage(GL_FRAGMENT_SHADER, fragment);
        if (!vs || !fs)
            return 0;
        const GLuint linked = glCreateProgram();
        glAttachShader(linked, vs);
        glAttachShader(linked, fs);
        glLinkProgram(linked);
        glDeleteShader(vs);
        glDeleteShader(fs);
        GLint good = 0;
        glGetProgramiv(linked, GL_LINK_STATUS, &good);
        if (!good)
        {
            char log[2048] = "";
            glGetProgramInfoLog(linked, sizeof(log), nullptr, log);
            std::fprintf(stderr, "[gs-gpu] %s program: %s\n", name, log);
            glDeleteProgram(linked);
            return 0;
        }
        return linked;
    }

    bool Renderer::Impl::Start()
    {
        if (!glfwGetCurrentContext())
            return false;
        program = Compile(kVertexShader, kFragmentShader, "draw");
        fillProgram = Compile(kFillVertexShader, kFillFragmentShader, "fill");
        if (!program || !fillProgram)
            return false;
#define PS2X_UNIFORM(field, name) at.field = glGetUniformLocation(program, name)
        PS2X_UNIFORM(toClip, "uToClip");
        PS2X_UNIFORM(texture, "uTexture");
        PS2X_UNIFORM(textureMode, "uTextureMode");
        PS2X_UNIFORM(linear, "uLinear");
        PS2X_UNIFORM(tfx, "uTfx");
        PS2X_UNIFORM(tcc, "uTcc");
        PS2X_UNIFORM(wrap, "uWrap");
        PS2X_UNIFORM(region, "uRegion");
        PS2X_UNIFORM(texSize, "uTexSize");
        PS2X_UNIFORM(texOrigin, "uTexOrigin");
        PS2X_UNIFORM(texExtent, "uTexExtent");
        PS2X_UNIFORM(surfaceTexel, "uSurfaceTexel");
        PS2X_UNIFORM(texAlphaFixed, "uTexAlphaFixed");
        PS2X_UNIFORM(ta0, "uTa0");
        PS2X_UNIFORM(aem, "uAem");
        PS2X_UNIFORM(fog, "uFog");
        PS2X_UNIFORM(fogColour, "uFogColour");
        PS2X_UNIFORM(alphaTest, "uAlphaTest");
        PS2X_UNIFORM(atst, "uAtst");
        PS2X_UNIFORM(aref, "uAref");
        PS2X_UNIFORM(output, "uOutput");
        PS2X_UNIFORM(blendSource, "uBlendSource");
        PS2X_UNIFORM(blendDest, "uBlendDest");
        PS2X_UNIFORM(blendAlpha, "uBlendAlpha");
        PS2X_UNIFORM(fix, "uFix");
        PS2X_UNIFORM(pabe, "uPabe");
        PS2X_UNIFORM(fba, "uFba");
        PS2X_UNIFORM(frame, "uFrame");
        PS2X_UNIFORM(date, "uDate");
        PS2X_UNIFORM(alphaSplit, "uAlphaSplit");
        PS2X_UNIFORM(shuffle, "uShuffle");
#undef PS2X_UNIFORM
        fillAt.rect = glGetUniformLocation(fillProgram, "uRect");
        fillAt.size = glGetUniformLocation(fillProgram, "uSize");
        fillAt.source = glGetUniformLocation(fillProgram, "uSource");
        fillAt.depth = glGetUniformLocation(fillProgram, "uDepth");

        glGenVertexArrays(1, &vertexArray);
        glBindVertexArray(vertexArray);
        glGenBuffers(1, &vertexBuffer);
        glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<const void *>(offsetof(Vertex, x)));
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<const void *>(offsetof(Vertex, s)));
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 4, GL_UNSIGNED_BYTE, GL_TRUE, sizeof(Vertex), reinterpret_cast<const void *>(offsetof(Vertex, r)));
        glEnableVertexAttribArray(3);
        glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<const void *>(offsetof(Vertex, fog)));

        auto sampler = [](GLenum filter)
        {
            GLuint made = 0;
            glGenSamplers(1, &made);
            glSamplerParameteri(made, GL_TEXTURE_MIN_FILTER, static_cast<GLint>(filter));
            glSamplerParameteri(made, GL_TEXTURE_MAG_FILTER, static_cast<GLint>(filter));
            glSamplerParameteri(made, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glSamplerParameteri(made, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            return made;
        };
        nearest = sampler(GL_NEAREST);
        linear = sampler(GL_LINEAR);

        glGenTextures(1, &scratchColour);
        glBindTexture(GL_TEXTURE_2D, scratchColour);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, kScratchSize, kScratchSize, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
        glGenTextures(1, &scratchDepth);
        glBindTexture(GL_TEXTURE_2D, scratchDepth);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, kScratchSize, kScratchSize, 0, GL_RED, GL_FLOAT, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
        glGenFramebuffers(1, &scratchFramebuffer);
        glBindFramebuffer(GL_FRAMEBUFFER, scratchFramebuffer);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, scratchColour, 0);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        glDisable(GL_CULL_FACE);
        glDisable(GL_DITHER);
        glDepthRange(0.0, 1.0);
        return glGetError() == GL_NO_ERROR;
    }

    Renderer::Impl::Surface *Renderer::Impl::Find(uint32_t id)
    {
        const auto found = surfaces.find(id);
        return found != surfaces.end() ? &found->second : nullptr;
    }

    void Renderer::Impl::Allocate(Surface &surface)
    {
        glGenTextures(1, &surface.texture);
        glBindTexture(GL_TEXTURE_2D, surface.texture);
        const int width = std::max(surface.width * surface.scaleX, 1), height = std::max(surface.height * surface.scaleY, 1);
        if (surface.kind == SurfaceKind::Depth)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
        else
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    }

    // A frame buffer object for a colour surface, a depth surface or both.
    GLuint Renderer::Impl::FramebufferFor(uint32_t colour, uint32_t depth)
    {
        const uint64_t key = (static_cast<uint64_t>(colour) << 32) | depth;
        const auto found = framebuffers.find(key);
        if (found != framebuffers.end())
            return found->second;
        GLuint made = 0;
        glGenFramebuffers(1, &made);
        glBindFramebuffer(GL_FRAMEBUFFER, made);
        if (const Surface *surface = colour ? Find(colour) : nullptr)
        {
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, surface->texture, 0);
            glDrawBuffer(GL_COLOR_ATTACHMENT0);
            glReadBuffer(GL_COLOR_ATTACHMENT0);
        }
        else
        {
            glDrawBuffer(GL_NONE);
            glReadBuffer(GL_NONE);
        }
        if (const Surface *surface = depth ? Find(depth) : nullptr)
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, surface->texture, 0);
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
            std::fprintf(stderr, "[gs-gpu] frame buffer of surfaces %u and %u is not complete\n", colour, depth);
        framebuffers.emplace(key, made);
        return made;
    }

    void Renderer::Impl::ForgetFramebuffersOf(uint32_t surface)
    {
        for (auto entry = framebuffers.begin(); entry != framebuffers.end();)
        {
            if (static_cast<uint32_t>(entry->first >> 32) == surface || static_cast<uint32_t>(entry->first) == surface)
            {
                glDeleteFramebuffers(1, &entry->second);
                entry = framebuffers.erase(entry);
            }
            else
            {
                ++entry;
            }
        }
    }

    void Renderer::Impl::DoResize(const Command &command)
    {
        Surface *surface = Find(command.surface);
        if (!surface || (surface->width == command.width && surface->height == command.height))
            return;
        Surface old = *surface;
        const bool depth = old.kind == SurfaceKind::Depth;
        const GLuint from = depth ? FramebufferFor(0, command.surface) : FramebufferFor(command.surface, 0);
        // The old texture stays attached to `from` until that is forgotten.
        framebuffers.erase(depth ? static_cast<uint64_t>(command.surface) : static_cast<uint64_t>(command.surface) << 32);
        ForgetFramebuffersOf(command.surface);
        surface->width = command.width;
        surface->height = command.height;
        Allocate(*surface);
        const GLuint to = depth ? FramebufferFor(0, command.surface) : FramebufferFor(command.surface, 0);
        glBindFramebuffer(GL_FRAMEBUFFER, to);
        glDisable(GL_SCISSOR_TEST);
        glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
        glDepthMask(GL_TRUE);
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
        glClearDepth(0.0);
        glClear(depth ? GL_DEPTH_BUFFER_BIT : GL_COLOR_BUFFER_BIT);
        const int width = std::min(old.width, surface->width) * old.scaleX, height = std::min(old.height, surface->height) * old.scaleY;
        glBindFramebuffer(GL_READ_FRAMEBUFFER, from);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, to);
        glBlitFramebuffer(0, 0, width, height, 0, 0, width, height, depth ? GL_DEPTH_BUFFER_BIT : GL_COLOR_BUFFER_BIT, GL_NEAREST);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glDeleteFramebuffers(1, &from);
        glDeleteTextures(1, &old.texture);
    }

    void Renderer::Impl::DoFill(const Command &command, const Batch &batch)
    {
        Surface *surface = Find(command.surface);
        const bool depth = command.type == Command::FillDepth;
        if (!surface || (surface->kind == SurfaceKind::Depth) != depth || surface->kind == SurfaceKind::Texture ||
            command.width <= 0 || command.height <= 0 || command.width > kScratchSize)
            return;
        glBindFramebuffer(GL_FRAMEBUFFER, depth ? FramebufferFor(0, command.surface) : FramebufferFor(command.surface, 0));
        glViewport(0, 0, surface->width * surface->scaleX, surface->height * surface->scaleY);
        glDisable(GL_SCISSOR_TEST);
        glDisable(GL_BLEND);
        glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
        if (depth)
        {
            glEnable(GL_DEPTH_TEST);
            glDepthFunc(GL_ALWAYS);
            glDepthMask(GL_TRUE);
        }
        else
        {
            glDisable(GL_DEPTH_TEST);
        }
        glUseProgram(fillProgram);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, depth ? scratchDepth : scratchColour);
        glBindSampler(0, nearest);
        glUniform1i(fillAt.source, 0);
        glUniform1i(fillAt.depth, depth ? 1 : 0);
        const uint8_t *pixels = batch.blob.data() + command.first;
        for (int done = 0; done < command.height; done += kScratchSize)
        {
            const int rows = std::min(command.height - done, kScratchSize);
            if (depth)
                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, command.width, rows, GL_RED, GL_FLOAT,
                                pixels + static_cast<size_t>(done) * command.width * 4u);
            else
                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, command.width, rows, GL_RGBA, GL_UNSIGNED_BYTE,
                                pixels + static_cast<size_t>(done) * command.width * 4u);
            const float toClipX = 2.0f / static_cast<float>(surface->width), toClipY = 2.0f / static_cast<float>(surface->height);
            const float x0 = static_cast<float>(command.x) * toClipX - 1.0f, x1 = static_cast<float>(command.x + command.width) * toClipX - 1.0f;
            const float y0 = static_cast<float>(command.y + done) * toClipY - 1.0f, y1 = static_cast<float>(command.y + done + rows) * toClipY - 1.0f;
            glUniform4f(fillAt.rect, x0, y0, x1, y1);
            glUniform2f(fillAt.size, static_cast<float>(command.width), static_cast<float>(rows));
            glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        }
    }

    void Renderer::Impl::DoDraw(const Pipeline &p, size_t first, size_t count, const Command &command)
    {
        if (p.ztst == 0u)
            return;
        Surface *colour = Find(p.colour);
        if (!colour || colour->kind != SurfaceKind::Colour)
            return;
        Surface *depthSurface = p.depth ? Find(p.depth) : nullptr;
        const bool depthUsed = depthSurface && depthSurface->kind == SurfaceKind::Depth && (p.ztst >= 2u || !p.zmask);
        const bool writesDepth = depthUsed && !p.zmask;

        // The alpha test: which pixels each pass keeps and what it writes of them.
        struct Pass
        {
            int alphaTest;
            bool rgb, alpha, depth;
        };
        Pass list[2];
        int listed = 0;
        const bool rgb = p.writeRgb != 0u, alpha = p.writeAlpha != 0u;
        if (!p.alphaTest || p.atst == 1u)
        {
            list[listed++] = {0, rgb, alpha, writesDepth};
        }
        else if (p.atst == 0u)
        {
            // Nothing passes: what a failing pixel still writes is all there is.
            if (p.afail == 1u)
                list[listed++] = {0, rgb, alpha, false};
            else if (p.afail == 2u)
                list[listed++] = {0, false, false, writesDepth};
            else if (p.afail == 3u)
                list[listed++] = {0, rgb, false, false};
        }
        else
        {
            switch (p.afail)
            {
            case 0u:
                list[listed++] = {1, rgb, alpha, writesDepth};
                break;
            case 1u:
                if (!writesDepth)
                {
                    list[listed++] = {0, rgb, alpha, false};
                }
                else
                {
                    list[listed++] = {1, rgb, alpha, true};
                    list[listed++] = {2, rgb, alpha, false};
                }
                break;
            case 2u:
                list[listed++] = {1, rgb, alpha, writesDepth};
                list[listed++] = {2, false, false, writesDepth};
                break;
            default:
                if (!writesDepth && !alpha)
                {
                    list[listed++] = {0, rgb, false, false};
                }
                else
                {
                    list[listed++] = {1, rgb, alpha, writesDepth};
                    list[listed++] = {2, rgb, false, false};
                }
                break;
            }
        }

        // Blending: the colour times x + y * alpha, plus (or minus, or
        // taken from) what is there times the same.
        enum class Blend
        {
            None,
            General,
            FrameAlphaAdd, // colour * the frame's alpha + what is there
            OnePlusAlpha,  // what is there * (1 + alpha), less colour * alpha perhaps
        };
        Blend blend = Blend::None;
        GLenum equation = GL_FUNC_ADD;
        float sourceX = 1.0f, sourceY = 0.0f, destX = 0.0f, destY = 0.0f;
        int alphaFrom = p.csel == 2u ? 2 : 0;
        float fix = static_cast<float>(p.fix);
        bool lessColour = false;
        if (p.abe)
        {
            int x0 = p.dsel == 0u ? 1 : 0, x1 = (p.asel == 0u ? 1 : 0) - (p.bsel == 0u ? 1 : 0);
            int y0 = p.dsel == 1u ? 1 : 0, y1 = (p.asel == 1u ? 1 : 0) - (p.bsel == 1u ? 1 : 0);
            blend = Blend::General;
            if (p.csel == 1u)
            {
                if (!p.frameAlphaOne && x0 == 0 && x1 == 1 && y0 == 1 && y1 == 0)
                {
                    blend = Blend::FrameAlphaAdd;
                }
                else
                {
                    // A 24-bit frame's alpha is 0x80 whatever is in it.
                    // Anything else this cannot do is drawn as if it were.
                    static bool told = false;
                    if (!p.frameAlphaOne && !told)
                    {
                        told = true;
                        std::fprintf(stderr, "[gs-gpu] blending %u%u%u%u with the frame's alpha is drawn with alpha 0x80\n",
                                     p.asel, p.bsel, p.csel, p.dsel);
                    }
                    alphaFrom = 2;
                    fix = 128.0f;
                }
            }
            if (blend == Blend::General)
            {
                if (y0 == 1 && y1 == 1)
                {
                    blend = Blend::OnePlusAlpha;
                    lessColour = x1 < 0;
                }
                else if (x0 == 0 && x1 < 0)
                {
                    equation = GL_FUNC_REVERSE_SUBTRACT;
                    x1 = 1;
                }
                else if (y0 == 0 && y1 < 0)
                {
                    equation = GL_FUNC_SUBTRACT;
                    y1 = 1;
                }
                sourceX = static_cast<float>(x0);
                sourceY = static_cast<float>(x1);
                destX = static_cast<float>(y0);
                destY = static_cast<float>(y1);
                // A fixed alpha past one makes 1 - alpha a subtraction for every pixel.
                if (blend == Blend::General && equation == GL_FUNC_ADD && alphaFrom == 2 && fix > 128.0f && y0 == 1 && y1 == -1)
                {
                    equation = GL_FUNC_SUBTRACT;
                    destX = -1.0f;
                    destY = 1.0f;
                }
            }
        }
        // ...and the pixel's own alpha past one, for those pixels.
        const bool split = blend == Blend::General && equation == GL_FUNC_ADD && alphaFrom == 0 && p.alphaOver && destX == 1.0f &&
                           destY == -1.0f;

        glBindFramebuffer(GL_FRAMEBUFFER, FramebufferFor(p.colour, depthUsed ? p.depth : 0u));
        glViewport(0, 0, colour->width * colour->scaleX, colour->height * colour->scaleY);
        glEnable(GL_SCISSOR_TEST);
        glScissor(p.scissor[0] * colour->scaleX, p.scissor[1] * colour->scaleY, (p.scissor[2] - p.scissor[0] + 1) * colour->scaleX,
                  (p.scissor[3] - p.scissor[1] + 1) * colour->scaleY);
        glUseProgram(program);
        glUniform4f(at.toClip, 2.0f / static_cast<float>(colour->width), 2.0f / static_cast<float>(colour->height), -1.0f, -1.0f);

        // Part of the surface being drawn into, copied where the shader can
        // read it (it may not read what it writes): the same pixels at the
        // same place in a texture at least as large.
        const int pixelsWide = colour->width * colour->scaleX, pixelsHigh = colour->height * colour->scaleY;
        auto copyFrame = [&](int left, int top, int right, int bottom)
        {
            if (frameCopy.texture == 0u || frameCopy.width < pixelsWide || frameCopy.height < pixelsHigh)
            {
                if (frameCopy.texture == 0u)
                {
                    glGenTextures(1, &frameCopy.texture);
                    glGenFramebuffers(1, &frameCopy.framebuffer);
                }
                frameCopy.width = std::max(frameCopy.width, pixelsWide);
                frameCopy.height = std::max(frameCopy.height, pixelsHigh);
                glActiveTexture(GL_TEXTURE1);
                glBindTexture(GL_TEXTURE_2D, frameCopy.texture);
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, frameCopy.width, frameCopy.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
                glBindFramebuffer(GL_FRAMEBUFFER, frameCopy.framebuffer);
                glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, frameCopy.texture, 0);
                glActiveTexture(GL_TEXTURE0);
            }
            left = std::max(left * colour->scaleX, 0);
            top = std::max(top * colour->scaleY, 0);
            right = std::min(right * colour->scaleX, pixelsWide);
            bottom = std::min(bottom * colour->scaleY, pixelsHigh);
            if (left < right && top < bottom)
            {
                glDisable(GL_SCISSOR_TEST);
                glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
                glBindFramebuffer(GL_READ_FRAMEBUFFER, FramebufferFor(p.colour, 0u));
                glBindFramebuffer(GL_DRAW_FRAMEBUFFER, frameCopy.framebuffer);
                glBlitFramebuffer(left, top, right, bottom, left, top, right, bottom, GL_COLOR_BUFFER_BIT, GL_NEAREST);
                glEnable(GL_SCISSOR_TEST);
            }
            glBindFramebuffer(GL_FRAMEBUFFER, FramebufferFor(p.colour, depthUsed ? p.depth : 0u));
        };

        const Surface *texture = p.textureMode != TextureNone ? Find(p.texture) : nullptr;
        const bool ownTexture = texture && p.textureMode == TextureSurface && p.texture == p.colour;
        if (ownTexture)
            copyFrame(static_cast<int>(std::floor(command.sampled[0])) - 2, static_cast<int>(std::floor(command.sampled[1])) - 2,
                      static_cast<int>(std::ceil(command.sampled[2])) + 2, static_cast<int>(std::ceil(command.sampled[3])) + 2);
        if (texture)
        {
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, ownTexture ? frameCopy.texture : texture->texture);
            glBindSampler(0, p.linear ? linear : nearest);
            glUniform1i(at.texture, 0);
            glUniform1i(at.textureMode, p.textureMode);
            glUniform1i(at.linear, p.linear);
            glUniform1i(at.tfx, p.tfx);
            glUniform1i(at.tcc, p.tcc);
            glUniform1i(at.shuffle, p.shuffle == 0u ? 0 : 1 + static_cast<int>((p.shuffle - 1u) & 1u));
            glUniform2i(at.wrap, p.wrapU, p.wrapV);
            glUniform4i(at.region, p.region[0], p.region[1], p.region[2], p.region[3]);
            glUniform2i(at.texSize, p.texSize[0], p.texSize[1]);
            glUniform2i(at.texOrigin, p.texOrigin[0], p.texOrigin[1]);
            glUniform2i(at.texExtent, p.texExtent[0], p.texExtent[1]);
            if (ownTexture)
                glUniform2f(at.surfaceTexel, static_cast<float>(colour->scaleX) / static_cast<float>(frameCopy.width),
                            static_cast<float>(colour->scaleY) / static_cast<float>(frameCopy.height));
            else
                glUniform2f(at.surfaceTexel, 1.0f / static_cast<float>(std::max(texture->width, 1)),
                            1.0f / static_cast<float>(std::max(texture->height, 1)));
            glUniform1i(at.texAlphaFixed, p.texAlphaFixed);
            glUniform1f(at.ta0, static_cast<float>(p.ta0));
            glUniform1i(at.aem, p.aem);
        }
        else
        {
            glUniform1i(at.textureMode, TextureNone);
        }
        glUniform1i(at.fog, p.fge);
        glUniform3f(at.fogColour, static_cast<float>(p.fogR), static_cast<float>(p.fogG), static_cast<float>(p.fogB));
        glUniform1i(at.atst, p.atst);
        glUniform1f(at.aref, static_cast<float>(p.aref));
        glUniform1i(at.blendAlpha, alphaFrom);
        glUniform1f(at.fix, fix);
        glUniform1i(at.pabe, p.abe && p.pabe && blend == Blend::General && equation == GL_FUNC_ADD ? 1 : 0);
        glUniform1i(at.fba, p.fba && !p.frameAlphaOne ? 1 : 0);
        if (p.date && !p.frameAlphaOne && command.width > 0 && command.height > 0)
        {
            copyFrame(command.x, command.y, command.x + command.width, command.y + command.height);
            glActiveTexture(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D, frameCopy.texture);
            glBindSampler(1, nearest);
            glActiveTexture(GL_TEXTURE0);
            glUniform1i(at.frame, 1);
            glUniform1i(at.date, p.datm ? 2 : 1);
        }
        else
        {
            glUniform1i(at.date, 0);
        }

        if (depthUsed)
        {
            glEnable(GL_DEPTH_TEST);
            glDepthFunc(p.ztst == 2u ? GL_GEQUAL : (p.ztst == 3u ? GL_GREATER : GL_ALWAYS));
        }
        else
        {
            glDisable(GL_DEPTH_TEST);
        }

        ++draws;
        for (int i = 0; i < listed; ++i)
        {
            const Pass &pass = list[i];
            if (!pass.rgb && !pass.alpha && !pass.depth)
                continue;
            glUniform1i(at.alphaTest, pass.alphaTest);
            auto draw = [&](bool writeRgb, bool writeAlpha, bool writeDepth)
            {
                // A channel shuffle writes one half of the pixel.
                const bool half = p.shuffle != 0u, high = half && ((p.shuffle - 1u) & 2u) != 0u;
                if (half)
                    glColorMask(!high, !high, high, high);
                else
                    glColorMask(writeRgb, writeRgb, writeRgb, writeAlpha);
                glDepthMask(writeDepth ? GL_TRUE : GL_FALSE);
                glDrawArrays(GL_TRIANGLES, static_cast<GLint>(first), static_cast<GLsizei>(count));
                ++passes;
            };
            switch (blend)
            {
            case Blend::None:
                glDisable(GL_BLEND);
                glUniform1i(at.output, 0);
                glUniform2f(at.blendSource, 1.0f, 0.0f);
                glUniform2f(at.blendDest, 0.0f, 0.0f);
                draw(pass.rgb, pass.alpha, pass.depth);
                break;
            case Blend::General:
                glEnable(GL_BLEND);
                glBlendEquationSeparate(equation, GL_FUNC_ADD);
                glBlendFuncSeparate(GL_ONE, GL_SRC1_COLOR, GL_ONE, GL_ZERO);
                glUniform1i(at.output, 0);
                glUniform2f(at.blendSource, sourceX, sourceY);
                glUniform2f(at.blendDest, destX, destY);
                if (!split)
                {
                    draw(pass.rgb, pass.alpha, pass.depth);
                    break;
                }
                glUniform1i(at.alphaSplit, 1);
                draw(pass.rgb, pass.alpha, pass.depth);
                glUniform1i(at.alphaSplit, 2);
                glBlendEquationSeparate(GL_FUNC_SUBTRACT, GL_FUNC_ADD);
                glUniform2f(at.blendDest, -1.0f, 1.0f);
                draw(pass.rgb, pass.alpha, pass.depth);
                glUniform1i(at.alphaSplit, 0);
                break;
            case Blend::FrameAlphaAdd:
                // The frame's alpha is 0x80 for one, which reads as a half:
                // added twice, the second time storing the pixel's own.
                glEnable(GL_BLEND);
                glBlendEquationSeparate(GL_FUNC_ADD, GL_FUNC_ADD);
                glUniform1i(at.output, 0);
                glUniform2f(at.blendSource, 1.0f, 0.0f);
                glUniform2f(at.blendDest, 0.0f, 0.0f);
                if (pass.rgb)
                {
                    glBlendFuncSeparate(GL_DST_ALPHA, GL_ONE, GL_ZERO, GL_ONE);
                    draw(true, false, false);
                }
                glBlendFuncSeparate(GL_DST_ALPHA, GL_ONE, GL_ONE, GL_ZERO);
                draw(pass.rgb, pass.alpha, pass.depth);
                break;
            case Blend::OnePlusAlpha:
                // More than what is there cannot be had from it in one
                // go: half the result first (which nothing along the way
                // can exceed), then that doubled.
                glEnable(GL_BLEND);
                if (pass.rgb)
                {
                    glBlendEquationSeparate(GL_FUNC_REVERSE_SUBTRACT, GL_FUNC_ADD);
                    glBlendFuncSeparate(GL_ONE, GL_SRC1_COLOR, GL_ZERO, GL_ONE);
                    glUniform1i(at.output, 0);
                    glUniform2f(at.blendSource, 0.0f, lessColour ? 0.5f : 0.0f);
                    glUniform2f(at.blendDest, 0.5f, 0.5f);
                    draw(true, false, false);
                }
                glBlendEquationSeparate(GL_FUNC_ADD, GL_FUNC_ADD);
                glBlendFuncSeparate(GL_DST_COLOR, GL_ONE, GL_ONE, GL_ZERO);
                glUniform1i(at.output, 1);
                glUniform2f(at.blendSource, 1.0f, 0.0f);
                glUniform2f(at.blendDest, 0.0f, 0.0f);
                draw(pass.rgb, pass.alpha, pass.depth);
                break;
            }
        }
    }

    void Renderer::Impl::DoRead(const Command &command)
    {
        Reading::State &state = *command.reading;
        std::vector<uint8_t> pixels;
        uint32_t width = 0, height = 0;
        bool good = false;
        const Surface *surface = failed ? nullptr : Find(command.surface);
        if (surface && surface->kind == SurfaceKind::Colour && command.width > 0 && command.height > 0)
        {
            width = static_cast<uint32_t>(command.width * surface->scaleX);
            height = static_cast<uint32_t>(command.height * surface->scaleY);
            pixels.assign(static_cast<size_t>(width) * height * 4u, 0u);
            // What of it the surface has; the rest stays black.
            const int x = command.x * surface->scaleX, y = command.y * surface->scaleY;
            const int has = std::min(static_cast<int>(height), surface->height * surface->scaleY - y);
            const int across = std::min(static_cast<int>(width), surface->width * surface->scaleX - x);
            if (has > 0 && across > 0 && x >= 0 && y >= 0)
            {
                glBindFramebuffer(GL_FRAMEBUFFER, FramebufferFor(command.surface, 0));
                glPixelStorei(GL_PACK_ROW_LENGTH, static_cast<GLint>(width));
                glReadPixels(x, y, across, has, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
                glPixelStorei(GL_PACK_ROW_LENGTH, 0);
                good = true;
            }
        }
        {
            std::lock_guard<std::mutex> lock(state.mutex);
            state.pixels.swap(pixels);
            state.width = width;
            state.height = height;
            state.good = good;
            state.done = true;
        }
        state.ready.notify_all();
    }

    // The game's pixels of a surface: colour scaled down into the scratch
    // texture and read from there, depth read at full size and one in
    // every so many kept.
    void Renderer::Impl::DoReadBack(const Command &command)
    {
        Reading::State &state = *command.reading;
        std::vector<uint8_t> pixels;
        bool good = false;
        const Surface *surface = failed ? nullptr : Find(command.surface);
        const int width = command.width, height = command.height;
        if (surface && surface->kind != SurfaceKind::Texture && width > 0 && height > 0 && command.x >= 0 && command.y >= 0 &&
            command.x + width <= surface->width && command.y + height <= surface->height && width <= kScratchSize)
        {
            pixels.assign(static_cast<size_t>(width) * static_cast<size_t>(height) * 4u, 0u);
            const int sx = surface->scaleX, sy = surface->scaleY;
            glDisable(GL_SCISSOR_TEST);
            if (surface->kind == SurfaceKind::Colour)
            {
                if (sx == 1 && sy == 1)
                {
                    glBindFramebuffer(GL_FRAMEBUFFER, FramebufferFor(command.surface, 0));
                    glReadPixels(command.x, command.y, width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
                }
                else
                {
                    const GLuint from = FramebufferFor(command.surface, 0);
                    for (int done = 0; done < height; done += kScratchSize)
                    {
                        const int rows = std::min(height - done, kScratchSize);
                        glBindFramebuffer(GL_READ_FRAMEBUFFER, from);
                        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, scratchFramebuffer);
                        glBlitFramebuffer(command.x * sx, (command.y + done) * sy, (command.x + width) * sx, (command.y + done + rows) * sy, 0, 0,
                                          width, rows, GL_COLOR_BUFFER_BIT, GL_LINEAR);
                        glBindFramebuffer(GL_FRAMEBUFFER, scratchFramebuffer);
                        glReadPixels(0, 0, width, rows, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data() + static_cast<size_t>(done) * width * 4u);
                    }
                }
            }
            else
            {
                readDepth.resize(static_cast<size_t>(width) * sx * static_cast<size_t>(height) * sy);
                glBindFramebuffer(GL_FRAMEBUFFER, FramebufferFor(0, command.surface));
                glReadPixels(command.x * sx, command.y * sy, width * sx, height * sy, GL_DEPTH_COMPONENT, GL_FLOAT, readDepth.data());
                float *out = reinterpret_cast<float *>(pixels.data());
                for (int y = 0; y < height; ++y)
                {
                    const float *row = readDepth.data() + (static_cast<size_t>(y) * sy + sy / 2) * (static_cast<size_t>(width) * sx);
                    for (int x = 0; x < width; ++x)
                        *out++ = row[static_cast<size_t>(x) * sx + sx / 2];
                }
            }
            good = true;
        }
        {
            std::lock_guard<std::mutex> lock(state.mutex);
            state.pixels.swap(pixels);
            state.width = static_cast<uint32_t>(width);
            state.height = static_cast<uint32_t>(height);
            state.good = good;
            state.done = true;
        }
        state.ready.notify_all();
    }

    // Part of a surface copied to the texture the window draws next, made
    // opaque (the frame buffer's alpha is the game's business), and handed
    // over once the card has been told all of it.
    void Renderer::Impl::DoShow(const Command &command)
    {
        Reading::State &state = *command.reading;
        bool good = false;
        uint32_t width = 0, height = 0;
        const Surface *surface = failed ? nullptr : Find(command.surface);
        if (surface && surface->kind == SurfaceKind::Colour && command.width > 0 && command.height > 0)
        {
            width = static_cast<uint32_t>(command.width * surface->scaleX);
            height = static_cast<uint32_t>(command.height * surface->scaleY);
            auto &target = shownAs[shownNext];
            if (target.texture == 0u || target.width != static_cast<int>(width) || target.height != static_cast<int>(height))
            {
                if (target.texture == 0u)
                {
                    glGenTextures(1, &target.texture);
                    glGenFramebuffers(1, &target.framebuffer);
                }
                target.width = static_cast<int>(width);
                target.height = static_cast<int>(height);
                glBindTexture(GL_TEXTURE_2D, target.texture);
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, target.width, target.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
                glBindFramebuffer(GL_FRAMEBUFFER, target.framebuffer);
                glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, target.texture, 0);
            }
            glDisable(GL_SCISSOR_TEST);
            glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
            glBindFramebuffer(GL_FRAMEBUFFER, target.framebuffer);
            glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT);
            // What of it the surface has; the rest stays black.
            const int x = command.x * surface->scaleX, y = command.y * surface->scaleY;
            const int across = std::min(static_cast<int>(width), surface->width * surface->scaleX - x);
            const int down = std::min(static_cast<int>(height), surface->height * surface->scaleY - y);
            if (across > 0 && down > 0 && x >= 0 && y >= 0)
            {
                glBindFramebuffer(GL_READ_FRAMEBUFFER, FramebufferFor(command.surface, 0));
                glBindFramebuffer(GL_DRAW_FRAMEBUFFER, target.framebuffer);
                glBlitFramebuffer(x, y, x + across, y + down, 0, 0, across, down, GL_COLOR_BUFFER_BIT, GL_NEAREST);
                glBindFramebuffer(GL_FRAMEBUFFER, target.framebuffer);
                glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_TRUE);
                glClear(GL_COLOR_BUFFER_BIT);
                glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
            }
            glFlush();
            {
                std::lock_guard<std::mutex> lock(g_shownMutex);
                g_shown.texture = target.texture;
                g_shown.width = width;
                g_shown.height = height;
                ++g_shown.serial;
            }
            shownNext ^= 1;
            good = true;
        }
        {
            std::lock_guard<std::mutex> lock(state.mutex);
            state.width = width;
            state.height = height;
            state.good = good;
            state.done = true;
        }
        state.ready.notify_all();
    }

    void Renderer::Impl::Execute(Batch &batch)
    {
        if (failed)
        {
            for (const Command &command : batch.commands)
            {
                if (command.type == Command::Read)
                    DoRead(command);
                else if (command.type == Command::ReadBack)
                    DoReadBack(command);
                else if (command.type == Command::Show)
                    DoShow(command);
            }
            return;
        }
        ++batches;
        if (!batch.vertices.empty())
        {
            glBindVertexArray(vertexArray);
            glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
            glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(batch.vertices.size() * sizeof(Vertex)), batch.vertices.data(),
                         GL_STREAM_DRAW);
        }
        for (const Command &command : batch.commands)
        {
            switch (command.type)
            {
            case Command::NewSurface:
            {
                Surface surface;
                surface.kind = command.kind;
                surface.width = command.width;
                surface.height = command.height;
                surface.scaleX = command.scaleX;
                surface.scaleY = command.scaleY;
                Allocate(surface);
                if (surface.kind != SurfaceKind::Texture)
                {
                    surfaces[command.surface] = surface;
                    const bool depth = surface.kind == SurfaceKind::Depth;
                    glBindFramebuffer(GL_FRAMEBUFFER, depth ? FramebufferFor(0, command.surface) : FramebufferFor(command.surface, 0));
                    glDisable(GL_SCISSOR_TEST);
                    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
                    glDepthMask(GL_TRUE);
                    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
                    glClearDepth(0.0);
                    glClear(depth ? GL_DEPTH_BUFFER_BIT : GL_COLOR_BUFFER_BIT);
                }
                else
                {
                    surfaces[command.surface] = surface;
                }
                break;
            }
            case Command::Resize:
                DoResize(command);
                break;
            case Command::Release:
                if (Surface *surface = Find(command.surface))
                {
                    ForgetFramebuffersOf(command.surface);
                    glDeleteTextures(1, &surface->texture);
                    surfaces.erase(command.surface);
                }
                break;
            case Command::SetTexture:
                if (Surface *surface = Find(command.surface); surface && surface->kind == SurfaceKind::Texture)
                {
                    glBindTexture(GL_TEXTURE_2D, surface->texture);
                    if (surface->width == command.width && surface->height == command.height)
                    {
                        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, command.width, command.height, GL_RGBA, GL_UNSIGNED_BYTE,
                                        batch.blob.data() + command.first);
                    }
                    else
                    {
                        surface->width = command.width;
                        surface->height = command.height;
                        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, command.width, command.height, 0, GL_RGBA, GL_UNSIGNED_BYTE,
                                     batch.blob.data() + command.first);
                    }
                }
                break;
            case Command::FillColour:
            case Command::FillDepth:
                DoFill(command, batch);
                break;
            case Command::Draw:
                DoDraw(batch.pipelines[command.pipeline], command.first, command.count, command);
                break;
            case Command::Read:
                DoRead(command);
                break;
            case Command::ReadBack:
                DoReadBack(command);
                break;
            case Command::Show:
                DoShow(command);
                break;
            }
        }
        glFlush();
        static const bool stats = std::getenv("PS2X_GS_STATS") != nullptr;
        if (stats && (batches % 2000u) == 0u)
            std::fprintf(stderr, "[gs-gpu] %llu batches, %llu draws, %llu passes, %zu surfaces\n", static_cast<unsigned long long>(batches),
                         static_cast<unsigned long long>(draws), static_cast<unsigned long long>(passes), surfaces.size());
    }
}

#endif
