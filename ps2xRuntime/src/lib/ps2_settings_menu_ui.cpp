#include "runtime/ps2_settings_menu.h"

// The settings menu drawn (runtime/ps2_settings_menu.h): a panel over the
// game's picture, laid out in units of the window's height so that it is the
// same at any size. Built with the debug windows, whose library it uses; it
// draws on that library's foreground list and has no window of its own, so
// it takes no input and nothing of the debug windows' state.

#if defined(PS2X_ENABLE_DEBUG_UI) && !defined(PLATFORM_VITA)

#include "imgui.h"
#include "raylib.h"
#include "rlgl.h"

#include <algorithm>
#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace
{
    ImFont *g_font = nullptr; // a face of the system's; the library's own when there is none

    // PS2X_UI_FONT, else the first of the system's that is there.
    std::string fontFile()
    {
        if (const char *named = std::getenv("PS2X_UI_FONT"); named && *named)
            return named;
        std::vector<std::string> candidates;
#if defined(_WIN32)
        const char *windows = std::getenv("WINDIR");
        const std::string fonts = std::string(windows && *windows ? windows : "C:\\Windows") + "\\Fonts\\";
        candidates = {fonts + "segoeui.ttf", fonts + "arial.ttf", fonts + "tahoma.ttf"};
#elif defined(__APPLE__)
        candidates = {"/System/Library/Fonts/SFNS.ttf", "/System/Library/Fonts/HelveticaNeue.ttc",
                      "/System/Library/Fonts/Supplemental/Arial.ttf"};
#else
        candidates = {"/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/TTF/DejaVuSans.ttf",
                      "/usr/share/fonts/dejavu/DejaVuSans.ttf",
                      "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"};
#endif
        for (const std::string &candidate : candidates)
            if (FileExists(candidate.c_str()))
                return candidate;
        return {};
    }

    constexpr ImU32 kPanel = IM_COL32(17, 17, 21, 240);
    constexpr ImU32 kEdge = IM_COL32(255, 255, 255, 34);
    constexpr ImU32 kRule = IM_COL32(255, 255, 255, 26);
    constexpr ImU32 kTitle = IM_COL32(255, 255, 255, 255);
    constexpr ImU32 kText = IM_COL32(226, 226, 232, 255);
    constexpr ImU32 kDim = IM_COL32(150, 150, 162, 255);
    constexpr ImU32 kFaint = IM_COL32(96, 96, 108, 255);
    constexpr ImU32 kKeys = IM_COL32(128, 128, 140, 255);
    constexpr ImU32 kAccent = IM_COL32(222, 176, 92, 255);
    constexpr ImU32 kPicked = IM_COL32(222, 176, 92, 44);
    constexpr ImU32 kWaits = IM_COL32(236, 190, 110, 255);

    struct Pen
    {
        ImDrawList *list;
        ImFont *font;

        ImVec2 size(float height, const std::string &text, float wrap = 0.0f) const
        {
            return font->CalcTextSizeA(height, FLT_MAX, wrap, text.c_str(), text.c_str() + text.size());
        }

        void text(float height, ImVec2 at, ImU32 colour, const std::string &text, float wrap = 0.0f) const
        {
            list->AddText(font, height, at, colour, text.c_str(), text.c_str() + text.size(), wrap);
        }

        // A small arrow-head pointing left (-1) or right (1), its middle at `at`.
        void arrow(ImVec2 at, float reach, int towards, ImU32 colour) const
        {
            const float tip = at.x + reach * 0.5f * static_cast<float>(towards);
            const float back = at.x - reach * 0.5f * static_cast<float>(towards);
            list->AddTriangleFilled(ImVec2(tip, at.y), ImVec2(back, at.y - reach * 0.7f), ImVec2(back, at.y + reach * 0.7f),
                                    colour);
        }
    };

    void drawNotice(const Pen &pen, const ps2x::SettingsMenuState &menu, ImVec2 screen, float unit)
    {
        const float height = 17.0f * unit;
        const ImVec2 size = pen.size(height, menu.notice);
        const ImVec2 at(22.0f * unit, screen.y - 22.0f * unit - size.y);
        const float strength = std::clamp(menu.noticeStrength, 0.0f, 1.0f);
        const ImVec2 pad(10.0f * unit, 6.0f * unit);
        pen.list->AddRectFilled(ImVec2(at.x - pad.x, at.y - pad.y), ImVec2(at.x + size.x + pad.x, at.y + size.y + pad.y),
                                IM_COL32(0, 0, 0, static_cast<int>(150.0f * strength)), 6.0f * unit);
        pen.text(height, at, IM_COL32(240, 240, 244, static_cast<int>(235.0f * strength)), menu.notice);
    }
}

void ps2xSetupSettingsMenuUi()
{
    const std::string file = fontFile();
    if (!file.empty())
        g_font = ImGui::GetIO().Fonts->AddFontFromFileTTF(file.c_str(), 20.0f);
}

void ps2xDrawSettingsMenu(const ps2x::SettingsMenuState &menu)
{
    const ImVec2 screen = ImGui::GetIO().DisplaySize;
    if (screen.x < 64.0f || screen.y < 64.0f)
        return;
    const float unit = std::clamp(screen.y / 720.0f, 0.6f, 4.0f);
    const Pen pen{ImGui::GetForegroundDrawList(), g_font ? g_font : ImGui::GetFont()};
    if (!menu.open)
    {
        if (!menu.notice.empty() && menu.noticeStrength > 0.0f)
            drawNotice(pen, menu, screen, unit);
        return;
    }

    const float body = 20.0f * unit, small = 16.0f * unit, title = 27.0f * unit;
    const float pad = 30.0f * unit, rowHeight = 38.0f * unit;
    const float width = std::min(screen.x - 32.0f * unit, 800.0f * unit);
    const float height = std::min(screen.y - 32.0f * unit, 560.0f * unit);
    const ImVec2 corner((screen.x - width) * 0.5f, (screen.y - height) * 0.5f);
    const float left = corner.x + pad, right = corner.x + width - pad, bottom = corner.y + height - pad;

    // The picture dimmed, and the panel.
    pen.list->AddRectFilled(ImVec2(0.0f, 0.0f), screen, IM_COL32(0, 0, 0, 120));
    pen.list->AddRectFilled(corner, ImVec2(corner.x + width, corner.y + height), kPanel, 10.0f * unit);
    pen.list->AddRect(corner, ImVec2(corner.x + width, corner.y + height), kEdge, 10.0f * unit, 0, 1.0f * unit);

    // The heading, and the pages beside it.
    float y = corner.y + pad * 0.85f;
    pen.text(title, ImVec2(left, y), kTitle, "Settings");
    {
        float x = right;
        for (size_t n = menu.pages.size(); n-- > 0;)
        {
            const ImVec2 size = pen.size(body, menu.pages[n]);
            x -= size.x;
            const float top = y + title - body - 2.0f * unit;
            pen.text(body, ImVec2(x, top), n == menu.page ? kTitle : kDim, menu.pages[n]);
            if (n == menu.page)
                pen.list->AddRectFilled(ImVec2(x, top + size.y + 4.0f * unit), ImVec2(x + size.x, top + size.y + 6.5f * unit),
                                        kAccent, 1.0f * unit);
            x -= 26.0f * unit;
        }
    }
    y += title + 16.0f * unit;
    pen.list->AddLine(ImVec2(left, y), ImVec2(right, y), kRule, 1.0f * unit);
    y += 12.0f * unit;

    // From the bottom: which buttons do what, and the picked line's words.
    const float keysTop = bottom - small;
    pen.text(small, ImVec2(left, keysTop), kKeys, menu.keys);
    const float wordsHeight = body * 1.3f * 2.0f;
    const float wordsTop = keysTop - 14.0f * unit - wordsHeight;
    pen.text(body * 0.92f, ImVec2(left, wordsTop), kDim, menu.description, right - left);
    const float ruleY = wordsTop - 12.0f * unit;
    pen.list->AddLine(ImVec2(left, ruleY), ImVec2(right, ruleY), kRule, 1.0f * unit);

    // The lines, as many as there is room for, the picked one among them.
    const float room = ruleY - 8.0f * unit - y;
    const size_t fit = std::max<size_t>(1u, static_cast<size_t>(room / rowHeight));
    static size_t first = 0;
    static size_t shownPage = 0;
    if (shownPage != menu.page)
        shownPage = menu.page, first = 0;
    if (menu.line < first)
        first = menu.line;
    if (menu.line >= first + fit)
        first = menu.line + 1u - fit;
    first = std::min(first, menu.lines.size() > fit ? menu.lines.size() - fit : 0u);

    const float valuesLeft = corner.x + width * 0.50f, valuesRight = right - 6.0f * unit;
    for (size_t n = first; n < menu.lines.size() && n < first + fit; ++n)
    {
        const ps2x::SettingsMenuLine &line = menu.lines[n];
        const float top = y + static_cast<float>(n - first) * rowHeight;
        const float middle = top + rowHeight * 0.5f;
        const bool picked = n == menu.line;
        if (picked)
        {
            pen.list->AddRectFilled(ImVec2(left - 12.0f * unit, top + 2.0f * unit),
                                    ImVec2(right + 12.0f * unit, top + rowHeight - 2.0f * unit), kPicked, 6.0f * unit);
            pen.list->AddRectFilled(ImVec2(left - 12.0f * unit, top + 7.0f * unit),
                                    ImVec2(left - 9.0f * unit, top + rowHeight - 7.0f * unit), kAccent, 1.5f * unit);
        }
        const bool does = line.value.empty();
        pen.text(body, ImVec2(left + 4.0f * unit, middle - body * 0.5f), does && !picked ? kDim : picked ? kTitle : kText,
                 line.label);
        if (does)
            continue;
        const std::string value = line.waits ? line.value + "  (next start)" : line.value;
        const ImVec2 size = pen.size(body, value);
        pen.text(body, ImVec2((valuesLeft + valuesRight - size.x) * 0.5f, middle - body * 0.5f),
                 line.waits ? kWaits : picked ? kTitle : kText, value);
        const float reach = 8.0f * unit;
        pen.arrow(ImVec2(valuesLeft + reach, middle), reach, -1, line.before ? (picked ? kAccent : kDim) : kFaint & 0x60FFFFFFu);
        pen.arrow(ImVec2(valuesRight - reach, middle), reach, 1, line.after ? (picked ? kAccent : kDim) : kFaint & 0x60FFFFFFu);
    }
    // More above or below than there is room for.
    if (first > 0u)
        pen.list->AddTriangleFilled(ImVec2(right + 2.0f * unit, y + 2.0f * unit), ImVec2(right + 10.0f * unit, y + 10.0f * unit),
                                    ImVec2(right - 6.0f * unit, y + 10.0f * unit), kDim);
    if (first + fit < menu.lines.size())
        pen.list->AddTriangleFilled(ImVec2(right + 2.0f * unit, ruleY - 4.0f * unit),
                                    ImVec2(right - 6.0f * unit, ruleY - 12.0f * unit),
                                    ImVec2(right + 10.0f * unit, ruleY - 12.0f * unit), kDim);
}

void ps2xGrabSettingsMenu(const ps2x::SettingsMenuState &menu)
{
    static const std::string prefix = []
    {
        const char *named = std::getenv("PS2X_SETTINGS_MENU_GRAB");
        return std::string(named ? named : "");
    }();
    if (prefix.empty())
        return;
    std::string shows = menu.open ? "open" : menu.notice.empty() ? "shut" : "notice";
    if (menu.open)
    {
        shows += " " + std::to_string(menu.page) + " " + std::to_string(menu.line);
        for (const ps2x::SettingsMenuLine &line : menu.lines)
            shows += "|" + line.value + (line.waits ? "*" : "");
    }
    static std::string last;
    static int count = 0;
    static int wait = -1;
    if (shows != last)
    {
        last = shows;
        wait = 3; // a few frames on, when the picture behind has caught up
    }
    if (wait < 0 || wait-- > 0)
        return;
    rlDrawRenderBatchActive(); // what was drawn is still queued
    Image shot{};
    shot.width = GetRenderWidth();
    shot.height = GetRenderHeight();
    shot.mipmaps = 1;
    shot.format = PIXELFORMAT_UNCOMPRESSED_R8G8B8A8;
    shot.data = rlReadScreenPixels(shot.width, shot.height);
    char name[32];
    std::snprintf(name, sizeof(name), "_%03d.png", count++);
    ExportImage(shot, (prefix + name).c_str());
    UnloadImage(shot);
}

#else

void ps2xSetupSettingsMenuUi() {}
void ps2xDrawSettingsMenu(const ps2x::SettingsMenuState &) {}
void ps2xGrabSettingsMenu(const ps2x::SettingsMenuState &) {}

#endif
