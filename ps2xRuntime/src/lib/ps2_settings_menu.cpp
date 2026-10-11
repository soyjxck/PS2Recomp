#include "runtime/ps2_settings_menu.h"
#include "runtime/ps2_display_mode.h"
#include "runtime/ps2_input.h"
#include "runtime/ps2_settings.h"
#include "runtime/ps2_settings_store.h"
#include "ps2_host_backend.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

// The settings menu's lines and what the pad does to them. See
// runtime/ps2_settings_menu.h; the drawing is ps2_settings_menu_ui.cpp.

namespace
{
    // The pad word's bits, a bit set for a button that is down.
    constexpr uint16_t kUp = 0x0010u, kDown = 0x0040u, kLeft = 0x0080u, kRight = 0x0020u;
    constexpr uint16_t kCross = 0x4000u, kCircle = 0x2000u, kL1 = 0x0400u, kR1 = 0x0800u, kStart = 0x0008u;

    constexpr double kChordSeconds = 0.3;   // L3 and R3 held this long
    constexpr double kRepeatAfter = 0.40;   // a direction held repeats after this
    constexpr double kRepeatEvery = 0.11;
    constexpr double kStartNotice = 7.0;    // how long the line at the start shows
    constexpr int kFramesToShow = 4;        // frames the game runs for a change to show

    using ps2x::settings_detail::lowered;
    using ps2x::settings_detail::trimmed;

    struct Row
    {
        ps2x::SettingRow row;
        int running = 0;   // the choice in force; -1 for `other`
        int shown = 0;     // the choice the menu has
        std::string other; // a value that is none of the choices
    };

    struct Menu
    {
        std::vector<Row> rows;
        bool runtimeRows = false; // the runtime's own are in
        bool valued = false;      // the rows have been given their values
        bool drawn = false;       // something has asked for the state
        size_t page = 0, line = 0;
        uint16_t held = 0;
        uint16_t repeating = 0;
        double repeatAt = 0.0;
        double chordSince = -1.0;
        bool chordSpent = false;
        bool unsaved = false;
        double begun = -1.0;
        std::string notice;
        double noticeUntil = 0.0;
        ps2x::SettingsMenuState state;
    };

    Menu &menu()
    {
        static Menu instance;
        return instance;
    }

    std::atomic<bool> g_open{false};
    std::atomic<int> g_frames{0}; // frames the game may run while the menu is open

    bool scripted()
    {
        static const bool is = []
        {
            const char *script = std::getenv("PS2X_PAD_SCRIPT");
            return script != nullptr && script[0] != '\0';
        }();
        return is;
    }

    bool isNumber(const std::string &text, double &number)
    {
        char *end = nullptr;
        number = std::strtod(text.c_str(), &end);
        return end != text.c_str() && *end == '\0';
    }

    // The choice a value written in the file or the environment means: the
    // same word, the same number written another way, or on and off however
    // they are written. -1 when it is none of them.
    int choiceFor(const ps2x::SettingRow &row, const std::string &written)
    {
        const std::string value = lowered(trimmed(written));
        for (size_t n = 0; n < row.choices.size(); ++n)
            if (lowered(row.choices[n].value) == value)
                return static_cast<int>(n);
        double number = 0.0, other = 0.0;
        if (isNumber(value, number))
            for (size_t n = 0; n < row.choices.size(); ++n)
                if (isNumber(row.choices[n].value, other) && other == number)
                    return static_cast<int>(n);
        int on = -1, off = -1;
        for (size_t n = 0; n < row.choices.size(); ++n)
        {
            const std::string choice = lowered(row.choices[n].value);
            if (choice == "on")
                on = static_cast<int>(n);
            else if (choice == "off")
                off = static_cast<int>(n);
        }
        if (on >= 0 && off >= 0 && row.choices.size() == 2u)
            return value == "on" || value == "yes" || value == "true" || std::atof(value.c_str()) != 0.0 ? on : off;
        return -1;
    }

    void giveValue(Row &row, const char *written)
    {
        row.other.clear();
        if (!written || !*written)
            row.running = static_cast<int>(std::min(row.row.otherwise, row.row.choices.size() - 1u));
        else if ((row.running = choiceFor(row.row, written)) < 0)
            row.other = written;
        row.shown = row.running;
    }

    // ---- the runtime's own lines: the picture -----------------------------

    void addRuntimeRows(Menu &all)
    {
        if (all.runtimeRows)
            return;
        all.runtimeRows = true;
        const char *page = "Graphics";
        std::vector<ps2x::SettingRow> rows;
        {
            ps2x::SettingRow row;
            row.page = page;
            row.key = "display.fullscreen";
            row.environment = "PS2X_FULLSCREEN";
            row.label = "Display Mode";
            row.description = "A window, the whole screen without a frame (borderless), or the screen taken over (full screen). "
                              "F11 switches as well.";
            row.choices = {{"off", "Window"}, {"borderless", "Borderless"}, {"exclusive", "Full Screen"}};
            row.apply = [](const std::string &value)
            { ps2x::setDisplayMode(ps2x::displayModeFrom(value.c_str(), ps2x::DisplayMode::Window)); };
            row.current = []
            { return std::string(ps2x::displayModeValue(ps2x::displayMode())); };
            rows.push_back(std::move(row));
        }
        {
            ps2x::SettingRow row;
            row.page = page;
            row.key = "display.window";
            row.environment = "PS2X_WINDOW";
            row.label = "Window Size";
            row.description = "The size of the window when the game is not on the whole screen.";
            row.choices = {{"960x540", "960 x 540"},     {"1280x720", "1280 x 720"},   {"1600x900", "1600 x 900"},
                           {"1920x1080", "1920 x 1080"}, {"2560x1440", "2560 x 1440"}, {"3840x2160", "3840 x 2160"}};
            row.otherwise = 1;
            row.apply = [](const std::string &value)
            {
                int width = 0, height = 0;
                if (std::sscanf(value.c_str(), "%dx%d", &width, &height) == 2 && width >= 64 && height >= 64 &&
                    ps2x::displayMode() == ps2x::DisplayMode::Window)
                    SetWindowSize(width, height);
            };
            rows.push_back(std::move(row));
        }
        {
            ps2x::SettingRow row;
            row.page = page;
            row.key = "display.aspect";
            row.environment = "PS2X_ASPECT";
            row.label = "Aspect Ratio";
            row.description = "The shape of the picture.";
            row.choices = {{"4:3", "4:3"}, {"16:9", "16:9"}, {"16:10", "16:10"}, {"21:9", "21:9"}};
            row.apply = [](const std::string &)
            {
                float across = 0.0f, up = 0.0f;
                if (ps2x::settingPair("PS2X_ASPECT", "display.aspect", across, up))
                    ps2x::setPresentationAspect(across / up);
            };
            rows.push_back(std::move(row));
        }
        {
            ps2x::SettingRow row;
            row.page = page;
            row.key = "display.resolution";
            row.environment = "PS2X_GS_SCALE";
            row.label = "Resolution";
            row.description = "How many times the game's own resolution the picture is drawn at.";
            row.choices = {{"1", "1x (PlayStation 2)"}, {"2", "2x"}, {"3", "3x"}, {"4", "4x"}};
            rows.push_back(std::move(row));
        }
        {
            ps2x::SettingRow row;
            row.page = page;
            row.key = "display.fields";
            row.environment = "PS2X_GS_FIELDS";
            row.label = "Interlacing";
            row.description = "Whole frames are steady and twice as detailed. Fields are the half pictures a "
                              "television was sent, their lines doubled.";
            row.choices = {{"whole", "Whole Frames"}, {"double", "Fields"}};
            rows.push_back(std::move(row));
        }
        {
            ps2x::SettingRow row;
            row.page = page;
            row.key = "display.filter";
            row.environment = "PS2X_FILTER";
            row.label = "Scaling";
            row.description = "How the picture is scaled up to the window: smooth, or with its pixels sharp.";
            row.choices = {{"linear", "Smooth"}, {"nearest", "Sharp"}};
            row.apply = [](const std::string &value) { ps2x::setDisplayFilter(value != "nearest"); };
            rows.push_back(std::move(row));
        }
        {
            ps2x::SettingRow row;
            row.page = "Sound";
            row.key = "audio.volume";
            row.environment = "PS2X_VOLUME";
            row.label = "Volume";
            row.description = "How loud the game is.";
            row.choices = {{"0", "Off"},  {"10", "10%"}, {"20", "20%"}, {"30", "30%"}, {"40", "40%"},  {"50", "50%"},
                           {"60", "60%"}, {"70", "70%"}, {"80", "80%"}, {"90", "90%"}, {"100", "100%"}};
            row.otherwise = 10;
            row.apply = [](const std::string &value)
            { SetMasterVolume(std::clamp(static_cast<float>(std::atof(value.c_str())), 0.0f, 100.0f) / 100.0f); };
            rows.push_back(std::move(row));
        }
        {
            ps2x::SettingRow row;
            row.page = "Controls";
            row.key = "input.mouse_look";
            row.label = "Mouse Camera";
            row.description = "The mouse moves the camera (the right stick) while the keyboard is in use.";
            row.choices = {{"on", "On"}, {"off", "Off"}};
            row.apply = [](const std::string &value) { ps2xInputSetMouseLook(value == "on"); };
            rows.push_back(std::move(row));
        }
        {
            ps2x::SettingRow row;
            row.page = "Controls";
            row.key = "input.mouse_sensitivity";
            row.label = "Mouse Sensitivity";
            row.description = "How far a movement of the mouse turns the camera.";
            row.choices = {{"0.5", "50%"},   {"0.75", "75%"}, {"1.0", "100%"}, {"1.25", "125%"},
                           {"1.5", "150%"}, {"2.0", "200%"}, {"3.0", "300%"}};
            row.otherwise = 2;
            row.apply = [](const std::string &value)
            { ps2xInputSetMouseSensitivity(static_cast<float>(std::atof(value.c_str()))); };
            rows.push_back(std::move(row));
        }
        // In front of what a game has added already, and not in the place
        // of a line of the game's with the same key.
        std::vector<Row> merged;
        for (ps2x::SettingRow &row : rows)
        {
            const auto same = std::find_if(all.rows.begin(), all.rows.end(),
                                           [&row](const Row &other) { return other.row.key == row.key; });
            if (same == all.rows.end())
                merged.push_back(Row{std::move(row)});
            else
            {
                merged.push_back(std::move(*same));
                all.rows.erase(same);
            }
        }
        for (Row &row : all.rows)
            merged.push_back(std::move(row));
        all.rows = std::move(merged);

        // A line that asks for a place under another gets it.
        for (size_t n = 0; n < all.rows.size(); ++n)
        {
            const std::string after = all.rows[n].row.after;
            if (after.empty())
                continue;
            const auto under = std::find_if(all.rows.begin(), all.rows.end(),
                                            [&after](const Row &other) { return other.row.key == after; });
            if (under == all.rows.end())
                continue;
            const size_t to = static_cast<size_t>(under - all.rows.begin());
            if (to + 1u == n)
                continue;
            Row moved = std::move(all.rows[n]);
            moved.row.after.clear();
            all.rows.erase(all.rows.begin() + static_cast<std::ptrdiff_t>(n));
            all.rows.insert(all.rows.begin() + static_cast<std::ptrdiff_t>(to < n ? to + 1u : to), std::move(moved));
            n = static_cast<size_t>(-1); // from the top again: the lines have moved
        }
    }

    // ---- moving about ------------------------------------------------------

    std::vector<std::string> pagesOf(const Menu &all)
    {
        std::vector<std::string> pages;
        for (const Row &row : all.rows)
            if (std::find(pages.begin(), pages.end(), row.row.page) == pages.end())
                pages.push_back(row.row.page);
        return pages;
    }

    std::vector<Row *> rowsOf(Menu &all, const std::string &page)
    {
        std::vector<Row *> rows;
        for (Row &row : all.rows)
            if (row.row.page == page)
                rows.push_back(&row);
        return rows;
    }

    void choose(Menu &all, Row &row, int choice)
    {
        if (choice == row.shown || choice < 0 || choice >= static_cast<int>(row.row.choices.size()))
            return;
        row.shown = choice;
        const std::string &value = row.row.choices[static_cast<size_t>(choice)].value;
        ps2x::setSetting(row.row.key.c_str(), value.c_str());
        all.unsaved = !ps2x::saveSettings(); // at once: the window may be closed with the menu open
        if (row.row.apply)
        {
            row.row.apply(value);
            row.running = choice;
            row.other.clear();
            g_frames.store(kFramesToShow, std::memory_order_relaxed);
        }
    }

    void open(Menu &all)
    {
        addRuntimeRows(all);
        if (!all.valued)
        {
            all.valued = true;
            for (Row &row : all.rows)
                giveValue(row, ps2x::setting(row.row.environment.empty() ? nullptr : row.row.environment.c_str(),
                                             row.row.key.c_str()));
        }
        for (Row &row : all.rows)
            if (row.row.current && row.shown == row.running)
                giveValue(row, row.row.current().c_str());
        all.held = 0xFFFFu; // what is down now does not count until it has been let go
        all.repeating = 0u;
        all.noticeUntil = 0.0;
        ps2xInputBlock(true);
        g_frames.store(0, std::memory_order_relaxed);
        g_open.store(true, std::memory_order_release);
    }

    void close(Menu &all, double now)
    {
        g_open.store(false, std::memory_order_release);
        ps2xInputBlock(false);
        if (all.unsaved)
        {
            if (ps2x::saveSettings())
                all.unsaved = false;
            else
            {
                all.notice = std::string(ps2x::settingsPath()) + " could not be written: the changes last until the game is closed";
                all.noticeUntil = now + 8.0;
            }
        }
    }

    std::string keyName(uint16_t button)
    {
        const std::string label = ps2xInputKeyLabel(button);
        static const struct
        {
            const char *label, *name;
        } names[] = {{"UP", "Up"},       {"DN", "Down"},   {"LT", "Left"},      {"RT", "Right"},      {"SPC", "Space"},
                     {"ENT", "Enter"},   {"ESC", "Esc"},   {"TAB", "Tab"},      {"BKS", "Backspace"}, {"SHF", "Shift"},
                     {"CTL", "Ctrl"},    {"ALT", "Alt"},   {"LMB", "Left Click"}, {"RMB", "Right Click"},
                     {"MMB", "Middle Click"}};
        for (const auto &name : names)
            if (label == name.label)
                return name.name;
        return label.empty() ? "-" : label;
    }

    std::string keysFor()
    {
        const int device = ps2xInputDevice();
        if (device < 0)
            return keyName(kUp) + " / " + keyName(kDown) + "  Select      " + keyName(kLeft) + " / " + keyName(kRight) +
                   "  Change      " + keyName(kL1) + " / " + keyName(kR1) + "  Page      " + keyName(kCircle) +
                   " or F2  Close";
        const std::string name = lowered(ps2xInputDeviceName());
        const bool xbox = name.find("xbox") != std::string::npos || name.find("xinput") != std::string::npos;
        return std::string("D-Pad  Select and Change      ") + (xbox ? "LB / RB" : "L1 / R1") + "  Page      " +
               (xbox ? "B" : "Circle") + "  Close";
    }

    void describe(Menu &all, double now)
    {
        ps2x::SettingsMenuState &state = all.state;
        state.open = g_open.load(std::memory_order_relaxed);
        state.lines.clear();
        state.description.clear();
        state.notice.clear();
        state.noticeStrength = 0.0f;
        if (!state.open)
        {
            // A line in the corner: how to get here, for the game's first
            // seconds; or what went wrong at closing.
            if (now < all.noticeUntil)
            {
                state.notice = all.notice;
                state.noticeStrength = static_cast<float>(std::min(1.0, all.noticeUntil - now));
            }
            else if (all.begun >= 0.0 && now - all.begun < kStartNotice && !scripted())
            {
                state.notice = "F2 or L3 + R3: Settings";
                state.noticeStrength = static_cast<float>(std::min({1.0, (now - all.begun) * 2.0, kStartNotice - (now - all.begun)}));
            }
            return;
        }
        state.pages = pagesOf(all);
        all.page = std::min(all.page, state.pages.empty() ? 0u : state.pages.size() - 1u);
        state.page = all.page;
        const std::vector<Row *> rows = state.pages.empty() ? std::vector<Row *>() : rowsOf(all, state.pages[all.page]);
        all.line = std::min(all.line, rows.size()); // one more: the line that restores
        state.line = all.line;
        for (const Row *row : rows)
        {
            ps2x::SettingsMenuLine line;
            line.label = row->row.label;
            line.value = row->shown >= 0 ? row->row.choices[static_cast<size_t>(row->shown)].text
                                         : "Custom (" + row->other + ")";
            line.before = row->shown != 0;
            line.after = row->shown < 0 || row->shown + 1 < static_cast<int>(row->row.choices.size());
            line.waits = !row->row.apply && row->shown != row->running;
            state.lines.push_back(std::move(line));
        }
        ps2x::SettingsMenuLine restore;
        restore.label = "Restore Defaults";
        state.lines.push_back(std::move(restore));
        if (all.line < rows.size())
        {
            const Row &row = *rows[all.line];
            state.description = row.row.description;
            if (!row.row.apply)
                state.description += state.description.empty() ? "Takes effect the next time the game starts."
                                                               : " Takes effect the next time the game starts.";
        }
        else
            state.description = "Puts every setting on this page back to what it was at first.";
        state.keys = keysFor();
    }
}

void ps2x::addSettingRow(SettingRow row)
{
    if (row.choices.empty() || row.key.empty())
        return;
    Menu &all = menu();
    const auto same = std::find_if(all.rows.begin(), all.rows.end(),
                                   [&row](const Row &other) { return other.row.key == row.key; });
    if (same != all.rows.end())
        same->row = std::move(row);
    else
        all.rows.push_back(Row{std::move(row)});
}

bool ps2x::settingsMenuOpen()
{
    return g_open.load(std::memory_order_acquire);
}

const ps2x::SettingsMenuState &ps2x::settingsMenuState()
{
    menu().drawn = true;
    return menu().state;
}

void ps2x::settingsMenuFrame()
{
    Menu &all = menu();
    if (!all.drawn || scripted())
        return;
    const double now = GetTime();
    if (all.begun < 0.0)
        all.begun = now;

    // F2, or both sticks pressed in on any controller for a moment.
    bool toggle = IsKeyPressed(KEY_F2);
    bool chord = false;
    for (int gamepad = 0; gamepad < 4; ++gamepad)
        chord = chord || (IsGamepadAvailable(gamepad) && IsGamepadButtonDown(gamepad, GAMEPAD_BUTTON_LEFT_THUMB) &&
                          IsGamepadButtonDown(gamepad, GAMEPAD_BUTTON_RIGHT_THUMB));
    if (!chord)
    {
        all.chordSince = -1.0;
        all.chordSpent = false;
    }
    else if (all.chordSince < 0.0)
        all.chordSince = now;
    else if (!all.chordSpent && now - all.chordSince >= kChordSeconds)
    {
        all.chordSpent = true;
        toggle = true;
    }

    if (!g_open.load(std::memory_order_relaxed))
    {
        if (toggle)
            open(all);
        describe(all, now);
        return;
    }

    // The pad is read here only while the menu is open: the game is not
    // reading it then.
    const uint16_t down = ps2xInputButtons();
    uint16_t pressed = static_cast<uint16_t>(down & ~all.held);
    all.held = down;
    // A direction that stays down repeats.
    const uint16_t directions = static_cast<uint16_t>(down & (kUp | kDown | kLeft | kRight));
    if ((pressed & (kUp | kDown | kLeft | kRight)) != 0u)
    {
        all.repeating = static_cast<uint16_t>(pressed & (kUp | kDown | kLeft | kRight));
        all.repeatAt = now + kRepeatAfter;
    }
    else if (all.repeating != 0u && (directions & all.repeating) == all.repeating)
    {
        if (now >= all.repeatAt)
        {
            pressed |= all.repeating;
            all.repeatAt = now + kRepeatEvery;
        }
    }
    else
        all.repeating = 0u;

    if (toggle || (pressed & (kCircle | kStart)) != 0u)
    {
        close(all, now);
        describe(all, now);
        return;
    }

    const std::vector<std::string> pages = pagesOf(all);
    if (!pages.empty())
    {
        if ((pressed & kR1) != 0u)
            all.page = (all.page + 1u) % pages.size(), all.line = 0;
        if ((pressed & kL1) != 0u)
            all.page = (all.page + pages.size() - 1u) % pages.size(), all.line = 0;
        const std::vector<Row *> rows = rowsOf(all, pages[std::min(all.page, pages.size() - 1u)]);
        const size_t lines = rows.size() + 1u;
        if ((pressed & kDown) != 0u)
            all.line = (all.line + 1u) % lines;
        if ((pressed & kUp) != 0u)
            all.line = (all.line + lines - 1u) % lines;
        if (all.line < rows.size())
        {
            Row &row = *rows[all.line];
            const int count = static_cast<int>(row.row.choices.size());
            if ((pressed & kRight) != 0u)
                choose(all, row, row.shown < 0 ? 0 : row.shown + 1);
            if ((pressed & kLeft) != 0u)
                choose(all, row, row.shown < 0 ? count - 1 : row.shown - 1);
            if ((pressed & kCross) != 0u)
                choose(all, row, row.shown < 0 ? 0 : (row.shown + 1) % count);
        }
        else if ((pressed & kCross) != 0u)
            for (Row *row : rows)
                choose(all, *row, static_cast<int>(std::min(row->row.otherwise, row->row.choices.size() - 1u)));
    }
    describe(all, now);
}

// Asked by the EE at a vblank, again and again while it answers yes: the
// game stands still while the menu is open, but for the few frames a change
// is given to show in.
bool ps2xSettingsMenuHolds()
{
    if (!g_open.load(std::memory_order_acquire))
        return false;
    int frames = g_frames.load(std::memory_order_relaxed);
    while (frames > 0)
        if (g_frames.compare_exchange_weak(frames, frames - 1, std::memory_order_relaxed))
            return false;
    return true;
}
