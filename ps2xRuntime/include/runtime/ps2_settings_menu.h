#ifndef PS2_SETTINGS_MENU_H
#define PS2_SETTINGS_MENU_H

// The settings menu: pages of settings over the game's picture, worked with
// the pad or the keyboard, for what the settings file holds.
//
// F2 opens and closes it, and so do L3 and R3 held together (the two sticks
// pressed in), which the game hears of as well but few games mind. While it
// is open the game stands still and its pad is at rest. Up and down pick a
// line, left and right change its value, L1 and R1 turn the page, circle or
// START closes. A change is put into the settings file when the menu
// closes, and into effect at once where the line knows how (`apply`); the
// others say that they wait for the next start.
//
// The runtime puts in the lines for the picture; a game adds its own, or
// its own in the place of one of those, with addSettingRow() before the
// window opens (a game override's apply function is a good place). The menu
// is drawn with the debug windows' library, so a build without
// PS2X_ENABLE_DEBUG_UI has none, and a run driven by PS2X_PAD_SCRIPT has
// none either.

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

namespace ps2x
{
    struct SettingChoice
    {
        std::string value; // as the settings file has it
        std::string text;  // as the menu shows it
    };

    // One line of the menu: a setting and the values to choose from.
    struct SettingRow
    {
        std::string page;        // the page it is on: "Graphics", "Game"...
        std::string key;         // "section.key" in the settings file
        std::string environment; // the variable that overrides the file, or ""
        std::string label;
        std::string description; // a line or two, shown under the lines
        std::string after;       // the key of the line this one goes under, or "" for the page's end
        std::vector<SettingChoice> choices;
        size_t otherwise = 0; // the choice in force when nothing is set
        // Puts a value into effect while the game runs; called on the
        // thread with the window, between two frames. Without one the
        // setting is read when the game starts, and the menu says that a
        // change waits for that.
        std::function<void(const std::string &value)> apply;
        // The value in force now, for a setting that something besides the
        // menu changes (a key that switches the full screen). Asked each
        // time the menu opens.
        std::function<std::string()> current;
    };

    // Lines appear in the order they are added, pages in the order their
    // first lines are; a line with the key of one already there takes its
    // place. On the thread with the window, before the menu is first opened.
    void addSettingRow(SettingRow row);

    // The menu as it is to be drawn.
    struct SettingsMenuLine
    {
        std::string label;
        std::string value;   // empty for a line that does something
        bool before = false; // there is a value to the left
        bool after = false;  // and to the right
        bool waits = false;  // changed, and in force from the next start
    };

    struct SettingsMenuState
    {
        bool open = false;
        std::vector<std::string> pages;
        size_t page = 0;
        std::vector<SettingsMenuLine> lines; // of the page that is open
        size_t line = 0;
        std::string description; // of the line picked
        std::string keys;        // which buttons do what, on the device in use
        std::string notice;      // a line for the corner of the picture while the menu is shut
        float noticeStrength = 0.0f; // 0..1
    };

    // Once for every pass of the host's loop, on the thread with the window
    // and between two frames: reads the pad and the keys, moves about,
    // changes and applies.
    void settingsMenuFrame();

    // What there is to draw. Until this has been asked for once the menu
    // does not open: nothing would show it.
    const SettingsMenuState &settingsMenuState();

    // Is it open? Any thread.
    bool settingsMenuOpen();
}

// The drawing, with the debug windows' library (ps2_settings_menu_ui.cpp):
// once after that library is set up, and inside each of its frames.
void ps2xSetupSettingsMenuUi();
void ps2xDrawSettingsMenu(const ps2x::SettingsMenuState &menu);
// PS2X_SETTINGS_MENU_GRAB=<prefix>: the window as a PNG each time what the
// menu shows has changed, for tests. After the library's frame has ended.
void ps2xGrabSettingsMenu(const ps2x::SettingsMenuState &menu);

#endif // PS2_SETTINGS_MENU_H
