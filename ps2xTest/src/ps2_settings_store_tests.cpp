#include "MiniTest.h"
#include "runtime/ps2_display_mode.h"
#include "runtime/ps2_settings.h"
#include "runtime/ps2_settings_store.h"

#include <string>
#include <utility>
#include <vector>

// The settings file rewritten with changed values (ps2_settings_store.cpp),
// and a value set while the game runs standing in the file's place.

namespace
{
    using Values = std::vector<std::pair<std::string, std::string>>;
}

void register_ps2_settings_store_tests()
{
    MiniTest::Case("SettingsStore", [](TestCase &tc)
                   {
        tc.Run("a key's line is rewritten where it stands", [](TestCase &t)
               {
            const std::string file = "; top of the file\n"
                                     "[display]\n"
                                     "; the picture's shape   [PS2X_ASPECT]\n"
                                     "aspect = 16:9\n"
                                     "window = 1280x720   ; its size\n"
                                     "\n"
                                     "[input]\n"
                                     "cross = space, x\n";
            const std::string want = "; top of the file\n"
                                     "[display]\n"
                                     "; the picture's shape   [PS2X_ASPECT]\n"
                                     "aspect = 4:3\n"
                                     "window = 1920x1080   ; its size\n"
                                     "\n"
                                     "[input]\n"
                                     "cross = space, x\n";
            t.Equals(ps2x::settingsFileWith(file, Values{{"display.aspect", "4:3"}, {"display.window", "1920x1080"}}),
                     want, "values change on their own lines; comments and the rest stay");
        });

        tc.Run("a key the file lacks goes to the end of its section", [](TestCase &t)
               {
            const std::string file = "[display]\n"
                                     "aspect = 16:9\n"
                                     "\n"
                                     "[input]\n"
                                     "cross = x\n";
            const std::string want = "[display]\n"
                                     "aspect = 16:9\n"
                                     "hud = fit\n"
                                     "\n"
                                     "[input]\n"
                                     "cross = x\n"
                                     "\n"
                                     "[game]\n"
                                     "soldiers = 80,30\n"
                                     "enemy_distance = 2\n";
            t.Equals(ps2x::settingsFileWith(file, Values{{"display.hud", "fit"},
                                                         {"game.soldiers", "80,30"},
                                                         {"game.enemy_distance", "2"}}),
                     want, "a new key under its section's last line, a new section at the file's end");
        });

        tc.Run("the file keeps its line ends and its spelling", [](TestCase &t)
               {
            t.Equals(ps2x::settingsFileWith("[Display]\r\n  Aspect=16:9\r\n", Values{{"display.aspect", "21:9"}}),
                     std::string("[Display]\r\n  Aspect = 21:9\r\n"), "CRLF, the section's and the key's own spelling");
            t.Equals(ps2x::settingsFileWith("[display]\n; aspect = 4:3\naspect = 16:9", Values{{"display.aspect", "16:10"}}),
                     std::string("[display]\n; aspect = 4:3\naspect = 16:10\n"), "a commented line is not the key's");
            t.Equals(ps2x::settingsFileWith("", Values{{"display.filter", "nearest"}}),
                     std::string("[display]\nfilter = nearest\n"), "an empty file gets the section");
            t.Equals(ps2x::settingsFileWith("[a]\nx = 1\n[b]\nx = 2\n", Values{{"b.x", "3"}}),
                     std::string("[a]\nx = 1\n[b]\nx = 3\n"), "the key of the right section");
        });

        tc.Run("a value set while running stands in the file's place", [](TestCase &t)
               {
            t.IsNull(ps2x::setting(nullptr, "store_test.value"), "nothing before it is set");
            ps2x::setSetting("store_test.value", "first");
            t.Equals(std::string(ps2x::setting(nullptr, "store_test.value")), std::string("first"), "the value set");
            ps2x::setSetting("store_test.value", "second");
            t.Equals(std::string(ps2x::setting("PS2X_TEST_NO_SUCH_VARIABLE", "store_test.value")), std::string("second"),
                     "the value set again");
            ps2x::setSetting("store_test.value", "");
            t.IsNull(ps2x::setting(nullptr, "store_test.value"), "set to nothing, there is none");
        });

        tc.Run("[display] fullscreen names one of three modes", [](TestCase &t)
               {
            using ps2x::DisplayMode;
            const auto from = [](const char *text)
            { return ps2x::displayModeFrom(text, DisplayMode::Window); };
            t.IsTrue(from("off") == DisplayMode::Window, "off is a window");
            t.IsTrue(from("borderless") == DisplayMode::Borderless, "borderless");
            t.IsTrue(from("exclusive") == DisplayMode::Exclusive, "exclusive");
            t.IsTrue(from(" Exclusive ") == DisplayMode::Exclusive, "whatever its case and the blanks about it");
            // A file written when there were two keeps doing what it did.
            t.IsTrue(from("on") == DisplayMode::Borderless, "on was the whole screen without a frame, and is");
            t.IsTrue(from("1") == DisplayMode::Borderless, "and so was 1");
            t.IsTrue(from("0") == DisplayMode::Window, "0 is a window");
            t.IsTrue(ps2x::displayModeFrom(nullptr, DisplayMode::Exclusive) == DisplayMode::Exclusive, "nothing set: what the caller says");
            t.IsTrue(ps2x::displayModeFrom("sideways", DisplayMode::Borderless) == DisplayMode::Borderless,
                     "a word that names none: what the caller says");
            // What the menu writes is what is read back.
            for (const DisplayMode mode : {DisplayMode::Window, DisplayMode::Borderless, DisplayMode::Exclusive})
                t.IsTrue(from(ps2x::displayModeValue(mode)) == mode, "a mode's own word names it");
        }); });
}
