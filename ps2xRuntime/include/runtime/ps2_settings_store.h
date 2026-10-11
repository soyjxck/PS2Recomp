#ifndef PS2_SETTINGS_STORE_H
#define PS2_SETTINGS_STORE_H

// Settings changed while the game runs, and written back to the file.
//
// ps2x::setting() (ps2_settings.h) reads the settings file once. What is set
// here stands in the place of the file's value, and of the environment's,
// for the rest of the run, and saveSettings() writes it into the file: on
// the key's own line where the file has one, at the end of its section
// where it has not, with every comment and everything else left as it was.
//
// Code that asked for a setting once and kept the answer does not hear of a
// change; what a change does while the game runs is up to whoever makes it
// (the settings menu calls a row's `apply`).

#include <string>
#include <utility>
#include <vector>

namespace ps2x
{
    // "section.key" in lower case, as for ps2x::setting(). Any thread.
    void setSetting(const char *key, const char *value);

    // Writes what has been set since the last time. False when the file
    // could not be written (true when there was nothing to write).
    bool saveSettings();

    // The settings file: PS2X_SETTINGS, else settings.ini where the game was
    // started.
    const char *settingsPath();

    // The text of a settings file with these values ("section.key", value)
    // put into it: what saveSettings() makes of the file.
    std::string settingsFileWith(const std::string &text,
                                 const std::vector<std::pair<std::string, std::string>> &values);
}

#endif // PS2_SETTINGS_STORE_H
