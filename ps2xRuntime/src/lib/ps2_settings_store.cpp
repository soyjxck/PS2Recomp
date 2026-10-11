#include "runtime/ps2_settings_store.h"
#include "runtime/ps2_settings.h"

#include <cstdio>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <map>
#include <mutex>
#include <string>
#include <system_error>
#include <vector>

namespace
{
    struct Store
    {
        std::mutex mutex;
        // A value handed out by ps2x::setting() may be in use on another
        // thread for as long as the game runs, so none is ever freed.
        std::deque<std::string> kept;
        std::map<std::string, const char *> changed;
        std::vector<std::string> unsaved; // keys, in the order they were first set
    };

    Store &store()
    {
        static Store instance;
        return instance;
    }

    using ps2x::settings_detail::lowered;
    using ps2x::settings_detail::trimmed;

    // A line of the file taken apart: "[section]", "key = value ; comment",
    // or anything else.
    struct Line
    {
        std::string section; // of a "[section]" line, lowered
        std::string key;     // of a "key = value" line, lowered
        bool isSection = false;
    };

    Line parsed(const std::string &text)
    {
        Line line;
        std::string body = text;
        const size_t comment = body.find_first_of(";#");
        if (comment != std::string::npos)
            body.erase(comment);
        body = trimmed(body);
        if (body.size() >= 2 && body.front() == '[' && body.back() == ']')
        {
            line.isSection = true;
            line.section = lowered(trimmed(body.substr(1, body.size() - 2)));
            return line;
        }
        const size_t equals = body.find('=');
        if (equals != std::string::npos)
            line.key = lowered(trimmed(body.substr(0, equals)));
        return line;
    }

    // "key = value" in the place of a line that set the key: the key as the
    // file spells it, and a comment after the value kept.
    std::string rewritten(const std::string &old, const std::string &value)
    {
        const size_t equals = old.find('=');
        std::string line = old.substr(0, equals);
        while (!line.empty() && (line.back() == ' ' || line.back() == '\t'))
            line.pop_back();
        line += " = " + value;
        const size_t comment = old.find_first_of(";#", equals);
        if (comment != std::string::npos)
        {
            size_t from = comment;
            while (from > equals + 1 && (old[from - 1] == ' ' || old[from - 1] == '\t'))
                --from;
            line += old.substr(from);
        }
        return line;
    }
}

std::string ps2x::settingsFileWith(const std::string &text,
                                   const std::vector<std::pair<std::string, std::string>> &values)
{
    // The file line by line, with the line ends it has.
    std::vector<std::string> lines;
    bool crlf = false;
    for (size_t at = 0; at < text.size();)
    {
        size_t end = text.find('\n', at);
        if (end == std::string::npos)
            end = text.size();
        std::string line = text.substr(at, end - at);
        if (!line.empty() && line.back() == '\r')
        {
            crlf = true;
            line.pop_back();
        }
        lines.push_back(std::move(line));
        at = end + 1;
    }

    for (const auto &[name, value] : values)
    {
        const size_t dot = name.find('.');
        const std::string section = dot == std::string::npos ? std::string() : name.substr(0, dot);
        const std::string key = dot == std::string::npos ? name : name.substr(dot + 1);

        // The section's lines: from after its "[section]" (the file's start
        // for a key with no section) to the next section.
        size_t first = 0, last = lines.size();
        bool haveSection = section.empty();
        for (size_t n = 0; n < lines.size(); ++n)
        {
            const Line line = parsed(lines[n]);
            if (!line.isSection)
                continue;
            if (haveSection)
            {
                last = n;
                break;
            }
            if (line.section == section)
            {
                haveSection = true;
                first = n + 1;
            }
        }
        if (!haveSection)
        {
            if (!lines.empty() && !trimmed(lines.back()).empty())
                lines.emplace_back();
            lines.push_back("[" + section + "]");
            lines.push_back(key + " = " + value);
            continue;
        }
        bool written = false;
        for (size_t n = first; n < last && !written; ++n)
            if (parsed(lines[n]).key == key)
            {
                lines[n] = rewritten(lines[n], value);
                written = true;
            }
        if (!written)
        {
            // After the section's last line that says anything.
            size_t at = last;
            while (at > first && trimmed(lines[at - 1]).empty())
                --at;
            lines.insert(lines.begin() + static_cast<std::ptrdiff_t>(at), key + " = " + value);
        }
    }

    std::string out;
    for (const std::string &line : lines)
        out += line + (crlf ? "\r\n" : "\n");
    return out;
}

namespace
{
    const char *changed(const char *key)
    {
        Store &all = store();
        std::lock_guard lock(all.mutex);
        const auto found = all.changed.find(key);
        return found == all.changed.end() ? nullptr : found->second;
    }
}

void ps2x::setSetting(const char *key, const char *value)
{
    if (!key || !*key)
        return;
    Store &all = store();
    std::lock_guard lock(all.mutex);
    all.kept.emplace_back(value ? value : "");
    all.changed[key] = all.kept.back().c_str();
    bool listed = false;
    for (const std::string &name : all.unsaved)
        listed = listed || name == key;
    if (!listed)
        all.unsaved.emplace_back(key);
    settings_detail::changedLookup().store(&changed, std::memory_order_release);
}

const char *ps2x::settingsPath()
{
    static const std::string path = []
    {
        const char *named = std::getenv("PS2X_SETTINGS");
        return std::string(named && *named ? named : "settings.ini");
    }();
    return path.c_str();
}

bool ps2x::saveSettings()
{
    Store &all = store();
    std::vector<std::pair<std::string, std::string>> writes;
    {
        std::lock_guard lock(all.mutex);
        for (const std::string &key : all.unsaved)
            writes.emplace_back(key, all.changed[key]);
    }
    if (writes.empty())
        return true;

    std::string text;
    if (FILE *in = std::fopen(settingsPath(), "rb"))
    {
        char buffer[4096];
        size_t got;
        while ((got = std::fread(buffer, 1, sizeof(buffer), in)) > 0)
            text.append(buffer, got);
        std::fclose(in);
    }
    text = settingsFileWith(text, writes);

    // Written beside the file and put in its place, so that a failure
    // half-way leaves the old one.
    const std::string path = settingsPath();
    const std::string fresh = path + ".new";
    FILE *out = std::fopen(fresh.c_str(), "wb");
    if (!out)
        return false;
    bool good = std::fwrite(text.data(), 1, text.size(), out) == text.size();
    good = std::fclose(out) == 0 && good;
    std::error_code failed;
    if (good)
        std::filesystem::rename(fresh, path, failed);
    if (!good || failed)
    {
        std::filesystem::remove(fresh, failed);
        return false;
    }

    // What was set again meanwhile is still to be written.
    std::lock_guard lock(all.mutex);
    for (const auto &[name, value] : writes)
        if (value == all.changed[name])
            std::erase(all.unsaved, name);
    return true;
}
