#ifndef PS2_SETTINGS_H
#define PS2_SETTINGS_H

// What the player has set, from a settings file and from the environment.
//
// The file is an INI: "[section]" lines, "key = value" lines, and comments
// from ';' or '#' to the end of a line. It is PS2X_SETTINGS if that is set,
// else settings.ini in the directory the game is started from, and it is
// read once, the first time something is asked for. A setting is asked for
// by "section.key", in lower case.
//
// Everything in the file can also be given as an environment variable,
// which wins: test runs are driven that way.

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>

namespace ps2x
{
    namespace settings_detail
    {
        inline std::string trimmed(const std::string &text)
        {
            size_t first = 0, last = text.size();
            while (first < last && std::isspace(static_cast<unsigned char>(text[first])))
                ++first;
            while (last > first && std::isspace(static_cast<unsigned char>(text[last - 1])))
                --last;
            return text.substr(first, last - first);
        }

        inline std::string lowered(std::string text)
        {
            for (char &c : text)
                c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            return text;
        }

        inline const std::map<std::string, std::string> &file()
        {
            static const std::map<std::string, std::string> values = []
            {
                std::map<std::string, std::string> read;
                const char *named = std::getenv("PS2X_SETTINGS");
                FILE *in = std::fopen(named && *named ? named : "settings.ini", "r");
                if (!in)
                    return read;
                std::string section;
                char raw[1024];
                while (std::fgets(raw, sizeof(raw), in))
                {
                    std::string line(raw);
                    const size_t comment = line.find_first_of(";#");
                    if (comment != std::string::npos)
                        line.erase(comment);
                    line = trimmed(line);
                    if (line.size() >= 2 && line.front() == '[' && line.back() == ']')
                    {
                        section = lowered(trimmed(line.substr(1, line.size() - 2)));
                        continue;
                    }
                    const size_t equals = line.find('=');
                    if (equals == std::string::npos)
                        continue;
                    const std::string key = lowered(trimmed(line.substr(0, equals)));
                    if (!key.empty())
                        read[section.empty() ? key : section + "." + key] = trimmed(line.substr(equals + 1));
                }
                std::fclose(in);
                return read;
            }();
            return values;
        }
    }

    // The environment variable if it is set, else the file's "section.key",
    // else nullptr. Either name may be nullptr.
    inline const char *setting(const char *environment, const char *key)
    {
        if (environment)
            if (const char *value = std::getenv(environment); value && *value)
                return value;
        if (key)
        {
            const auto &values = settings_detail::file();
            const auto found = values.find(key);
            if (found != values.end() && !found->second.empty())
                return found->second.c_str();
        }
        return nullptr;
    }

    // "on", "yes", "true" and numbers other than 0 are true; anything else
    // that is there is false; `otherwise` when it is not there at all.
    inline bool settingOn(const char *environment, const char *key, bool otherwise)
    {
        const char *value = setting(environment, key);
        if (!value)
            return otherwise;
        const std::string text = settings_detail::lowered(value);
        return text == "on" || text == "yes" || text == "true" || std::atof(text.c_str()) != 0.0;
    }

    // "16:9" or "1280x720" or "1.78": the two numbers of a setting written
    // as a pair, the second 1 when only one is given. False when the
    // setting is not there or is not numbers.
    inline bool settingPair(const char *environment, const char *key, float &first, float &second)
    {
        const char *value = setting(environment, key);
        if (!value)
            return false;
        float a = 0.0f, b = 1.0f;
        char between = 0;
        const int fields = std::sscanf(value, "%f %c %f", &a, &between, &b);
        if (fields < 1 || a <= 0.0f || (fields == 3 && b <= 0.0f))
            return false;
        first = a;
        second = fields == 3 ? b : 1.0f;
        return true;
    }
}

#endif // PS2_SETTINGS_H
