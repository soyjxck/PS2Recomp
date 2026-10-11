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
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <atomic>

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

        // What ps2x::setSetting (ps2_settings_store.h) has put in the
        // place of a key's value for this run. The store gives its
        // lookup here the first time something is set; before that,
        // and in a program without the store, there is none.
        using ChangedLookup = const char *(*)(const char *key);
        inline std::atomic<ChangedLookup> &changedLookup()
        {
            static std::atomic<ChangedLookup> lookup{nullptr};
            return lookup;
        }
    }

    // What has been set for the key while the game runs, else the
    // environment variable if it is set, else the file's "section.key",
    // else nullptr. Either name may be nullptr.
    inline const char *setting(const char *environment, const char *key)
    {
        if (key)
            if (const auto changed = settings_detail::changedLookup().load(std::memory_order_acquire))
                if (const char *value = changed(key))
                    return *value ? value : nullptr;
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

namespace ps2x
{
    // The shape the picture is shown in from now on, width over height, in
    // place of the [display] aspect; 0 goes back to that. For a game that
    // draws some of its screens for a wider picture than others (its scenes
    // widened, its movies and menus not). Any thread may call it.
    void setPresentationAspect(float widthOverHeight);

    // How the picture is drawn to the window from now on, in place of the
    // [display] filter: smooth (linear) or each pixel as it is (nearest).
    // Any thread may call it.
    void setDisplayFilter(bool smooth);

    // What a game draws flat over its scenes (gauges, maps, text) is laid
    // out for a 4:3 picture. Once the scenes are made for a wider one and
    // shown stretched out to it, that layer is stretched with them. This
    // has the GS draw the layer narrower by `narrow` (the game's aspect
    // over the picture's, 0.75 for 16:9; 1 or 0 for as the game sends it),
    // so that it comes out the shape it was drawn. The layer is what goes
    // into a frame `width` pixels wide at `depth` exactly, the nearest
    // there is, or at `ceiling` and past it; not what spans the whole
    // width (a fade, a band behind text) and not what is drawn from a
    // texture of whole colours (a picture of the scene, worked over).
    // - Pieces that touch make a cluster, which is narrowed about where
    //   its first piece is, and so stays where the game put it (a gauge
    //   in its place, a cursor over what it points at). A cluster that
    //   starts inside one of `sides` is narrowed about that side of the
    //   frame instead and keeps to its corner; a piece of it that reaches
    //   the far side still does.
    // - Text, drawn from the texture at `text`, goes with the cluster it
    //   starts in, and about the middle of the frame if it starts in none.
    // - Sprites tested against the scene's depth, and what lies between
    //   `depth` and `ceiling`, are marks and labels set over things in the
    //   scene, each a run of primitives at one depth: narrowed about the
    //   first of the run.
    struct FlatLayer
    {
        float narrow = 1.0f;
        float width = 640.0f;
        float depth = 65535.0f;
        float ceiling = 16777215.0f;
        uint32_t text = 0; // TBP0; 0 for no such texture
        struct Side
        {
            float x0 = 0.0f, y0 = 0.0f, x1 = 0.0f, y1 = 0.0f;
            int towards = 0; // -1 the left of the frame, 1 the right, 0 unused
        } sides[4]{};
    };
    // Any thread may call it.
    void setFlatLayer(const FlatLayer &layer);

    // How long a call that the IOP answers takes, in emulated microseconds:
    // the caller of a blocking one (sceSifCallRpc without SIF_RPC_M_NOWAIT,
    // sceCdSearchFile, sceCdDiskReady(0)...) waits that long, and the end
    // function of a no-wait one runs that much later. 0, the default, is no
    // time at all: the IOP's side of a call is done when the EE's function
    // returns. On a console the answer crosses the SIF some hundreds of
    // microseconds later and the caller's thread is off the processor
    // meanwhile; a game whose threads count on that says how long in its
    // override. PS2X_SIF_RPC_DELAY_US wins over this.
    void setSifRpcDelay(uint32_t microseconds);

    // How many vblanks the GS, and VU1 before it, may be behind the EE
    // (GsWorker::lagVBlanks in runtime/gs/gs_worker.h says what that means).
    // 0, the default: never behind. A game whose frames take the host longer
    // to draw than to prepare says how far in its override: 2 is one frame of
    // a game that draws every other vblank. PS2X_GS_LAG wins over this.
    void setGsLag(uint32_t vblanks);
}

#endif // PS2_SETTINGS_H
