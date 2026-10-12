#include "runtime/ps2_pad.h"
#include "runtime/ps2_input.h"
#include "runtime/ps2_settings.h"
#include "ps2_host_backend.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

// The host's side of the PS2 pad: any controller, or the keyboard and mouse.
//
// The device used last drives the pad. A button or a stick pushed on any of
// the host's controllers makes that controller the one that is read; a bound
// key or mouse button makes it the keyboard and mouse. The two are never
// merged, so a controller in use plays exactly as it always has and keys
// cannot interfere with it. Before anything is touched it is the first
// controller, or the keyboard when there is none.
//
// Keys are bound to PS2 buttons, not to what a game does with them, in
// settings.ini:
//
//   [input]
//   cross = space, x          ; one or more keys or mouse buttons
//   stick_up = w              ; the left stick, pushed all the way
//   mouse_look = on           ; the mouse is the right stick
//   mouse_sensitivity = 1.0
//
// Buttons: cross circle square triangle l1 r1 l2 r2 l3 r3 start select
// up down left right; the left stick stick_up stick_down stick_left
// stick_right; the right stick look_up look_down look_left look_right.
// Keys: letters, digits, f1..f12, kp0..kp9, space enter tab escape backspace
// up down left right, lshift rshift lctrl rctrl lalt ralt, mouse1 (left)
// mouse2 (right) mouse3 (middle) mouse4 mouse5; none unbinds.
//
// A run driven by PS2X_PAD_SCRIPT reads no real device at all: what is on the
// desk does not get into a test.

namespace
{
    constexpr uint8_t kPadAnalogMarker = 0x73;
    constexpr uint8_t kPadStickCenter = 0x80;
    constexpr int kGamepads = 4; // raylib's MAX_GAMEPADS

    constexpr uint16_t PAD_LEFT = 0x0080u;
    constexpr uint16_t PAD_DOWN = 0x0040u;
    constexpr uint16_t PAD_RIGHT = 0x0020u;
    constexpr uint16_t PAD_UP = 0x0010u;
    constexpr uint16_t PAD_START = 0x0008u;
    constexpr uint16_t PAD_R3 = 0x0004u;
    constexpr uint16_t PAD_L3 = 0x0002u;
    constexpr uint16_t PAD_SELECT = 0x0001u;
    constexpr uint16_t PAD_SQUARE = 0x8000u;
    constexpr uint16_t PAD_CROSS = 0x4000u;
    constexpr uint16_t PAD_CIRCLE = 0x2000u;
    constexpr uint16_t PAD_TRIANGLE = 0x1000u;
    constexpr uint16_t PAD_R1 = 0x0800u;
    constexpr uint16_t PAD_L1 = 0x0400u;
    constexpr uint16_t PAD_R2 = 0x0200u;
    constexpr uint16_t PAD_L2 = 0x0100u;

    // What the host loop and the pad reads, on different threads, tell each other.
    std::atomic<int> g_device{-2};      // a controller's index; -1 keyboard and mouse; -2 nothing used yet
    std::atomic<bool> g_uiOpen{false};  // a debug window wants the mouse
    std::atomic<int> g_blocked{0};      // 1 a menu of the host's has the pad; 2 it had, until all is let go
    std::atomic<float> g_lookX{0.0f};   // the mouse as the right stick, -1..1
    std::atomic<float> g_lookY{0.0f};
    // The mouse's two settings as changed while the game runs; below 0, as the file has them.
    std::atomic<int> g_mouseLook{-1};
    std::atomic<float> g_sensitivity{-1.0f};
    // A game that takes the mouse's movement itself (ps2xInputMouseLookDirect):
    // what has moved and not been taken yet, while the mouse is the camera's.
    std::atomic<bool> g_lookDirect{false};
    std::mutex g_takenMutex;
    bool g_takeable = false;
    float g_takeX = 0.0f, g_takeY = 0.0f;

    // A key or mouse button, as raylib numbers them; mouse buttons from 1000.
    constexpr int kMouseBase = 1000;

    int keyNamed(std::string name)
    {
        name = ps2x::settings_detail::lowered(ps2x::settings_detail::trimmed(name));
        if (name.size() == 1)
        {
            const char c = name[0];
            if (c >= 'a' && c <= 'z')
                return KEY_A + (c - 'a');
            if (c >= '0' && c <= '9')
                return KEY_ZERO + (c - '0');
        }
        if (name.size() >= 2 && name[0] == 'f' && std::isdigit(static_cast<unsigned char>(name[1])))
        {
            const int number = std::atoi(name.c_str() + 1);
            if (number >= 1 && number <= 12)
                return KEY_F1 + (number - 1);
        }
        if (name.size() == 3 && name.compare(0, 2, "kp") == 0 && std::isdigit(static_cast<unsigned char>(name[2])))
            return KEY_KP_0 + (name[2] - '0');
        if (name.size() == 6 && name.compare(0, 5, "mouse") == 0 && name[5] >= '1' && name[5] <= '5')
        {
            static constexpr int buttons[] = {MOUSE_BUTTON_LEFT, MOUSE_BUTTON_RIGHT, MOUSE_BUTTON_MIDDLE,
                                              MOUSE_BUTTON_SIDE, MOUSE_BUTTON_EXTRA};
            return kMouseBase + buttons[name[5] - '1'];
        }
        static const struct
        {
            const char *name;
            int key;
        } named[] = {
            {"space", KEY_SPACE},      {"enter", KEY_ENTER},        {"return", KEY_ENTER},
            {"tab", KEY_TAB},          {"escape", KEY_ESCAPE},      {"esc", KEY_ESCAPE},
            {"backspace", KEY_BACKSPACE}, {"up", KEY_UP},           {"down", KEY_DOWN},
            {"left", KEY_LEFT},        {"right", KEY_RIGHT},        {"lshift", KEY_LEFT_SHIFT},
            {"rshift", KEY_RIGHT_SHIFT}, {"lctrl", KEY_LEFT_CONTROL}, {"rctrl", KEY_RIGHT_CONTROL},
            {"lalt", KEY_LEFT_ALT},    {"ralt", KEY_RIGHT_ALT},     {"comma", KEY_COMMA},
            {"period", KEY_PERIOD},    {"slash", KEY_SLASH},        {"semicolon", KEY_SEMICOLON},
            {"minus", KEY_MINUS},      {"equal", KEY_EQUAL},        {"insert", KEY_INSERT},
            {"delete", KEY_DELETE},    {"home", KEY_HOME},          {"end", KEY_END},
            {"pageup", KEY_PAGE_UP},   {"pagedown", KEY_PAGE_DOWN},
        };
        for (const auto &entry : named)
            if (name == entry.name)
                return entry.key;
        return 0;
    }

    bool keyDown(int key)
    {
        return key >= kMouseBase ? IsMouseButtonDown(key - kMouseBase) : IsKeyDown(key);
    }

    using Keys = std::vector<int>;

    bool anyDown(const Keys &keys)
    {
        for (const int key : keys)
            if (keyDown(key))
                return true;
        return false;
    }

    // A key as a prompt can show it: see runtime/ps2_input.h.
    std::string keyLabel(int key)
    {
        if (key >= KEY_A && key <= KEY_Z)
            return std::string(1, static_cast<char>('A' + (key - KEY_A)));
        if (key >= KEY_ZERO && key <= KEY_NINE)
            return std::string(1, static_cast<char>('0' + (key - KEY_ZERO)));
        if (key >= KEY_F1 && key <= KEY_F12)
            return "F" + std::to_string(key - KEY_F1 + 1);
        if (key >= KEY_KP_0 && key <= KEY_KP_9)
            return "K" + std::to_string(key - KEY_KP_0);
        switch (key)
        {
        case kMouseBase + MOUSE_BUTTON_LEFT: return "LMB";
        case kMouseBase + MOUSE_BUTTON_RIGHT: return "RMB";
        case kMouseBase + MOUSE_BUTTON_MIDDLE: return "MMB";
        case kMouseBase + MOUSE_BUTTON_SIDE: return "M4";
        case kMouseBase + MOUSE_BUTTON_EXTRA: return "M5";
        case KEY_SPACE: return "SPC";
        case KEY_ENTER: return "ENT";
        case KEY_TAB: return "TAB";
        case KEY_ESCAPE: return "ESC";
        case KEY_BACKSPACE: return "BKS";
        case KEY_UP: return "UP";
        case KEY_DOWN: return "DN";
        case KEY_LEFT: return "LT";
        case KEY_RIGHT: return "RT";
        case KEY_LEFT_SHIFT:
        case KEY_RIGHT_SHIFT: return "SHF";
        case KEY_LEFT_CONTROL:
        case KEY_RIGHT_CONTROL: return "CTL";
        case KEY_LEFT_ALT:
        case KEY_RIGHT_ALT: return "ALT";
        case KEY_INSERT: return "INS";
        case KEY_DELETE: return "DEL";
        case KEY_HOME: return "HOM";
        case KEY_END: return "END";
        case KEY_PAGE_UP: return "PGU";
        case KEY_PAGE_DOWN: return "PGD";
        default: return "KEY";
        }
    }

    // A key as a sentence can name it: see runtime/ps2_input.h.
    std::string keyName(int key)
    {
        if ((key >= KEY_A && key <= KEY_Z) || (key >= KEY_ZERO && key <= KEY_NINE) || (key >= KEY_F1 && key <= KEY_F12))
            return keyLabel(key);
        if (key >= KEY_KP_0 && key <= KEY_KP_9)
            return "Numpad " + std::to_string(key - KEY_KP_0);
        switch (key)
        {
        case kMouseBase + MOUSE_BUTTON_LEFT: return "Left Mouse Button";
        case kMouseBase + MOUSE_BUTTON_RIGHT: return "Right Mouse Button";
        case kMouseBase + MOUSE_BUTTON_MIDDLE: return "Middle Mouse Button";
        case kMouseBase + MOUSE_BUTTON_SIDE: return "Mouse Button 4";
        case kMouseBase + MOUSE_BUTTON_EXTRA: return "Mouse Button 5";
        case KEY_SPACE: return "Space";
        case KEY_ENTER: return "Enter";
        case KEY_TAB: return "Tab";
        case KEY_ESCAPE: return "Escape";
        case KEY_BACKSPACE: return "Backspace";
        case KEY_UP: return "Up Arrow";
        case KEY_DOWN: return "Down Arrow";
        case KEY_LEFT: return "Left Arrow";
        case KEY_RIGHT: return "Right Arrow";
        case KEY_LEFT_SHIFT: return "Left Shift";
        case KEY_RIGHT_SHIFT: return "Right Shift";
        case KEY_LEFT_CONTROL: return "Left Ctrl";
        case KEY_RIGHT_CONTROL: return "Right Ctrl";
        case KEY_LEFT_ALT: return "Left Alt";
        case KEY_RIGHT_ALT: return "Right Alt";
        case KEY_INSERT: return "Insert";
        case KEY_DELETE: return "Delete";
        case KEY_HOME: return "Home";
        case KEY_END: return "End";
        case KEY_PAGE_UP: return "Page Up";
        case KEY_PAGE_DOWN: return "Page Down";
        default: return "";
        }
    }

    // The keys of [input] <name>, or of `otherwise` where the file has none.
    Keys bound(const char *name, const char *otherwise)
    {
        const std::string key = std::string("input.") + name;
        const char *value = ps2x::setting(nullptr, key.c_str());
        const std::string text = value ? value : otherwise;
        Keys keys;
        size_t at = 0;
        while (at <= text.size())
        {
            const size_t comma = std::min(text.find(',', at), text.size());
            if (const int found = keyNamed(text.substr(at, comma - at)))
                keys.push_back(found);
            at = comma + 1;
        }
        return keys;
    }

    struct Bindings
    {
        struct Button
        {
            uint16_t mask;
            Keys keys;
            std::string label; // of the first key, for a prompt
            std::string name;  // and for a sentence
        };
        std::vector<Button> buttons;
        Keys stick[4]; // up, down, left, right
        Keys look[4];
        bool mouseLook = true;
        float sensitivity = 1.0f;

        Bindings()
        {
            buttons = {
                {PAD_CROSS, bound("cross", "space, x")},
                {PAD_CIRCLE, bound("circle", "c, escape")},
                {PAD_SQUARE, bound("square", "mouse1, z, kp0")},
                {PAD_TRIANGLE, bound("triangle", "mouse2, v, kp1")},
                {PAD_L1, bound("l1", "q")},
                {PAD_R1, bound("r1", "e")},
                {PAD_L2, bound("l2", "lshift")},
                {PAD_R2, bound("r2", "rshift, f")},
                {PAD_L3, bound("l3", "none")},
                {PAD_R3, bound("r3", "mouse3")},
                {PAD_START, bound("start", "enter")},
                {PAD_SELECT, bound("select", "tab")},
                {PAD_UP, bound("up", "up")},
                {PAD_DOWN, bound("down", "down")},
                {PAD_LEFT, bound("left", "left")},
                {PAD_RIGHT, bound("right", "right")},
            };
            for (Button &button : buttons)
                if (!button.keys.empty())
                {
                    button.label = keyLabel(button.keys.front());
                    button.name = keyName(button.keys.front());
                }
            stick[0] = bound("stick_up", "w");
            stick[1] = bound("stick_down", "s");
            stick[2] = bound("stick_left", "a");
            stick[3] = bound("stick_right", "d");
            look[0] = bound("look_up", "none");
            look[1] = bound("look_down", "none");
            look[2] = bound("look_left", "none");
            look[3] = bound("look_right", "none");
            mouseLook = ps2x::settingOn(nullptr, "input.mouse_look", true);
            if (const char *value = ps2x::setting(nullptr, "input.mouse_sensitivity"))
                sensitivity = std::clamp(static_cast<float>(std::atof(value)), 0.05f, 20.0f);
        }

        bool anyKey() const
        {
            for (const Button &button : buttons)
                if (anyDown(button.keys))
                    return true;
            for (int i = 0; i < 4; ++i)
                if (anyDown(stick[i]) || anyDown(look[i]))
                    return true;
            return false;
        }
    };

    const Bindings &bindings()
    {
        static const Bindings instance;
        return instance;
    }

    // Is this controller being used? A stick has to be pushed well clear of
    // where a worn one rests.
    bool gamepadUsed(int gamepad)
    {
        for (int button = GAMEPAD_BUTTON_LEFT_FACE_UP; button <= GAMEPAD_BUTTON_RIGHT_THUMB; ++button)
            if (IsGamepadButtonDown(gamepad, button))
                return true;
        for (const int axis : {GAMEPAD_AXIS_LEFT_X, GAMEPAD_AXIS_LEFT_Y, GAMEPAD_AXIS_RIGHT_X, GAMEPAD_AXIS_RIGHT_Y})
            if (std::fabs(GetGamepadAxisMovement(gamepad, axis)) > 0.5f)
                return true;
        return false;
    }

    // The device to read now: see the top of the file.
    int chooseDevice()
    {
        int device = g_device.load(std::memory_order_relaxed);
        int first = -1;
        for (int gamepad = kGamepads - 1; gamepad >= 0; --gamepad)
            if (IsGamepadAvailable(gamepad))
                first = gamepad;
        if (device >= 0 && !IsGamepadAvailable(device))
            device = -2; // unplugged
        if (device == -2)
            device = first; // -1 with no controller: the keyboard
        // The one in use stays unless another is touched.
        const bool currentUsed = device >= 0 ? gamepadUsed(device) : bindings().anyKey();
        if (!currentUsed)
        {
            for (int gamepad = 0; gamepad < kGamepads; ++gamepad)
                if (gamepad != device && IsGamepadAvailable(gamepad) && gamepadUsed(gamepad))
                {
                    device = gamepad;
                    break;
                }
            if (device >= 0 && !gamepadUsed(device) && bindings().anyKey())
                device = -1;
        }
        if (device != g_device.load(std::memory_order_relaxed))
        {
            if (device >= 0)
                std::fprintf(stderr, "[input] controller %d: %s\n", device, GetGamepadName(device));
            else
                std::fprintf(stderr, "[input] keyboard and mouse\n");
        }
        g_device.store(device, std::memory_order_relaxed);
        return device;
    }

    uint8_t stickByte(float value)
    {
        return static_cast<uint8_t>(128 + value * 127);
    }

    uint8_t keyAxis(const Keys &low, const Keys &high, float other)
    {
        float value = other;
        if (anyDown(low))
            value -= 1.0f;
        if (anyDown(high))
            value += 1.0f;
        return stickByte(std::clamp(value, -1.0f, 1.0f));
    }
}

namespace
{
    bool scripted()
    {
        static const bool is = []
        {
            const char *script = std::getenv("PS2X_PAD_SCRIPT");
            return script != nullptr && script[0] != '\0';
        }();
        return is;
    }

    // The device in use into a pad's state: the buttons (a bit clear for
    // one that is down) and the sticks into data[2..7]. The buttons are
    // returned as well.
    uint16_t readDevices(uint8_t *data);
}

bool PSPadBackend::readState(int /*port*/, int /*slot*/, uint8_t *data, size_t size)
{
    if (!data || size < 32)
        return false;

    std::memset(data, 0, 32);
    data[0] = 0x01;
    data[1] = kPadAnalogMarker;
    data[2] = 0xFF;
    data[3] = 0xFF;
    data[4] = data[5] = data[6] = data[7] = kPadStickCenter;
    if (scripted())
        return true;

    const uint16_t btns = readDevices(data);

    // A menu of the host's has the pad, or had it: the game sees a pad
    // at rest until every button has been let go, so that the press
    // that closed the menu is not also the game's.
    if (const int blocked = g_blocked.load(std::memory_order_relaxed); blocked != 0)
    {
        int had = 2;
        if (blocked == 2 && btns == 0xFFFFu)
            g_blocked.compare_exchange_strong(had, 0, std::memory_order_relaxed);
        data[2] = data[3] = 0xFF;
        data[4] = data[5] = data[6] = data[7] = kPadStickCenter;
    }

    // PS2X_PAD_TRACE=1: the pad's state, each time it changes.
    static const bool trace = []
    {
        const char *value = std::getenv("PS2X_PAD_TRACE");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    if (trace)
    {
        static uint8_t last[8] = {};
        if (std::memcmp(last, data, sizeof(last)) != 0)
        {
            std::memcpy(last, data, sizeof(last));
            std::fprintf(stderr, "[pad] device %d pressed %04x left %3u,%3u right %3u,%3u\n",
                         g_device.load(std::memory_order_relaxed),
                         static_cast<unsigned>(static_cast<uint16_t>(~(data[2] | (data[3] << 8)))), data[6], data[7],
                         data[4], data[5]);
        }
    }
    return true;
}

namespace
{
uint16_t readDevices(uint8_t *data)
{
    uint16_t btns = 0xFFFFu;
    auto clearBit = [&btns](uint16_t mask)
    { btns &= ~mask; };

    const int device = chooseDevice();
    if (device >= 0)
    {
        if (IsGamepadButtonDown(device, GAMEPAD_BUTTON_LEFT_FACE_UP))
            clearBit(PAD_UP);
        if (IsGamepadButtonDown(device, GAMEPAD_BUTTON_LEFT_FACE_DOWN))
            clearBit(PAD_DOWN);
        if (IsGamepadButtonDown(device, GAMEPAD_BUTTON_LEFT_FACE_LEFT))
            clearBit(PAD_LEFT);
        if (IsGamepadButtonDown(device, GAMEPAD_BUTTON_LEFT_FACE_RIGHT))
            clearBit(PAD_RIGHT);
        if (IsGamepadButtonDown(device, GAMEPAD_BUTTON_RIGHT_FACE_DOWN))
            clearBit(PAD_CROSS);
        if (IsGamepadButtonDown(device, GAMEPAD_BUTTON_RIGHT_FACE_RIGHT))
            clearBit(PAD_CIRCLE);
        if (IsGamepadButtonDown(device, GAMEPAD_BUTTON_RIGHT_FACE_LEFT))
            clearBit(PAD_SQUARE);
        if (IsGamepadButtonDown(device, GAMEPAD_BUTTON_RIGHT_FACE_UP))
            clearBit(PAD_TRIANGLE);
        if (IsGamepadButtonDown(device, GAMEPAD_BUTTON_LEFT_TRIGGER_1))
            clearBit(PAD_L1);
        if (IsGamepadButtonDown(device, GAMEPAD_BUTTON_RIGHT_TRIGGER_1))
            clearBit(PAD_R1);
        if (IsGamepadButtonDown(device, GAMEPAD_BUTTON_LEFT_TRIGGER_2))
            clearBit(PAD_L2);
        if (IsGamepadButtonDown(device, GAMEPAD_BUTTON_RIGHT_TRIGGER_2))
            clearBit(PAD_R2);
        if (IsGamepadButtonDown(device, GAMEPAD_BUTTON_MIDDLE_RIGHT))
            clearBit(PAD_START);
        if (IsGamepadButtonDown(device, GAMEPAD_BUTTON_MIDDLE_LEFT))
            clearBit(PAD_SELECT);
        if (IsGamepadButtonDown(device, GAMEPAD_BUTTON_LEFT_THUMB))
            clearBit(PAD_L3);
        if (IsGamepadButtonDown(device, GAMEPAD_BUTTON_RIGHT_THUMB))
            clearBit(PAD_R3);

        float lx = GetGamepadAxisMovement(device, GAMEPAD_AXIS_LEFT_X);
        float ly = GetGamepadAxisMovement(device, GAMEPAD_AXIS_LEFT_Y);
        float rx = GetGamepadAxisMovement(device, GAMEPAD_AXIS_RIGHT_X);
        float ry = GetGamepadAxisMovement(device, GAMEPAD_AXIS_RIGHT_Y);
        data[6] = static_cast<uint8_t>(128 + lx * 127);
        data[7] = static_cast<uint8_t>(128 + ly * 127);
        data[4] = static_cast<uint8_t>(128 + rx * 127);
        data[5] = static_cast<uint8_t>(128 + ry * 127);
    }
    else
    {
        const Bindings &keys = bindings();
        for (const Bindings::Button &button : keys.buttons)
            if (anyDown(button.keys))
                clearBit(button.mask);
        data[6] = keyAxis(keys.stick[2], keys.stick[3], 0.0f);
        data[7] = keyAxis(keys.stick[0], keys.stick[1], 0.0f);
        data[4] = keyAxis(keys.look[2], keys.look[3], g_lookX.load(std::memory_order_relaxed));
        data[5] = keyAxis(keys.look[0], keys.look[1], g_lookY.load(std::memory_order_relaxed));
    }

    data[2] = static_cast<uint8_t>(btns & 0xFF);
    data[3] = static_cast<uint8_t>(btns >> 8);
    return btns;
}
}

int ps2xInputDevice()
{
    return g_device.load(std::memory_order_relaxed);
}

const char *ps2xInputDeviceName()
{
    const int device = g_device.load(std::memory_order_relaxed);
    const char *name = device >= 0 ? GetGamepadName(device) : nullptr;
    return name ? name : "";
}

const char *ps2xInputKeyLabel(uint16_t button)
{
    for (const Bindings::Button &bound : bindings().buttons)
        if (bound.mask == button)
            return bound.label.c_str();
    return "";
}

const char *ps2xInputKeyName(uint16_t button)
{
    for (const Bindings::Button &bound : bindings().buttons)
        if (bound.mask == button)
            return bound.name.c_str();
    return "";
}

uint16_t ps2xInputButtons()
{
    if (scripted())
        return 0u;
    uint8_t data[8] = {0, 0, 0xFF, 0xFF, kPadStickCenter, kPadStickCenter, kPadStickCenter, kPadStickCenter};
    uint16_t down = static_cast<uint16_t>(~readDevices(data));
    // The left stick past half-way counts as the directional buttons.
    if (data[6] < 64)
        down |= PAD_LEFT;
    if (data[6] > 192)
        down |= PAD_RIGHT;
    if (data[7] < 64)
        down |= PAD_UP;
    if (data[7] > 192)
        down |= PAD_DOWN;
    return down;
}

void ps2xInputBlock(bool block)
{
    if (block)
        g_blocked.store(1, std::memory_order_relaxed);
    else
    {
        int had = 1;
        g_blocked.compare_exchange_strong(had, 2, std::memory_order_relaxed);
    }
}

void ps2xInputSetMouseLook(bool on)
{
    g_mouseLook.store(on ? 1 : 0, std::memory_order_relaxed);
}

void ps2xInputSetMouseSensitivity(float sensitivity)
{
    g_sensitivity.store(std::clamp(sensitivity, 0.05f, 20.0f), std::memory_order_relaxed);
}

void ps2xInputSetUiOpen(bool open)
{
    g_uiOpen.store(open, std::memory_order_relaxed);
}

void ps2xInputMouseLookDirect(bool on)
{
    g_lookDirect.store(on, std::memory_order_relaxed);
}

bool ps2xInputTakeMouseLook(float &dx, float &dy)
{
    std::lock_guard<std::mutex> lock(g_takenMutex);
    dx = g_takeX;
    dy = g_takeY;
    g_takeX = g_takeY = 0.0f;
    if (!g_takeable)
        dx = dy = 0.0f;
    return g_takeable;
}

void ps2xInputFrame()
{
    // Controllers this build of the window library has no layout for: a
    // gamecontrollerdb.txt (the SDL list) beside the game is read once.
    static const bool mappingsLoaded = []
    {
        const char *named = std::getenv("PS2X_GAMEPAD_DB");
        const char *path = named && *named ? named : "gamecontrollerdb.txt";
        if (FileExists(path))
            if (char *text = LoadFileText(path))
            {
                SetGamepadMappings(text);
                UnloadFileText(text);
            }
        return true;
    }();
    (void)mappingsLoaded;

    // The mouse is the right stick while the keyboard and mouse are in use
    // and the game's window has the focus; it is the desktop's the rest of
    // the time.
    static bool captured = false;
    static float lookX = 0.0f, lookY = 0.0f;
    const Bindings &keys = bindings();
    const int look = g_mouseLook.load(std::memory_order_relaxed);
    const float asked = g_sensitivity.load(std::memory_order_relaxed);
    const float sensitivity = asked >= 0.0f ? asked : keys.sensitivity;
    const bool want = (look < 0 ? keys.mouseLook : look != 0) && g_device.load(std::memory_order_relaxed) == -1 &&
                      IsWindowFocused() &&
                      !g_uiOpen.load(std::memory_order_relaxed);
    if (want != captured)
    {
        captured = want;
        if (captured)
            DisableCursor();
        else
            EnableCursor();
        lookX = lookY = 0.0f;
    }
    // A game that takes the movement itself has it as it comes, and the
    // stick stays at rest; not while a menu of the host's holds the pad, or
    // the camera would jump by all the mouse did there when it closes.
    const bool direct = captured && g_lookDirect.load(std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lock(g_takenMutex);
        g_takeable = direct && g_blocked.load(std::memory_order_relaxed) == 0;
        if (g_takeable)
        {
            const Vector2 moved = GetMouseDelta();
            g_takeX += moved.x * sensitivity;
            g_takeY += moved.y * sensitivity;
        }
        else
        {
            g_takeX = g_takeY = 0.0f;
        }
    }
    if (direct)
    {
        lookX = lookY = 0.0f;
    }
    else if (captured)
    {
        // A stick is a speed and the mouse a distance: how fast the mouse
        // is moving, smoothed over a twentieth of a second, is how far the
        // stick is pushed.
        const float seconds = std::clamp(GetFrameTime(), 0.0005f, 0.1f);
        const Vector2 moved = GetMouseDelta();
        constexpr float kFullAt = 1700.0f; // pixels a second that push the stick all the way
        const float blend = 1.0f - std::exp(-seconds / 0.05f);
        lookX += (moved.x / seconds * sensitivity / kFullAt - lookX) * blend;
        lookY += (moved.y / seconds * sensitivity / kFullAt - lookY) * blend;
    }
    g_lookX.store(std::clamp(lookX, -1.0f, 1.0f), std::memory_order_relaxed);
    g_lookY.store(std::clamp(lookY, -1.0f, 1.0f), std::memory_order_relaxed);
}
