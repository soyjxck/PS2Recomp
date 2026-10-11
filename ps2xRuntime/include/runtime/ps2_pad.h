#ifndef PS2_PAD_H
#define PS2_PAD_H

#include <cstddef>
#include <cstdint>

class PSPadBackend
{
public:
    PSPadBackend() = default;
    ~PSPadBackend() = default;

    bool readState(int port, int slot, uint8_t *data, size_t size);
};

// The part of the host's input that belongs to the thread with the window:
// the mouse given to the game while the keyboard and mouse are in use, and
// taken back when the window loses the focus or a debug window is open.
// Called once for every pass of the host's loop.
void ps2xInputFrame();
void ps2xInputSetUiOpen(bool open);

#endif
