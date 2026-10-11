#ifndef PS2_INPUT_H
#define PS2_INPUT_H

// What the host's input (ps2_pad.cpp) can tell a game about itself, for a
// game that shows the player which button to press.

#include <cstdint>

// The device driving the pad: a controller's index; -1 for the keyboard and
// mouse; -2 before anything has been read.
int ps2xInputDevice();

// That controller's name as the system gives it ("Xbox Controller",
// "DualSense Wireless Controller"); "" for the keyboard and mouse.
const char *ps2xInputDeviceName();

// The first key bound to a pad button, as a label of up to three characters
// for a prompt: "Z", "F5", "SPC", "LMB"; "UP" "DN" "LT" "RT" for the arrow
// keys; "" when nothing is bound. `button` is the button's bit in the pad
// word (0x4000 cross, 0x2000 circle, 0x8000 square, 0x1000 triangle, ...).
const char *ps2xInputKeyLabel(uint16_t button);

#endif
