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

// The same key named in full, for a sentence: "Q", "F5", "Space", "Enter",
// "Left Shift", "Up Arrow", "Numpad 0", "Left Mouse Button", "Mouse Button
// 4"; "" when nothing is bound.
const char *ps2xInputKeyName(uint16_t button);

// The pad as it is now, for a menu of the host's own: a bit set for each
// button that is down (the pad word's bits, as above), and the left stick
// past half-way counted as the directional buttons. It reads the devices
// itself, so it works while the game is not reading the pad; for the
// thread with the window. Nothing in a run driven by PS2X_PAD_SCRIPT.
uint16_t ps2xInputButtons();

// A menu of the host's takes the pad (true) and gives it back (false):
// the game sees a pad at rest meanwhile, and afterwards until every
// button has been let go.
void ps2xInputBlock(bool block);

// [input] mouse_look and mouse_sensitivity, changed while the game runs.
void ps2xInputSetMouseLook(bool on);
void ps2xInputSetMouseSensitivity(float sensitivity);

// For a game that turns its camera by the mouse itself. A stick is a speed
// and the mouse a distance, so the mouse as the right stick is the best a
// pad can carry and no more: it lags, and it tops out at whatever the game
// does with a stick pushed all the way.
//
// ps2xInputMouseLookDirect(true) says the game takes the mouse's movement
// instead: the right stick is then left at rest by the mouse (keys bound to
// it still push it). ps2xInputTakeMouseLook gives what the mouse has moved
// since it was last asked, right and down positive, in the mouse's own
// counts times the sensitivity, and true; or false and nothing while the
// mouse is not the camera's: mouse_look off, another device in use, the
// window without the focus, a menu or debug window open, the pad held by
// a menu of the host's, a run driven by PS2X_PAD_SCRIPT. From any thread.
void ps2xInputMouseLookDirect(bool on);
bool ps2xInputTakeMouseLook(float &dx, float &dy);

#endif
