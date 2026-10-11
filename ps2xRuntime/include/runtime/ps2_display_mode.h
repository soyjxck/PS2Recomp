#ifndef PS2_DISPLAY_MODE_H
#define PS2_DISPLAY_MODE_H

// How the game's window sits on the host's screen:
//   Window      a window, with its frame, the size the settings give it;
//   Borderless  a window without a frame over the whole screen: the desktop
//               stays as it is underneath, and other windows come and go
//               over it at once;
//   Exclusive   the screen itself, taken over in its own mode ("full screen"
//               proper): nothing of the desktop is composed with it.
// [display] fullscreen in the settings file (PS2X_FULLSCREEN) says which:
// off, borderless or exclusive. on (and yes, true, 1), from when there was
// one mode besides the window, is borderless, as it was.

namespace ps2x
{
    enum class DisplayMode
    {
        Window,
        Borderless,
        Exclusive,
    };

    // The setting's text for a mode, and the mode a text names (`otherwise`
    // for none, or one that names nothing).
    const char *displayModeValue(DisplayMode mode);
    DisplayMode displayModeFrom(const char *text, DisplayMode otherwise);

    // The window as it is now, and a change of it. On the thread with the
    // window, between two frames.
    DisplayMode displayMode();
    void setDisplayMode(DisplayMode mode);

    // F11: between the window and whichever of the other two was last in
    // use (borderless until one has been).
    void toggleDisplayMode();
}

#endif // PS2_DISPLAY_MODE_H
