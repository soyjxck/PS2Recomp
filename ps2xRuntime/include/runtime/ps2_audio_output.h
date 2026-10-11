#ifndef PS2_AUDIO_OUTPUT_H
#define PS2_AUDIO_OUTPUT_H

#include <cstddef>
#include <cstdint>

// The SPU2's output on the host's audio device.
//
// The emulated IOP makes 48,000 frames for each of its seconds, in bursts as
// the EE's thread gets to them; the device takes them evenly, on a thread of
// its own, by a clock that is never quite the game's. Between the two is a
// short queue, and the device's side plays a little faster or slower (half a
// percent at most) to keep it about half full.
namespace ps2x::audio
{
    // Between InitAudioDevice() and CloseAudioDevice(). An unpaced run
    // (PS2X_TURBO) is not played, only dumped if that is asked for.
    void startOutput();
    void stopOutput();

    // Frames from the SPU2, left then right. From the thread that runs the
    // IOP; never waits.
    //   PS2X_AUDIO_DUMP=<file.wav>   also writes every frame to that file
    void pushOutput(const int16_t *samples, size_t frames);
}

#endif
