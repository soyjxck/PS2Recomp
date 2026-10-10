#pragma once

#include <cstdint>

class GS;

// Runs the GS (GIF packet decoding and rasterising) on its own thread, so the
// EE does not wait for a frame's drawing while it prepares the next packets.
//
// Packets are processed in the order they are posted. The EE side calls
// drain() wherever the guest could observe that the GS is behind: before a
// vblank is delivered, before it idles waiting for an interrupt (FINISH and
// SIGNAL come out of the packets), and before it reads GS state. Emulated
// timing therefore does not depend on how fast the host draws.
//
// PS2X_GS_THREAD=0 processes packets on the calling thread, as before.
namespace GsWorker
{
    void start(GS &gs);
    void stop();
    bool running();

    // Process a GIF packet (copied) after everything posted before it.
    void post(const uint8_t *data, uint32_t sizeBytes);

    // Returns when everything posted so far has been processed. True if there
    // was anything to wait for. Does nothing on the worker thread itself.
    bool drain();
}
