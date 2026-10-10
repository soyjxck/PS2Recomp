#pragma once

#include <cstdint>

class GS;

// Runs the GS (GIF packet decoding and rasterising) on its own thread, so the
// EE does not wait for a frame's drawing while it prepares the next packets.
//
// Packets are processed in the order they are posted. The EE side waits for
// the queue wherever the guest could observe that the GS is behind: when it
// sends a packet that raises FINISH or SIGNAL (the interrupt is then pending
// at the same emulated time as when the GS ran inline), before a vblank is
// delivered, before it idles, and before it reads GS state. Emulated timing
// therefore does not depend on how fast the host draws.
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

    // Called by a sender on the EE side after posting a packet that raises an
    // interrupt. Without a hook the caller waits for the queue (the interrupt
    // is pending when the send returns). A hook takes over the wait: it is
    // expected to arrange a drain() a fixed emulated time later, which models
    // the time the GS takes to draw and lets the EE run meanwhile.
    void interruptSent();
    void setInterruptSentHook(void (*hook)());

    // Whether a GIF packet writes the FINISH or SIGNAL register.
    bool raisesInterrupt(const uint8_t *gifPacket, uint32_t sizeBytes);

    // The same question for VIF1 data, where the registers sit in DIRECT
    // payloads: a scan for the register writes that may report one too many.
    bool mayRaiseInterrupt(const uint8_t *vifData, uint32_t sizeBytes);
}
