#pragma once

#include <cstdint>

#include "runtime/gs/gs_types.h"

class GS;
struct GSRegisters;

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
    // `registers`: the privileged registers the guest writes (lagsInterrupts
    // reads its interrupt mask there).
    void start(GS &gs, const GSRegisters *registers = nullptr);
    void stop();
    bool running();

    // Process a GIF packet (copied) after everything posted before it.
    void post(const uint8_t *data, uint32_t sizeBytes);

    // Where the EE side waits. PS2X_GS_STATS=1 gives the time by these.
    enum class Wait : uint8_t
    {
        Other,
        Interrupt,    // a packet that raises FINISH or SIGNAL was sent
        InterruptDue, // the same, PS2X_GS_FINISH_DELAY_US later
        VBlank,       // a vblank is about to be delivered
        Idle,         // no thread of the guest's can run
        Csr,          // the guest read CSR
        SyncPath,     // sceGsSyncPath
        Vif1,         // the guest used a VIF1 register
        Vu1Memory,    // the guest used VU1's memory
        Count,
    };

    // Returns when everything posted so far has been processed. True if there
    // was anything to wait for. Does nothing on the worker thread itself.
    bool drain(Wait why = Wait::Other);

    // The same for VU1's queue alone: what its registers and memory need.
    bool drainVu1(Wait why);

    // How many vblanks the GS, and VU1 before it, may be behind the EE:
    // ps2x::setGsLag, or PS2X_GS_LAG=<n>. 0, the default: never behind, as
    // described above. Otherwise the EE does not wait at a vblank for what it
    // sent during the frame, only for what it had sent n vblanks before, and
    // it does not wait before it idles, in sceGsSyncPath, or for a FINISH or
    // SIGNAL while the guest has both masked: a frame is drawn while the EE
    // prepares the next one. A game that sends a frame and then waits for the
    // vblank, as most do, otherwise has the host do the two in turn.
    //
    // The guest cannot tell: the paths read as idle either way, the picture
    // is shown with the display registers of the vblank the GS has reached
    // (postVBlank), the library's register writes take their turn in the
    // queue, and whatever reads the GS's or VU1's state (CSR, a stored image,
    // VU1's memory) or changes the interrupt mask still waits for the queue.
    // So emulated timing still does not depend on how fast the host draws.
    // What changes is the host's: the picture is up to n vblanks late when
    // the host cannot keep up, and not at all when it can.
    uint32_t lagVBlanks();

    // Whether the GS may be behind and the guest has both of its interrupts,
    // SIGNAL and FINISH, masked: nothing queued can then interrupt the guest,
    // and what it reads of the two in CSR waits for the queue by itself.
    bool lagsInterrupts();

    // The EE's vblank, with the display registers as they are at it, in its
    // place among what was posted.
    using Display = GSDisplayRegisters;
    void postVBlank(const Display &display);

    // While the GS may lag, the display of the last vblank it has passed:
    // what its VRAM is shown with (GS::latchHostPresentationFrame). False
    // otherwise, or before the first.
    bool shownDisplay(Display &display);

    // Returns when the GS has passed that vblank. True if it had to wait.
    bool waitForVBlank(uint64_t tick, Wait why = Wait::VBlank);

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
