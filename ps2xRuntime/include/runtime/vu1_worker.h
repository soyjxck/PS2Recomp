#pragma once

#include <cstdint>

class PS2Memory;

// Runs VIF1 (unpacking into VU1 memory, MSCAL, DIRECT) and the VU1
// microprograms on their own thread. VIF1 data and PATH3 GIF packets are
// copied to one queue and handled in the order the EE sent them, so the three
// paths still reach the GS in that order.
//
// The EE never sees VIF1 or VU1 busy: a transfer completes when it is queued,
// as it did when everything ran inline. Wherever the guest could look at what
// the worker owns (VIF1 registers, VU1 memory) the EE waits for the queue to
// empty first, and so does anything that waits for the GS.
//
// Opt-in with PS2X_VU_THREAD=1. A game that sends its display list and FINISH
// in one go (most do) makes the EE wait for the whole list either way, because
// the interrupt has to be pending when the send returns; the worker then only
// moves the VU1 work to another core without shortening the frame.
namespace Vu1Worker
{
    void start(PS2Memory &memory);
    void stop();
    bool running();
    bool onWorkerThread();

    // True when the caller should queue the work instead of doing it.
    bool defers();

    void postVif1(const uint8_t *data, uint32_t sizeBytes);
    void postPath3(const uint8_t *data, uint32_t sizeBytes);

    // Returns when everything posted so far has been handled. True if there
    // was anything to wait for.
    bool drain();
}
