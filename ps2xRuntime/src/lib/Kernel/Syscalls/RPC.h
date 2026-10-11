#pragma once

#include "ps2_syscalls.h"

#include <cstddef>
#include <functional>

namespace ps2_syscalls
{
    void SifStopModule(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime);
    void SifLoadModule(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime);
    void SifInitRpc(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime);
    void SifBindRpc(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime);
    void SifCallRpc(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime);
    void SifRegisterRpc(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime);
    void SifCheckStatRpc(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime);
    void SifSetRpcQueue(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime);
    void SifRemoveRpcQueue(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime);
    void SifRemoveRpc(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime);
    void sceSifCallRpc(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime);
    void sceSifSendCmd(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime);
    void sceRpcGetPacket(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime);

    // IOP -> EE RPC, for the IOP side's sceSifBindRpc and sceSifCallRpc: the
    // descriptor of a server the game registered on the EE (0 when there is
    // none under this SID), and one call of its function. `completion` gets
    // the reply when the function has returned, which is after this call.
    uint32_t findEeRpcServer(uint8_t *rdram, uint32_t sid);
    bool callEeRpcServer(uint8_t *rdram,
                         PS2Runtime *runtime,
                         uint32_t sid,
                         uint32_t function,
                         const void *send,
                         size_t sendSize,
                         size_t receiveSize,
                         std::function<void(const void *reply, size_t replySize)> completion);
}
