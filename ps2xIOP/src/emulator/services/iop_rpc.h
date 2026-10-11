#pragma once

#include "ps2x/iop/iop_types.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>

namespace ps2x::iop
{
    class IopHost;
}

namespace ps2x::iop::detail
{
    struct IopCpuState;
    class IopKernel;
    class IopMemory;

    class IopGuestExecutor
    {
    public:
        virtual ~IopGuestExecutor() = default;

        [[nodiscard]] virtual uint32_t executeGuestFunction(uint32_t address,
                                                            uint32_t a0,
                                                            uint32_t a1,
                                                            uint32_t a2,
                                                            uint32_t a3,
                                                            uint32_t gp) = 0;
        [[nodiscard]] virtual uint32_t executeGuestFunctionWithBudget(uint32_t address,
                                                                      uint32_t a0,
                                                                      uint32_t a1,
                                                                      uint32_t a2,
                                                                      uint32_t a3,
                                                                      uint32_t gp,
                                                                      uint32_t instructionBudget)
        {
            return executeGuestFunction(address, a0, a1, a2, a3, gp);
        }
    };

    class IopRpcBridge
    {
    public:
        IopRpcBridge(IopHost &host, IopMemory &memory, IopKernel &kernel) noexcept;

        void reset();
        [[nodiscard]] bool dispatchSifManImport(uint16_t ordinal, IopCpuState &cpu);
        [[nodiscard]] bool dispatchSifCmdImport(uint16_t ordinal, IopCpuState &cpu, IopGuestExecutor &executor);
        [[nodiscard]] RpcResult handleRpc(const RpcRequest &request, IopGuestExecutor &executor);
        void onSifTransfer(const SifTransfer &transfer);
        void removeServersInRange(uint32_t base, uint32_t size);

        [[nodiscard]] bool hasServer(uint32_t sid) const noexcept;
        [[nodiscard]] size_t serverCount() const noexcept { return m_servers.size(); }

    private:
        struct RpcServer
        {
            uint32_t sid = 0;
            uint32_t serverData = 0;
            uint32_t function = 0;
            uint32_t gp = 0;
            uint32_t buffer = 0;
            uint32_t callback = 0;
            uint32_t callbackBuffer = 0;
            uint32_t queue = 0;
        };

        // A call a module has made to a server on the EE (sceSifCallRpc on
        // the IOP), until the EE's function has returned.
        struct EeCall
        {
            uint32_t client = 0;
            uint32_t receive = 0;
            uint32_t receiveSize = 0;
            uint32_t endFunction = 0;
            uint32_t endParameter = 0;
            uint32_t gp = 0;
            int blockedThread = 0;
        };

        void completeEeCall(uint32_t token, const void *reply, size_t replySize, IopGuestExecutor &executor);

        IopHost &m_host;
        IopMemory &m_memory;
        IopKernel &m_kernel;
        std::unordered_map<uint32_t, RpcServer> m_servers;
        // Clients bound to a server on the EE, by address, and its SID.
        std::unordered_map<uint32_t, uint32_t> m_eeClients;
        std::unordered_map<uint32_t, EeCall> m_eeCalls;
        // What a reply checks before it touches this object: reset() and the
        // destructor let go of it.
        std::shared_ptr<int> m_eeCallEpoch = std::make_shared<int>(0);
        uint32_t m_nextEeCall = 1u;
        uint32_t m_nextRpcId = 1u;
        uint32_t m_nextDmaId = 1u;
        bool m_sifInitialized = false;
    };
}
