#include "ps2recomp/Emitters/function_table_emitter.h"
#include "ps2recomp/code_generator.h"
#include "ps2recomp/ps2_recompiler.h"
#include "ps2recomp/recompiler_reporter.h"
#include "ps2recomp/types.h"

#include <algorithm>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

namespace ps2recomp
{
    FunctionTableEmitter::FunctionTableEmitter(CodeGenerator &codeGenerator)
        : m_codeGenerator(codeGenerator)
    {
    }

    std::string FunctionTableEmitter::emit(const std::vector<Function> &functions, const std::map<uint32_t, std::string> &stubs)
    {
        (void)stubs;

        CodeGenerator &cg = m_codeGenerator;
        std::vector<std::pair<uint32_t, std::string>> entries;
        std::unordered_set<uint32_t> registeredAddresses;

        auto addEntry = [&](uint32_t address, const std::string &name)
        {
            if (name.empty())
            {
                return;
            }
            if ((address & 3u) != 0u)
            {
                std::ostringstream oss;
                oss << "Unaligned function table entry for " << name << " at 0x" << std::hex << address;

                if (cg.m_reporter)
                {
                    cg.m_reporter->errorAt("function-table", name, address, oss.str());
                }

                throw std::runtime_error(oss.str());
            }
            if (!registeredAddresses.insert(address).second)
            {
                return;
            }
            entries.emplace_back(address, name);
        };

        std::vector<std::pair<uint32_t, std::string>> normalFunctions;
        std::vector<std::pair<uint32_t, std::string>> stubFunctions;
        std::vector<std::pair<uint32_t, std::string>> systemCallFunctions;
        std::vector<std::pair<uint32_t, std::string>> libraryFunctions;

        for (const auto &function : functions)
        {
            if (!function.isRecompiled && !function.isStub && !function.isSkipped)
                continue;

            std::string generatedName = cg.getFunctionName(function.start);

            if (function.isSkipped)
            {
                libraryFunctions.emplace_back(function.start, generatedName);
            }
            else if (function.isStub)
            {
                const auto target = PS2Recompiler::resolveStubTarget(function.name);
                if (target == StubTarget::Syscall)
                {
                    systemCallFunctions.emplace_back(function.start, generatedName);
                }
                else
                {
                    stubFunctions.emplace_back(function.start, generatedName);
                }
            }
            else
            {
                normalFunctions.emplace_back(function.start, generatedName);
            }
        }

        if (cg.m_bootstrapInfo.valid)
        {
            std::string entryTarget = cg.m_bootstrapInfo.entryName;
            if (entryTarget.empty())
            {
                entryTarget = cg.getFunctionName(cg.m_bootstrapInfo.entry);
            }
            if (entryTarget.empty())
            {
                throw std::runtime_error("No entry function name available for registration.");
            }
            addEntry(cg.m_bootstrapInfo.entry, entryTarget);
        }

        for (const auto &[address, name] : normalFunctions)
        {
            addEntry(address, name);
        }

        for (const auto &[ownerStart, targets] : cg.m_resumeEntryTargetsByOwner)
        {
            const std::string ownerName = cg.getFunctionName(ownerStart);
            if (ownerName.empty())
            {
                continue;
            }

            for (uint32_t target : targets)
            {
                addEntry(target, ownerName);
            }
        }

        for (const auto &[address, name] : stubFunctions)
        {
            addEntry(address, name);
        }
        for (const auto &[address, name] : systemCallFunctions)
        {
            addEntry(address, name);
        }
        for (const auto &[address, name] : libraryFunctions)
        {
            addEntry(address, name);
        }

        std::sort(entries.begin(), entries.end(), [](const auto &a, const auto &b)
                  { return a.first < b.first; });

        std::stringstream ss;
        ss << "#include \"ps2_runtime.h\"\n";
        ss << "#include <ps2_recompiled_functions.h>\n";
        ss << "#include \"ps2_stubs.h\"\n";
        ss << "#include <ps2_recompiled_stubs.h>\n";
        ss << "#include \"ps2_syscalls.h\"\n\n";

        if (!cg.m_overlayName.empty())
        {
            emitOverlay(ss, functions, entries);
            return ss.str();
        }

        uint32_t tableBase = 0u;
        uint32_t tableEnd = 0u;
        uint32_t slotCount = 0u;
        if (!entries.empty())
        {
            tableBase = entries.front().first & ~3u;
            tableEnd = (entries.back().first + 4u + 3u) & ~3u;
            // The overlays put their functions in this table as they are loaded.
            for (const auto &[rangeStart, rangeEnd] : cg.m_overlayRanges)
            {
                tableBase = std::min(tableBase, rangeStart & ~3u);
                tableEnd = std::max(tableEnd, (rangeEnd + 3u) & ~3u);
            }
            slotCount = (tableEnd - tableBase) >> 2;
        }

        ss << "extern const uint32_t g_ps2RecompiledFunctionTableBase = 0x" << std::hex << tableBase << "u;\n";
        ss << "extern const uint32_t g_ps2RecompiledFunctionTableEnd = 0x" << std::hex << tableEnd << "u;\n";
        ss << "extern const uint32_t g_ps2RecompiledFunctionTableSlotCount = " << std::dec << slotCount << "u;\n";
        ss << "PS2Runtime::RecompiledFunction g_ps2RecompiledFunctionTable[" << std::dec << (slotCount == 0u ? 1u : slotCount) << "u] = {};\n\n";

        ss << "namespace {\n";
        ss << "struct GeneratedFunctionTableInitializer {\n";
        ss << "    GeneratedFunctionTableInitializer() {\n";
        for (const auto &[address, name] : entries)
        {
            const uint32_t slot = (address - tableBase) >> 2;
            ss << "        g_ps2RecompiledFunctionTable[" << std::dec << slot << "] = " << name
               << "; // 0x" << std::hex << address << std::dec << "\n";
        }
        ss << "    }\n";
        ss << "};\n";
        ss << "static const GeneratedFunctionTableInitializer g_generatedFunctionTableInitializer;\n";
        ss << "}\n";

        return ss.str();
    }

    namespace
    {
        // The runtime hashes guest memory the same way (PS2Runtime::refreshOverlays).
        constexpr uint64_t kFnvOffset = 0xCBF29CE484222325ull;
        constexpr uint32_t kOverlayProbeBytes = 256u;

        uint64_t fnv1a64(const uint8_t *data, size_t size, uint64_t hash)
        {
            for (size_t i = 0; i < size; ++i)
            {
                hash = (hash ^ data[i]) * 0x100000001B3ull;
            }
            return hash;
        }

        const uint8_t *sectionBytes(const std::vector<Section> &sections, uint32_t start, uint32_t end)
        {
            for (const Section &section : sections)
            {
                if (section.data != nullptr && !section.isBSS &&
                    start >= section.address && end <= section.address + section.size)
                {
                    return section.data + (start - section.address);
                }
            }
            return nullptr;
        }
    }

    // An overlay does not own the table: it hands the runtime its functions
    // and a hash of its instructions, by which the runtime knows the image
    // when the game has loaded it.
    void FunctionTableEmitter::emitOverlay(std::stringstream &ss,
                                           const std::vector<Function> &functions,
                                           const std::vector<std::pair<uint32_t, std::string>> &entries)
    {
        CodeGenerator &cg = m_codeGenerator;
        const std::string &name = cg.m_overlayName;

        std::vector<std::pair<uint32_t, uint32_t>> code;
        for (const Function &function : functions)
        {
            if ((function.isRecompiled || function.isStub || function.isSkipped) && function.end > function.start &&
                sectionBytes(cg.m_sections, function.start, function.end) != nullptr)
            {
                code.emplace_back(function.start, function.end);
            }
        }
        std::sort(code.begin(), code.end());
        std::vector<std::pair<uint32_t, uint32_t>> merged;
        for (const auto &range : code)
        {
            if (!merged.empty() && range.first <= merged.back().second)
            {
                merged.back().second = std::max(merged.back().second, range.second);
            }
            else
            {
                merged.push_back(range);
            }
        }
        if (merged.empty() || entries.empty())
        {
            throw std::runtime_error("Overlay '" + name + "' has no code to register.");
        }

        uint64_t codeHash = kFnvOffset;
        for (const auto &[start, end] : merged)
        {
            codeHash = fnv1a64(sectionBytes(cg.m_sections, start, end), end - start, codeHash);
        }
        const uint32_t probeSize = std::min(merged.front().second - merged.front().first, kOverlayProbeBytes);
        const uint64_t probeHash = fnv1a64(
            sectionBytes(cg.m_sections, merged.front().first, merged.front().first + probeSize), probeSize, kFnvOffset);

        ss << "namespace {\n";
        ss << "const PS2Runtime::OverlayFunction kOverlayFunctions_" << name << "[] = {\n";
        for (const auto &[address, functionName] : entries)
        {
            ss << "    {0x" << std::hex << address << std::dec << "u, " << functionName << "},\n";
        }
        ss << "};\n";
        ss << "const PS2Runtime::OverlayCodeRange kOverlayCode_" << name << "[] = {\n";
        for (const auto &[start, end] : merged)
        {
            ss << "    {0x" << std::hex << start << "u, 0x" << end << std::dec << "u},\n";
        }
        ss << "};\n";
        ss << "struct OverlayRegistrar_" << name << " {\n";
        ss << "    OverlayRegistrar_" << name << "() {\n";
        ss << "        PS2Runtime::registerOverlay({\"" << name << "\", kOverlayFunctions_" << name << ", "
           << entries.size() << "u, kOverlayCode_" << name << ", " << merged.size() << "u, 0x"
           << std::hex << probeHash << "ull, 0x" << codeHash << std::dec << "ull});\n";
        ss << "    }\n";
        ss << "};\n";
        ss << "static const OverlayRegistrar_" << name << " g_overlayRegistrar_" << name << ";\n";
        ss << "}\n";
    }
}
