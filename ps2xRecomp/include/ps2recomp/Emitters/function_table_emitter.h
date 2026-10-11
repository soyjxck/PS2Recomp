#ifndef PS2RECOMP_FUNCTION_TABLE_EMITTER_H
#define PS2RECOMP_FUNCTION_TABLE_EMITTER_H

#include <cstdint>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace ps2recomp
{
    struct Function;
    class CodeGenerator;

    class FunctionTableEmitter
    {
    public:
        explicit FunctionTableEmitter(CodeGenerator &codeGenerator);

        std::string emit(const std::vector<Function> &functions, const std::map<uint32_t, std::string> &stubs);

    private:
        void emitOverlay(std::stringstream &ss,
                         const std::vector<Function> &functions,
                         const std::vector<std::pair<uint32_t, std::string>> &entries);

        CodeGenerator &m_codeGenerator;
    };
}

#endif // PS2RECOMP_FUNCTION_TABLE_EMITTER_H
