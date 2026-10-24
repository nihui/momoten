// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#ifndef MOMOTEN_SPIRV_TEST_HELPERS_H
#define MOMOTEN_SPIRV_TEST_HELPERS_H

#include "spirv.hpp"

#include <cstdint>
#include <fstream>
#include <map>
#include <vector>

inline bool read_spirv(const char* path, std::vector<uint32_t>& words)
{
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream)
        return false;
    const std::streamoff size = stream.tellg();
    if (size < 0 || size % 4 != 0)
        return false;
    words.resize(static_cast<size_t>(size) / 4);
    stream.seekg(0, std::ios::beg);
    return words.empty() || static_cast<bool>(stream.read(reinterpret_cast<char*>(words.data()), size));
}

inline bool decorate_aliased_buffers(std::vector<uint32_t>& words, uint32_t bindings)
{
    // OpDecorate Binding identifies the views independently of IDs/names.
    std::vector<uint32_t> decorations;
    size_t insertion = words.size();
    for (size_t offset = 5; offset < words.size(); offset += words[offset] >> spv::WordCountShift)
    {
        const uint32_t count = words[offset] >> spv::WordCountShift;
        const spv::Op opcode = static_cast<spv::Op>(words[offset] & spv::OpCodeMask);
        if (!count || count > words.size() - offset)
            return false;
        if (opcode == spv::OpDecorate && count == 4 && words[offset + 2] == spv::DecorationBinding && words[offset + 3] < bindings)
        {
            decorations.push_back((3u << spv::WordCountShift) | spv::OpDecorate);
            decorations.push_back(words[offset + 1]);
            decorations.push_back(spv::DecorationAliased);
        }
        if (opcode >= spv::OpTypeVoid && opcode <= spv::OpTypeForwardPointer && insertion == words.size())
            insertion = offset;
    }
    if (decorations.size() != bindings * 3 || insertion == words.size())
        return false;
    words.insert(words.begin() + insertion, decorations.begin(), decorations.end());
    return true;
}

inline void copy_access_chain_pointers(std::vector<uint32_t>& words)
{
    std::vector<uint32_t> copied(words.begin(), words.begin() + 5);
    std::map<uint32_t, uint32_t> pointers;
    for (size_t offset = 5; offset < words.size(); offset += words[offset] >> spv::WordCountShift)
    {
        const uint32_t count = words[offset] >> spv::WordCountShift;
        const spv::Op opcode = static_cast<spv::Op>(words[offset] & spv::OpCodeMask);
        std::vector<uint32_t> instruction(words.begin() + offset, words.begin() + offset + count);
        // Replace load/store operands; leave each original access chain as
        // the source of an explicit pointer copy.
        if (opcode == spv::OpLoad || opcode == spv::OpStore)
        {
            const size_t operand = opcode == spv::OpLoad ? 3 : 1;
            const auto found = pointers.find(instruction[operand]);
            if (found != pointers.end())
                instruction[operand] = found->second;
        }
        copied.insert(copied.end(), instruction.begin(), instruction.end());
        if (opcode == spv::OpAccessChain || opcode == spv::OpInBoundsAccessChain)
        {
            const uint32_t id = copied[3]++;
            pointers[instruction[2]] = id;
            const uint32_t copy[] = {(4u << spv::WordCountShift) | spv::OpCopyObject, instruction[1], id, instruction[2]};
            copied.insert(copied.end(), copy, copy + 4);
        }
    }
    words.swap(copied);
}

inline void make_stores_volatile(std::vector<uint32_t>& words)
{
    std::vector<uint32_t> rewritten(words.begin(), words.begin() + 5);
    for (size_t offset = 5; offset < words.size(); offset += words[offset] >> spv::WordCountShift)
    {
        const uint32_t count = words[offset] >> spv::WordCountShift;
        const spv::Op opcode = static_cast<spv::Op>(words[offset] & spv::OpCodeMask);
        if (opcode == spv::OpStore && count == 3)
        {
            const uint32_t store[] = {(4u << spv::WordCountShift) | spv::OpStore,
                                      words[offset + 1], words[offset + 2], spv::MemoryAccessVolatileMask};
            rewritten.insert(rewritten.end(), store, store + 4);
        }
        else
            rewritten.insert(rewritten.end(), words.begin() + offset, words.begin() + offset + count);
    }
    words.swap(rewritten);
}

inline bool set_fp16_rounding(std::vector<uint32_t>& words, spv::FPRoundingMode rounding)
{
    std::map<uint32_t, bool> half_types;
    std::map<uint32_t, bool> conversions;
    size_t first_type = words.size();
    for (size_t i = 5; i < words.size(); i += words[i] >> 16)
    {
        const uint32_t count = words[i] >> 16;
        const spv::Op opcode = static_cast<spv::Op>(words[i] & 65535u);
        if (count == 0 || count > words.size() - i)
            return false;
        if (opcode >= spv::OpTypeVoid && opcode <= spv::OpTypeForwardPointer && first_type == words.size())
            first_type = i;
        if (opcode == spv::OpTypeFloat && count == 3 && words[i + 2] == 16)
            half_types[words[i + 1]] = true;
        if (opcode == spv::OpTypeVector && count == 4 && half_types[words[i + 2]])
            half_types[words[i + 1]] = true;
        if (opcode == spv::OpFConvert && count == 4 && half_types[words[i + 1]])
            conversions[words[i + 2]] = false;
    }
    if (conversions.empty() || first_type == words.size())
        return false;
    for (size_t i = 5; i < first_type; i += words[i] >> 16)
    {
        if ((words[i] & 65535u) == spv::OpDecorate && (words[i] >> 16) == 4
            && words[i + 2] == spv::DecorationFPRoundingMode && conversions.count(words[i + 1]))
        {
            words[i + 3] = rounding;
            conversions[words[i + 1]] = true;
        }
    }
    std::vector<uint32_t> decorations;
    for (const auto& conversion : conversions)
    {
        if (conversion.second)
            continue;
        decorations.push_back((4u << 16) | spv::OpDecorate);
        decorations.push_back(conversion.first);
        decorations.push_back(spv::DecorationFPRoundingMode);
        decorations.push_back(rounding);
    }
    words.insert(words.begin() + first_type, decorations.begin(), decorations.end());
    return true;
}

#endif // MOMOTEN_SPIRV_TEST_HELPERS_H
