// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#ifndef MOMOTEN_SPIRV_TEST_HELPERS_H
#define MOMOTEN_SPIRV_TEST_HELPERS_H

#include <cstdint>
#include <fstream>
#include <map>
#include <vector>

static bool read_spirv(const char* path, std::vector<uint32_t>& words)
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

static bool decorate_aliased_buffers(std::vector<uint32_t>& words, uint32_t bindings)
{
    // OpDecorate Binding identifies the views independently of IDs/names.
    std::vector<uint32_t> decorations;
    size_t insertion = words.size();
    for (size_t offset = 5; offset < words.size(); offset += words[offset] >> 16)
    {
        const uint32_t count = words[offset] >> 16;
        const uint32_t opcode = words[offset] & 0xffffu;
        if (!count || count > words.size() - offset)
            return false;
        if (opcode == 71 && count == 4 && words[offset + 2] == 33 && words[offset + 3] < bindings)
        {
            decorations.push_back((3u << 16) | 71u); // OpDecorate
            decorations.push_back(words[offset + 1]);
            decorations.push_back(20u); // Aliased
        }
        if (opcode >= 19 && opcode <= 39 && insertion == words.size())
            insertion = offset;
    }
    if (decorations.size() != bindings * 3 || insertion == words.size())
        return false;
    words.insert(words.begin() + insertion, decorations.begin(), decorations.end());
    return true;
}

static void copy_access_chain_pointers(std::vector<uint32_t>& words)
{
    std::vector<uint32_t> copied(words.begin(), words.begin() + 5);
    std::map<uint32_t, uint32_t> pointers;
    for (size_t offset = 5; offset < words.size(); offset += words[offset] >> 16)
    {
        const uint32_t count = words[offset] >> 16;
        const uint32_t opcode = words[offset] & 0xffffu;
        std::vector<uint32_t> instruction(words.begin() + offset, words.begin() + offset + count);
        // Replace load/store operands; leave each original access chain as
        // the source of an explicit pointer copy.
        if (opcode == 61 || opcode == 62) // OpLoad / OpStore
        {
            const size_t operand = opcode == 61 ? 3 : 1;
            const auto found = pointers.find(instruction[operand]);
            if (found != pointers.end())
                instruction[operand] = found->second;
        }
        copied.insert(copied.end(), instruction.begin(), instruction.end());
        if (opcode == 65 || opcode == 66) // OpAccessChain / OpInBoundsAccessChain
        {
            const uint32_t id = copied[3]++;
            pointers[instruction[2]] = id;
            const uint32_t copy[] = {(4u << 16) | 83u, instruction[1], id, instruction[2]}; // OpCopyObject
            copied.insert(copied.end(), copy, copy + 4);
        }
    }
    words.swap(copied);
}

#endif // MOMOTEN_SPIRV_TEST_HELPERS_H
