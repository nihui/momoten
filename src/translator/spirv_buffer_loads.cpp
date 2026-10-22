// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "translator_internal.h"

#include <algorithm>

#include "spirv-tools/optimizer.hpp"

namespace momoten {
namespace {

struct BufferLoadInstruction
{
    size_t offset;
    uint32_t count;
    Op opcode;
    uint32_t type;
    uint32_t result;
};

struct BufferLoadIndex
{
    uint32_t base;
    uint32_t type;
    int64_t offset;
};

struct BufferLoadCandidate
{
    uint32_t result;
    uint32_t buffer;
    uint32_t type;
    uint32_t bytes;
    size_t position;
    BufferLoadIndex index;
};

// Analyze an SSA copy, leaving the module used for code generation untouched.
// This exposes equal indices and loop-carried alignment without depending on
// source names, model shapes, or a particular compiler's private temporaries.
class BufferLoadPlanner : public Compiler
{
public:
    BufferLoadPlanner(const std::vector<uint32_t>& words_, const TranslationOptions& options_)
        : Compiler(words_.data(), words_.size()), words(words_), options(options_)
    {
    }

    std::map<uint32_t, CoalescedBufferLoad> run();

private:
    const uint32_t* operands(const BufferLoadInstruction& instruction) const
    {
        return words.data() + instruction.offset + 1;
    }

    const BufferLoadInstruction* definition(uint32_t id) const
    {
        const auto it = definitions.find(id);
        return it == definitions.end() ? 0 : &instructions[it->second];
    }

    bool integer32(uint32_t type) const
    {
        if (!type)
            return false;
        const SPIRType& t = get_type(type);
        return !t.pointer && t.array.empty() && t.columns == 1 && t.vecsize == 1
               && t.width == 32 && (t.basetype == SPIRType::Int || t.basetype == SPIRType::UInt);
    }

    bool constant(uint32_t id, int64_t& value) const;
    unsigned low_zero_bits(uint32_t id) const;
    unsigned instruction_zero_bits(const BufferLoadInstruction& instruction) const;
    BufferLoadIndex index(uint32_t id, unsigned depth = 0) const;
    uint32_t pointer_root(uint32_t id) const;
    bool ordered_load(const BufferLoadInstruction& instruction) const;
    bool candidate(const BufferLoadInstruction& instruction, size_t position, BufferLoadCandidate& load) const;
    bool invalidates_loads(const BufferLoadInstruction& instruction) const;
    void collect(const std::vector<BufferLoadCandidate>& candidates);

    const std::vector<uint32_t>& words;
    const TranslationOptions& options;
    std::vector<BufferLoadInstruction> instructions;
    std::map<uint32_t, size_t> definitions;
    std::vector<std::vector<size_t> > blocks;
    std::map<uint32_t, unsigned> zero_bits;
    std::map<uint32_t, CoalescedBufferLoad> groups;
};

bool BufferLoadPlanner::constant(uint32_t id, int64_t& value) const
{
    const BufferLoadInstruction* d = definition(id);
    if (!d || !integer32(d->type))
        return false;
    if (d->opcode == OpConstantNull)
    {
        value = 0;
        return true;
    }
    if (d->opcode != OpConstant || d->count != 4)
        return false;
    const uint32_t bits = operands(*d)[2];
    value = get_type(d->type).basetype == SPIRType::Int ? static_cast<int64_t>(static_cast<int32_t>(bits)) : static_cast<int64_t>(bits);
    return true;
}

unsigned BufferLoadPlanner::low_zero_bits(uint32_t id) const
{
    const auto it = zero_bits.find(id);
    return it == zero_bits.end() ? 0 : it->second;
}

unsigned BufferLoadPlanner::instruction_zero_bits(const BufferLoadInstruction& d) const
{
    if (!integer32(d.type))
        return 0;
    const uint32_t* ops = operands(d);
    int64_t value = 0;
    if (constant(d.result, value))
    {
        uint32_t bits = static_cast<uint32_t>(value);
        unsigned count = 0;
        while (count < 32 && (bits & 1) == 0)
        {
            count++;
            bits >>= 1;
        }
        return count;
    }
    switch (d.opcode)
    {
    case OpCopyObject:
    case OpBitcast:
        return low_zero_bits(ops[2]);
    case OpIAdd:
    case OpISub:
    case OpBitwiseOr:
    case OpBitwiseXor:
        return std::min(low_zero_bits(ops[2]), low_zero_bits(ops[3]));
    case OpIMul:
        return std::min(32u, low_zero_bits(ops[2]) + low_zero_bits(ops[3]));
    case OpBitwiseAnd:
        return std::max(low_zero_bits(ops[2]), low_zero_bits(ops[3]));
    case OpShiftLeftLogical:
        if (constant(ops[3], value) && value >= 0 && value < 32)
            return std::min(32u, low_zero_bits(ops[2]) + static_cast<unsigned>(value));
        return 0;
    case OpSelect:
        return std::min(low_zero_bits(ops[3]), low_zero_bits(ops[4]));
    case OpPhi:
    {
        unsigned count = 32;
        for (uint32_t i = 2; i + 1 < d.count - 1; i += 2)
            count = std::min(count, low_zero_bits(ops[i]));
        return count;
    }
    default:
        return 0;
    }
}

BufferLoadIndex BufferLoadPlanner::index(uint32_t id, unsigned depth) const
{
    const BufferLoadInstruction* d = definition(id);
    BufferLoadIndex result = {id, d ? d->type : 0, 0};
    if (!d || !integer32(d->type) || depth >= 32)
        return result;
    int64_t value = 0;
    if (constant(id, value))
    {
        result.base = 0;
        result.offset = value;
        return result;
    }
    const uint32_t* ops = operands(*d);
    if (d->opcode == OpCopyObject)
        return index(ops[2], depth + 1);
    if (d->opcode != OpIAdd && d->opcode != OpISub)
        return result;

    uint32_t base = ops[2];
    if (!constant(ops[3], value))
    {
        if (d->opcode != OpIAdd || !constant(ops[2], value))
            return result;
        base = ops[3];
    }
    else if (d->opcode == OpISub)
        value = -value;
    BufferLoadIndex before = index(base, depth + 1);
    if (before.type != d->type)
        return result;
    const uint32_t offset = static_cast<uint32_t>(before.offset + value);
    if (before.base == 0)
    {
        before.offset = get_type(d->type).basetype == SPIRType::Int ? static_cast<int64_t>(static_cast<int32_t>(offset)) : static_cast<int64_t>(offset);
        return before;
    }
    // Keep modular offsets here. collect() proves that no wrap boundary lies
    // *inside* the selected span. Wrapping before its first load is harmless.
    before.offset = offset;
    return before;
}

uint32_t BufferLoadPlanner::pointer_root(uint32_t id) const
{
    for (unsigned depth = 0; depth < 32; depth++)
    {
        const BufferLoadInstruction* d = definition(id);
        if (!d)
            return 0;
        if (d->opcode == OpVariable)
            return id;
        if (d->opcode != OpAccessChain && d->opcode != OpInBoundsAccessChain && d->opcode != OpCopyObject)
            return 0;
        id = operands(*d)[2];
    }
    return 0;
}

bool BufferLoadPlanner::ordered_load(const BufferLoadInstruction& d) const
{
    const uint32_t* ops = operands(d);
    if (d.count > 4 && ops[3] != MemoryAccessMaskNone && ops[3] != MemoryAccessAlignedMask)
        return true;
    const uint32_t root = pointer_root(ops[2]);
    const SPIRVariable* variable = maybe_get<SPIRVariable>(root);
    if (!variable)
        return true;
    if (has_decoration(root, DecorationVolatile) || has_decoration(root, DecorationCoherent)
        || has_decoration(ops[2], DecorationVolatile) || has_decoration(ops[2], DecorationCoherent))
        return true;
    if (variable->storage == StorageClassStorageBuffer || variable->storage == StorageClassUniform)
    {
        const Bitset flags = get_buffer_block_flags(root);
        if (flags.get(DecorationVolatile) || flags.get(DecorationCoherent))
            return true;
    }
    return false;
}

bool BufferLoadPlanner::candidate(const BufferLoadInstruction& d, size_t position, BufferLoadCandidate& load) const
{
    if (d.opcode != OpLoad || d.count < 4 || ordered_load(d))
        return false;
    const uint32_t* ops = operands(d);
    const SPIRType& type = get_type(d.type);
    const bool word_type = type.width == 32 && (type.basetype == SPIRType::Float || type.basetype == SPIRType::Int || type.basetype == SPIRType::UInt)
                           && (type.vecsize == 1 || type.vecsize == 2);
    // Reuse the existing device probe for the implementation-defined mapping
    // from uint2 to half4. No unprobed cross-width bitcast is introduced.
    const bool packed_half = options.packed_fp16_vector_loads && type.width == 16 && type.basetype == SPIRType::Half && type.vecsize == 4;
    if ((!word_type && !packed_half) || type.pointer || type.columns != 1 || !type.array.empty())
        return false;

    const BufferLoadInstruction* chain = definition(ops[2]);
    if (!chain || (chain->opcode != OpAccessChain && chain->opcode != OpInBoundsAccessChain) || chain->count != 6)
        return false;
    const uint32_t* access = operands(*chain);
    const SPIRVariable* variable = maybe_get<SPIRVariable>(access[2]);
    if (!variable || (variable->storage != StorageClassStorageBuffer && variable->storage != StorageClassUniform))
        return false;
    const SPIRType& block = get_variable_data_type(*variable);
    int64_t member = -1;
    if (block.basetype != SPIRType::Struct || block.member_types.size() != 1 || !constant(access[3], member) || member != 0)
        return false;
    const SPIRType& array = get_type(block.member_types[0]);
    const uint32_t bytes = type.width * type.vecsize / 8;
    if (array.op != OpTypeRuntimeArray || array.array.size() != 1 || array.columns != 1
        || array.basetype != type.basetype || array.width != type.width || array.vecsize != type.vecsize
        || get_decoration(block.member_types[0], DecorationArrayStride) != bytes
        || !has_member_decoration(block.self, 0, DecorationOffset) || get_member_decoration(block.self, 0, DecorationOffset) != 0)
        return false;
    const BufferLoadIndex address = index(access[4]);
    if (!integer32(address.type) || (address.base == 0 && address.offset < 0))
        return false;
    load = {d.result, access[2], d.type, bytes, position, address};
    return true;
}

bool BufferLoadPlanner::invalidates_loads(const BufferLoadInstruction& d) const
{
    const uint32_t* ops = operands(d);
    if (d.opcode == OpLoad)
        return ordered_load(d);
    if (d.opcode == OpStore)
    {
        const BufferLoadInstruction* pointer = definition(ops[0]);
        if (!pointer || !pointer->type || d.count > 3)
            return true;
        const StorageClass storage = get_type(pointer->type).storage;
        return storage != StorageClassFunction && storage != StorageClassPrivate;
    }
    if (d.opcode == OpFunctionCall || d.opcode == OpExtInst || d.opcode == OpControlBarrier || d.opcode == OpMemoryBarrier
        || (d.opcode >= OpAtomicLoad && d.opcode <= OpAtomicXor)
        || d.opcode == OpAtomicFlagTestAndSet || d.opcode == OpAtomicFlagClear
        || d.opcode == OpAtomicFAddEXT || d.opcode == OpAtomicFMinEXT || d.opcode == OpAtomicFMaxEXT)
        return true;
    if (d.opcode == OpNop || d.opcode == OpLine || d.opcode == OpNoLine)
        return false;
    // Other instructions with no result may have side effects. In particular,
    // writes through another descriptor can alias this buffer at dispatch.
    return d.result == 0;
}

void BufferLoadPlanner::collect(const std::vector<BufferLoadCandidate>& candidates)
{
    for (size_t i = 0; i < candidates.size(); i++)
    {
        const BufferLoadCandidate& first = candidates[i];
        if (groups.count(first.result))
            continue;
        std::vector<size_t> selected(1, i);
        const size_t count = 16 / first.bytes;
        // SPIR-V integer addition wraps. Sharing an affine base is not enough:
        // UINT_MAX and UINT_MAX + 1 are not neighboring buffer addresses.
        // Known zero bits partition the address space into aligned blocks;
        // the whole group must remain inside one such block. Signed indices
        // also have a discontinuity at 2^31. This permits, for example, both
        // [base, base+1] and [base+2, base+3] when base is only two-aligned.
        unsigned bits = first.index.base ? low_zero_bits(first.index.base) : 32;
        if (get_type(first.index.type).basetype == SPIRType::Int)
            bits = std::min(bits, 31u);
        const uint64_t span = uint64_t(1) << bits;
        if ((static_cast<uint64_t>(first.index.offset) & (span - 1)) + count > span)
            continue;
        for (size_t j = i + 1; j < candidates.size() && selected.size() < count; j++)
        {
            const BufferLoadCandidate& next = candidates[j];
            // Bound the scheduling distance and live temporary footprint.
            if (next.position - first.position > 32)
                break;
            if (!groups.count(next.result) && next.buffer == first.buffer && next.type == first.type
                && next.index.type == first.index.type && next.index.base == first.index.base
                && next.index.offset == first.index.offset + static_cast<int64_t>(selected.size()))
                selected.push_back(j);
        }
        if (selected.size() != count)
            continue;
        for (size_t j = 0; j < selected.size(); j++)
            groups[candidates[selected[j]].result] = {first.result, static_cast<uint32_t>(j * first.bytes / 4)};
    }
}

std::map<uint32_t, CoalescedBufferLoad> BufferLoadPlanner::run()
{
    bool in_block = false;
    for (size_t offset = 5; offset < words.size(); offset += words[offset] >> 16)
    {
        BufferLoadInstruction d = {offset, words[offset] >> 16, static_cast<Op>(words[offset] & 0xffffu), 0, 0};
        bool has_result = false;
        bool has_type = false;
        HasResultAndType(d.opcode, &has_result, &has_type);
        if (has_type)
            d.type = words[offset + 1];
        if (has_result)
        {
            d.result = words[offset + (has_type ? 2 : 1)];
            definitions[d.result] = instructions.size();
            zero_bits[d.result] = integer32(d.type) ? 32 : 0;
        }
        if (d.opcode == OpLabel)
        {
            blocks.push_back(std::vector<size_t>());
            in_block = true;
        }
        else if (d.opcode == OpFunctionEnd)
            in_block = false;
        else if (in_block)
            blocks.back().push_back(instructions.size());
        instructions.push_back(d);
    }
    // A descending fixed point handles both SSA phis and aligned induction
    // variables. Unknown values and OpUndef contribute no known zero bits.
    bool changed;
    do
    {
        changed = false;
        for (const BufferLoadInstruction& d : instructions)
        {
            if (!d.result)
                continue;
            const unsigned count = std::min(zero_bits[d.result], instruction_zero_bits(d));
            if (count != zero_bits[d.result])
            {
                zero_bits[d.result] = count;
                changed = true;
            }
        }
    } while (changed);

    for (const auto& block : blocks)
    {
        std::vector<BufferLoadCandidate> candidates;
        for (size_t position = 0; position < block.size(); position++)
        {
            const BufferLoadInstruction& d = instructions[block[position]];
            if (invalidates_loads(d))
            {
                collect(candidates);
                candidates.clear();
            }
            BufferLoadCandidate load;
            if (candidate(d, position, load))
                candidates.push_back(load);
        }
        collect(candidates);
    }
    return groups;
}

} // namespace

std::map<uint32_t, CoalescedBufferLoad> plan_coalesced_buffer_loads(
    const uint32_t* words, size_t word_count, const TranslationOptions& options)
{
    if (!options.native_buffer_vector_loads)
        return {};
    spvtools::Optimizer optimizer(SPV_ENV_VULKAN_1_1);
    optimizer.RegisterPass(spvtools::CreateSSARewritePass());
    std::vector<uint32_t> ssa;
    if (!optimizer.Run(words, word_count, &ssa))
        return {};
    BufferLoadPlanner planner(ssa, options);
    return planner.run();
}

} // namespace momoten
