// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "translator_internal.h"

#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace momoten {

bool CompilerOpenCL::emit_basic_subgroup_instruction(
    const Instruction& instruction, const uint32_t* ops)
{
    const Op opcode = static_cast<Op>(instruction.op);
    if (opcode != OpGroupNonUniformElect)
        return false;

    if (instruction.length != 3)
        throw std::runtime_error("OpGroupNonUniformElect has an invalid operand count");
    if (evaluate_constant_u32(ops[2]) != ScopeSubgroup)
        throw std::runtime_error("OpGroupNonUniformElect requires Subgroup scope");

    const char* expression = translation_options.subgroup_mode == SubgroupModeNative ? "sub_group_elect()" : "true";
    emit_op(ops[0], ops[1], expression, true);
    return true;
}

bool CompilerOpenCL::emit_subgroup_barrier_instruction(
    const Instruction& instruction, const uint32_t* ops)
{
    const Op opcode = static_cast<Op>(instruction.op);
    if (opcode != OpControlBarrier && opcode != OpMemoryBarrier)
        return false;

    const uint32_t execution_scope = opcode == OpControlBarrier ? evaluate_constant_u32(ops[0]) : ScopeSubgroup;
    const uint32_t memory_scope = evaluate_constant_u32(
        opcode == OpControlBarrier ? ops[1] : ops[0]);
    if (execution_scope != ScopeSubgroup && memory_scope != ScopeSubgroup)
        return false;

    if (opcode == OpControlBarrier && execution_scope != ScopeSubgroup)
        throw std::runtime_error("a subgroup memory scope with a wider execution barrier is unsupported");
    if (memory_scope != ScopeSubgroup)
        throw std::runtime_error("a subgroup execution barrier requires Subgroup memory scope");

    validate_barrier_instruction(opcode, ops);
    const uint32_t semantics = evaluate_constant_u32(
        opcode == OpControlBarrier ? ops[2] : ops[1]);

    if (semantics || opcode == OpControlBarrier)
    {
        if (!current_emitting_block)
            throw std::runtime_error("subgroup barrier emitted outside a basic block");
        flush_control_dependent_expressions(current_emitting_block->self);
        flush_all_active_variables();
    }

    std::string flags;
    if (semantics & MemorySemanticsWorkgroupMemoryMask)
        flags = "CLK_LOCAL_MEM_FENCE";
    if (semantics & (MemorySemanticsUniformMemoryMask | MemorySemanticsCrossWorkgroupMemoryMask | MemorySemanticsAtomicCounterMemoryMask))
        flags += flags.empty() ? "CLK_GLOBAL_MEM_FENCE" : " | CLK_GLOBAL_MEM_FENCE";

    if (opcode == OpControlBarrier && translation_options.subgroup_mode == SubgroupModeNative)
    {
        statement("sub_group_barrier(", flags.empty() ? "0" : flags, ");");
        return true;
    }

    // With singleton subgroups there is no peer invocation to wait for. A
    // legacy fence is nevertheless needed to retain the requested ordering
    // for this invocation. OpMemoryBarrier uses the same fence in native mode
    // because it must not introduce the control convergence of a barrier.
    if (!flags.empty())
    {
        if (semantics & MemorySemanticsAcquireReleaseMask || semantics & MemorySemanticsSequentiallyConsistentMask || (!(semantics & MemorySemanticsAcquireMask) && !(semantics & MemorySemanticsReleaseMask)))
            statement("mem_fence(", flags, ");");
        else if (semantics & MemorySemanticsAcquireMask)
            statement("read_mem_fence(", flags, ");");
        else
            statement("write_mem_fence(", flags, ");");
    }
    return true;
}

void CompilerOpenCL::validate_barrier_instruction(Op opcode, const uint32_t* ops)
{
    const uint32_t execution_scope = opcode == OpControlBarrier ? evaluate_constant_u32(ops[0]) : ScopeWorkgroup;
    const uint32_t memory_scope = evaluate_constant_u32(
        opcode == OpControlBarrier ? ops[1] : ops[0]);
    const uint32_t semantics = evaluate_constant_u32(
        opcode == OpControlBarrier ? ops[2] : ops[1]);

    if (opcode == OpControlBarrier && execution_scope != ScopeWorkgroup && execution_scope != ScopeSubgroup)
        throw std::runtime_error("OpenCL barrier execution scope must be Workgroup or Subgroup");
    if (memory_scope != ScopeWorkgroup && memory_scope != ScopeSubgroup)
        throw std::runtime_error("OpenCL barrier memory scope must be Workgroup or Subgroup");

    const uint32_t allowed = MemorySemanticsAcquireMask | MemorySemanticsReleaseMask | MemorySemanticsAcquireReleaseMask | MemorySemanticsSequentiallyConsistentMask | MemorySemanticsUniformMemoryMask | MemorySemanticsWorkgroupMemoryMask | MemorySemanticsCrossWorkgroupMemoryMask | MemorySemanticsSubgroupMemoryMask | MemorySemanticsAtomicCounterMemoryMask | MemorySemanticsImageMemoryMask;
    if ((semantics & ~allowed) != 0)
    {
        std::ostringstream message;
        message << "OpenCL C 1.0 barrier has unsupported memory semantics mask 0x"
                << std::hex << (semantics & ~allowed);
        throw std::runtime_error(message.str());
    }
}

} // namespace momoten
