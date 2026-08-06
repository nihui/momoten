// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#ifndef MOMOTEN_TRANSLATOR_INTERNAL_H
#define MOMOTEN_TRANSLATOR_INTERNAL_H

#include "momoten/spv_to_clc.h"

#include <map>
#include <set>
#include <string>
#include <vector>

#include "spirv_glsl.hpp"

namespace momoten {

using namespace spirv_cross;
using namespace spv;

struct ResourceInfo
{
    uint32_t variable_id;
    uint32_t descriptor_set;
    uint32_t binding;
    BufferAccess access;
    std::string instance_name;
};

bool resource_less(const ResourceInfo& a, const ResourceInfo& b);

std::vector<uint32_t> specialize_and_optimize_spirv(
    const uint32_t* words, size_t word_count, const TranslationOptions& options);

bool spirv_contains_opcode(const std::vector<uint32_t>& words, Op opcode);

const char* capability_name(Capability capability);

class CompilerOpenCL : public CompilerGLSL
{
public:
    CompilerOpenCL(const uint32_t* words, size_t word_count, const TranslationOptions& options,
                   bool requires_global_int32_atomics, bool contains_synchronization_barrier);

    void prepare(KernelABI& abi);
    bool needs_converged_returns() const;

protected:
    void emit_entry_point_declarations() override;
    void emit_header() override;
    void emit_buffer_block(const SPIRVariable& var) override;
    void emit_push_constant_block(const SPIRVariable& var) override;
    void emit_uniform(const SPIRVariable& var) override;
    void emit_function_prototype(SPIRFunction& func, const Bitset& return_flags) override;
    std::string to_func_call_arg(const SPIRFunction::Parameter& argument, uint32_t id) override;
    void append_global_func_args(const SPIRFunction& func, uint32_t index,
                                 SmallVector<std::string>& arglist) override;
    std::string type_to_glsl(const SPIRType& type, uint32_t id = 0) override;
    std::string layout_for_member(const SPIRType& type, uint32_t index) override;
    const char* to_storage_qualifiers_glsl(const SPIRVariable& var) override;
    std::string to_qualifiers_glsl(uint32_t id) override;
    std::string bitcast_glsl_op(const SPIRType& result_type, const SPIRType& argument_type) override;
    std::string builtin_to_glsl(BuiltIn builtin, StorageClass storage) override;
    std::string to_member_reference(uint32_t base, const SPIRType& type, uint32_t index,
                                    bool ptr_chain_is_resolved) override;
    void emit_instruction(const Instruction& instruction) override;
    void emit_store_statement(uint32_t lhs_expression, uint32_t rhs_expression) override;
    void emit_glsl_op(uint32_t result_type, uint32_t result_id, uint32_t op,
                      const uint32_t* args, uint32_t count) override;
    std::string constant_op_expression(const SPIRConstantOp& operation) override;
    std::string constant_expression_vector(const SPIRConstant& constant, uint32_t vector) override;

private:
    bool emit_robust_buffer_load(const Instruction& instruction, const uint32_t* ops);
    void emit_fp64_vector_helpers();
    bool emit_fp64_vector_instruction(const Instruction& instruction, const uint32_t* ops);
    void emit_integer_dot_product_helpers();
    bool emit_integer_dot_product_instruction(const Instruction& instruction, const uint32_t* ops);
    bool emit_basic_subgroup_instruction(const Instruction& instruction, const uint32_t* ops);
    bool emit_subgroup_barrier_instruction(const Instruction& instruction, const uint32_t* ops);
    bool access_chain_contains_matrix(uint32_t base, const uint32_t* indices, uint32_t count);
    bool emit_matrix_access_chain(const Instruction& instruction, const uint32_t* ops);
    static const char* function_pointer_address_space(StorageClass storage);
    void emit_constructor_macros(const char* base);
    void emit_struct_typedef(const SPIRType& type);

    void validate_capabilities();
    void apply_specializations();
    void localize_workgroup_variables();
    void validate_barrier_instruction(Op opcode, const uint32_t* ops);
    void override_workgroup_size();

    void validate_storage_block(const Resource& resource);
    void validate_push_constant_block(const Resource& resource, size_t& size);
    void reflect_resources(KernelABI& abi);

    TranslationOptions translation_options;
    bool requires_global_int32_atomics;
    bool requires_fp16;
    bool requires_fp64;
    bool requires_integer_dot_product;
    bool uses_workgroup_storage;
    bool workgroup_splittable;
    std::vector<ResourceInfo> resources;
    std::map<uint32_t, size_t> resource_indices;
    uint32_t push_constant_variable_id;
    uint32_t local_size[3];
    std::set<uint32_t> emitted_struct_typedefs;
    std::set<uint32_t> storage_block_types;
};

} // namespace momoten

#endif // MOMOTEN_TRANSLATOR_INTERNAL_H
