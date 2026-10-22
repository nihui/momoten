// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#ifndef MOMOTEN_INTEGER_DOT_PRODUCT_FIXTURE_H
#define MOMOTEN_INTEGER_DOT_PRODUCT_FIXTURE_H

#include <sstream>
#include <string>

inline std::string integer_dot_product_fixture(bool packed)
{
    std::ostringstream s;
    s << "OpCapability Shader\n"
      << "OpCapability DotProductKHR\n"
      << (packed ? "OpCapability DotProductInput4x8BitPackedKHR\n" : "OpCapability DotProductInput4x8BitKHR\n")
      << "OpExtension \"SPV_KHR_integer_dot_product\"\n"
      << "OpMemoryModel Logical GLSL450\n"
      << "OpEntryPoint GLCompute %main \"main\" %gid\n"
      << "OpExecutionMode %main LocalSize 1 1 1\n"
      << "OpDecorate %gid BuiltIn GlobalInvocationId\n"
      << "OpDecorate %array ArrayStride 4\n"
      << "OpMemberDecorate %block 0 Offset 0\n"
      << "OpDecorate %block Block\n"
      << "OpDecorate %input DescriptorSet 0\n"
      << "OpDecorate %input Binding 0\n"
      << "OpDecorate %input NonWritable\n"
      << "OpDecorate %output DescriptorSet 0\n"
      << "OpDecorate %output Binding 1\n"
      << "%void = OpTypeVoid\n"
      << "%fn = OpTypeFunction %void\n"
      << "%uint = OpTypeInt 32 0\n"
      << "%int = OpTypeInt 32 1\n"
      << "%v3uint = OpTypeVector %uint 3\n"
      << "%gidptr = OpTypePointer Input %v3uint\n"
      << "%gid = OpVariable %gidptr Input\n"
      << "%array = OpTypeRuntimeArray %uint\n"
      << "%block = OpTypeStruct %array\n"
      << "%blockptr = OpTypePointer StorageBuffer %block\n"
      << "%intptr = OpTypePointer StorageBuffer %uint\n"
      << "%input = OpVariable %blockptr StorageBuffer\n"
      << "%output = OpVariable %blockptr StorageBuffer\n";
    if (!packed)
        s << "%char = OpTypeInt 8 1\n"
          << "%uchar = OpTypeInt 8 0\n"
          << "%v4char = OpTypeVector %char 4\n"
          << "%v4uchar = OpTypeVector %uchar 4\n";
    for (unsigned i = 0; i <= 9; i++)
        s << "%c" << i << " = OpConstant %uint " << i << "\n";
    s << "%main = OpFunction %void None %fn\n"
      << "%entry = OpLabel\n"
      << "%position = OpLoad %v3uint %gid\n"
      << "%i = OpCompositeExtract %uint %position 0\n"
      << "%inbase = OpIMul %uint %i %c4\n"
      << "%outbase = OpIMul %uint %i %c9\n";
    for (unsigned i = 0; i < 3; i++)
        s << "%index" << i << " = OpIAdd %uint %inbase %c" << i << "\n"
          << "%ptr" << i << " = OpAccessChain %intptr %input %c0 %index" << i << "\n"
          << "%value" << i << " = OpLoad %uint %ptr" << i << "\n";
    s << "%acc = OpBitcast %int %value2\n";
    if (!packed)
        s << "%sa = OpBitcast %v4char %value0\n"
          << "%sb = OpBitcast %v4char %value1\n"
          << "%ua = OpBitcast %v4uchar %value0\n"
          << "%ub = OpBitcast %v4uchar %value1\n";
    const char* vector_a[] = {"%sa", "%ua", "%sa", "%sb"};
    const char* vector_b[] = {"%sb", "%ub", "%ub", "%ua"};
    for (unsigned i = 0; i < 8; i++)
    {
        const unsigned sign = i % 4;
        const char* opcode = sign == 0 ? "OpSDot" : sign == 1 ? "OpUDot"
                                                              : "OpSUDot";
        const char* a = packed ? (sign == 3 ? "%value1" : "%value0") : vector_a[sign];
        const char* b = packed ? (sign == 3 ? "%value0" : "%value1") : vector_b[sign];
        // Test signed operations with unsigned result bit patterns as well as
        // their usual signed results; the opcode determines interpretation.
        const bool unsigned_result = sign <= 1;
        s << "%dot" << i << " = " << opcode << (i >= 4 ? "AccSat" : "") << "KHR "
          << (unsigned_result ? "%uint " : "%int ") << a << " " << b;
        if (i >= 4)
            s << (unsigned_result ? " %value2" : " %acc");
        if (packed)
            s << " PackedVectorFormat4x8BitKHR";
        s << "\n";
        if (!unsigned_result)
            s << "%bits" << i << " = OpBitcast %uint %dot" << i << "\n";
        s << "%outindex" << i << " = OpIAdd %uint %outbase %c" << i << "\n"
          << "%outptr" << i << " = OpAccessChain %intptr %output %c0 %outindex" << i << "\n"
          << "OpStore %outptr" << i << " " << (unsigned_result ? "%dot" : "%bits") << i << "\n";
    }
    s << "%ordinary = OpIAdd %uint %dot0 %value2\n"
      << "%outindex8 = OpIAdd %uint %outbase %c8\n"
      << "%outptr8 = OpAccessChain %intptr %output %c0 %outindex8\n"
      << "OpStore %outptr8 %ordinary\n"
      << "OpReturn\n"
      << "OpFunctionEnd\n";
    return s.str();
}

#endif // MOMOTEN_INTEGER_DOT_PRODUCT_FIXTURE_H
