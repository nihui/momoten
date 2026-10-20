// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "translator/opencl_source.h"

#include <stdio.h>
#include <string>

int main()
{
    const struct TestCase
    {
        const char* name;
        const char* input;
        const char* expected;
    } cases[] = {
        {"scalar conversion", "short(value)", "((short)(value))"},
        {"boolean conversion", "bool(value)", "((bool)(value))"},
        {"nested constructors and swizzle",
         "float4(float2(x, y), float2(z, w)).xy",
         "((float4)(((float2)(x, y)), ((float2)(z, w)))).xy"},
        {"ordinary call and grouping",
         "select(float3((x + y)), float3(f(z)), mask)",
         "select(((float3)((x + y))), ((float3)(f(z))), mask)"},
        {"wide vectors", "uint16(uint8(a, b), uint8(c, d))",
         "((uint16)(((uint8)(a, b)), ((uint8)(c, d))))"},
        {"matrix aggregate",
         "momo_mat4(float4(1.0f), c1, c2, c3).c[0]",
         "((momo_mat4){{((float4)(1.0f)), c1, c2, c3}}).c[0]"},
        {"half matrix aggregate",
         "momo_hmat4(a, b, c, d)", "((momo_hmat4){{a, b, c, d}})"},
        {"whitespace", "double2 \n\t(x, y)", "((double2)(x, y))"},
        {"comments inside expressions",
         "int2(x /* float4(ignored) ) */, y // int(ignored)\n)",
         "((int2)(x /* float4(ignored) ) */, y // int(ignored)\n))"},
        {"quoted text and escaped quotes",
         R"source(log("float4(x) \" )", '(', ')'); int(x))source",
         R"source(log("float4(x) \" )", '(', ')'); ((int)(x)))source"},
        {"type declarations, native casts and identifier boundaries",
         "float4 value = (float4)(x); float4_helper(value); my_float4(x);",
         "float4 value = (float4)(x); float4_helper(value); my_float4(x);"},
        {"trailing comment", "int(x); // float4(y)",
         "((int)(x)); // float4(y)"}};

    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++)
    {
        std::string source = cases[i].input;
        momoten::lower_opencl_constructors(source);
        if (source != cases[i].expected)
        {
            fprintf(stderr, "opencl_source_test: %s\nactual: %s\nexpected: %s\n",
                    cases[i].name, source.c_str(), cases[i].expected);
            return 1;
        }
        const std::string lowered = source;
        momoten::lower_opencl_constructors(source);
        if (source != lowered)
        {
            fprintf(stderr, "opencl_source_test: repeated lowering changed %s\n", cases[i].name);
            return 1;
        }
    }
    return 0;
}
