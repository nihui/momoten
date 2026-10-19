// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "translator_internal.h"

namespace momoten {

void CompilerOpenCL::emit_constructor_macros(const char* base)
{
    static const char* suffixes[] = {"", "2", "3", "4", "8", "16"};
    for (const char* suffix : suffixes)
    {
        const std::string type = std::string(base) + suffix;
        // Apple exposes vector types as macros, e.g. float4 -> __float4.
        // Preserve their underlying types before replacing the macros with
        // GLSL-style constructors, or ordinary type declarations stop working.
        statement("#ifdef ", type);
        statement("typedef ", type, " momo_native_", type, ";");
        statement("#undef ", type);
        statement("typedef momo_native_", type, " ", type, ";");
        statement("#endif");
        statement("#define ", type, "(...) ((", type, ")(__VA_ARGS__))");
    }
}

void CompilerOpenCL::emit_fp64_vector_helpers()
{
    // OpenCL C 1.0 does not permit static function definitions, and plain
    // C99 inline linkage can leave an unresolved symbol on some drivers.
    static const char components[] = "xyzw";
    static const char* operation_names[] = {"add", "sub", "mul", "div"};
    static const char operation_tokens[] = "+-*/";

    for (uint32_t width = 2; width <= 4; width++)
    {
        const std::string suffix = std::to_string(width);
        std::string parameters;
        std::string splat_arguments;
        for (uint32_t component = 0; component < width; component++)
        {
            if (component != 0)
            {
                parameters += ", ";
                splat_arguments += ", ";
            }
            parameters += std::string("double ") + components[component];
            splat_arguments += "value";
        }

        statement("double", suffix, " momo_make_double", suffix, "(", parameters, ")");
        begin_scope();
        statement("double", suffix, " result;");
        for (uint32_t component = 0; component < width; component++)
            statement("result.", components[component], " = ", components[component], ";");
        statement("return result;");
        end_scope();

        statement("double", suffix, " momo_splat_double", suffix, "(double value)");
        begin_scope();
        statement("return momo_make_double", suffix, "(", splat_arguments, ");");
        end_scope();

        for (uint32_t operation = 0; operation < 4; operation++)
        {
            statement("double", suffix, " momo_fp64_", operation_names[operation], suffix,
                      "(double", suffix, " a, double", suffix, " b)");
            begin_scope();
            statement("double", suffix, " result;");
            for (uint32_t component = 0; component < width; component++)
                statement("result.", components[component], " = a.", components[component], " ",
                          operation_tokens[operation], " b.", components[component], ";");
            statement("return result;");
            end_scope();
        }

        statement("double", suffix, " momo_fp64_neg", suffix, "(double", suffix, " value)");
        begin_scope();
        statement("double", suffix, " result;");
        for (uint32_t component = 0; component < width; component++)
            statement("result.", components[component], " = -value.", components[component], ";");
        statement("return result;");
        end_scope();

        statement("double", suffix, " momo_fp64_scale", suffix,
                  "(double", suffix, " value, double scale)");
        begin_scope();
        statement("double", suffix, " result;");
        for (uint32_t component = 0; component < width; component++)
            statement("result.", components[component], " = value.", components[component], " * scale;");
        statement("return result;");
        end_scope();

        statement("double momo_fp64_dot", suffix,
                  "(double", suffix, " a, double", suffix, " b)");
        begin_scope();
        std::string dot_expression;
        for (uint32_t component = 0; component < width; component++)
        {
            if (component != 0)
                dot_expression += " + ";
            dot_expression += std::string("a.") + components[component] + " * b." + components[component];
        }
        statement("return ", dot_expression, ";");
        end_scope();
    }
}

} // namespace momoten
