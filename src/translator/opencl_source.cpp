// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "opencl_source.h"

#include <cctype>
#include <cstring>
#include <vector>

namespace momoten {

enum class ConstructorKind
{
    None,
    ScalarOrVector,
    Matrix
};

static ConstructorKind opencl_constructor_kind(const std::string& name)
{
    if (name == "momo_mat4" || name == "momo_hmat4")
        return ConstructorKind::Matrix;
    if (name == "bool")
        return ConstructorKind::ScalarOrVector;
    static const char* bases[] = {"char", "uchar", "short", "ushort", "int", "uint", "long", "ulong", "half", "float", "double"};
    for (size_t i = 0; i < sizeof(bases) / sizeof(bases[0]); i++)
    {
        const size_t length = strlen(bases[i]);
        if (name.compare(0, length, bases[i]) != 0)
            continue;
        const std::string width = name.substr(length);
        if (width.empty() || width == "2" || width == "3" || width == "4" || width == "8" || width == "16")
            return ConstructorKind::ScalarOrVector;
    }
    return ConstructorKind::None;
}

void lower_opencl_constructors(std::string& source)
{
    // SPIRV-Cross's constructor emitter is not virtual. Convert its GLSL
    // syntax here instead of defining variadic macros, which OpenCL C forbids.
    // Wrap the complete expression so following swizzles retain precedence.
    std::string result;
    result.reserve(source.size());
    std::vector<ConstructorKind> parentheses;
    for (size_t i = 0; i < source.size();)
    {
        const size_t begin = i;
        if (source.compare(i, 2, "//") == 0 || source.compare(i, 2, "/*") == 0)
        {
            const bool block = source[i + 1] == '*';
            const size_t end = source.find(block ? "*/" : "\n", i + 2);
            i = end == std::string::npos ? source.size() : end + (block ? 2 : 1);
            result.append(source, begin, i - begin);
            continue;
        }
        if (source[i] == '"' || source[i] == '\'')
        {
            const char quote = source[i++];
            while (i < source.size())
            {
                const char c = source[i++];
                if (c == '\\' && i < source.size())
                    i++;
                else if (c == quote)
                    break;
            }
            result.append(source, begin, i - begin);
            continue;
        }
        if (std::isalpha(static_cast<unsigned char>(source[i])) || source[i] == '_')
        {
            do
            {
                i++;
            } while (i < source.size() && (std::isalnum(static_cast<unsigned char>(source[i])) || source[i] == '_'));
            const std::string name = source.substr(begin, i - begin);
            const ConstructorKind kind = opencl_constructor_kind(name);
            size_t next = i;
            while (next < source.size() && std::isspace(static_cast<unsigned char>(source[next])))
                next++;
            if (kind != ConstructorKind::None && next < source.size() && source[next] == '(')
            {
                result += "((" + name + (kind == ConstructorKind::Matrix ? "){{" : ")(");
                parentheses.push_back(kind);
                i = next + 1;
            }
            else
                result += name;
            continue;
        }
        if (source[i] == '(')
            parentheses.push_back(ConstructorKind::None);
        else if (source[i] == ')' && !parentheses.empty())
        {
            const ConstructorKind kind = parentheses.back();
            parentheses.pop_back();
            if (kind != ConstructorKind::None)
            {
                result += kind == ConstructorKind::Matrix ? "}})" : "))";
                i++;
                continue;
            }
        }
        result += source[i++];
    }
    source.swap(result);
}

} // namespace momoten
