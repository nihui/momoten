# Contributing

Edit only the modular sources under `src/`, public headers under `include/`,
and their tests. The single-file distribution is generated in the build tree.
Use the checked-in `.clang-format`, which deliberately sets `ColumnLimit: 0`;
do not manually wrap code to a fixed display width.

Before submitting a change:

```bash
git submodule update --init --recursive
cmake -S . -B build -DMOMOTEN_BUILD_AMALGAMATION=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

Generic code must not include ncnn headers, enumerate ncnn shaders, or contain
layer-specific behavior. Consumer-specific validation belongs under
`integrations/`. Unsupported Vulkan and SPIR-V behavior must fail explicitly
rather than silently changing shader semantics.

New project-owned source files need the `Copyright 2026 nihui` notice and
`SPDX-License-Identifier: Apache-2.0` header. Keep upstream dependency notices
and licenses unchanged.
