# Third-party dependencies

momoten vendors the following source dependencies as pinned Git submodules
under `third_party/`:

- [Vulkan-Headers](https://github.com/KhronosGroup/Vulkan-Headers),
  Apache-2.0;
- [OpenCL-Headers](https://github.com/KhronosGroup/OpenCL-Headers),
  Apache-2.0;
- [SPIRV-Headers](https://github.com/KhronosGroup/SPIRV-Headers), MIT;
- [SPIRV-Tools](https://github.com/KhronosGroup/SPIRV-Tools), Apache-2.0;
- [SPIRV-Cross](https://github.com/KhronosGroup/SPIRV-Cross),
  Apache-2.0 or MIT.

Release packaging must include the license notices corresponding to the exact
pinned dependency revisions recorded by the repository's Git submodule links.
The dependency repositories contain their complete license texts.
