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

The Android 7 ELF lookup in `src/backend/android7_loader.h` follows the
[Android stub patch](https://github.com/nihui/mesa-turnip-android-driver/blob/d88dc83e06bcdd6dfa10413ef7e576c96b8b8830/mesa-implement-android_stub.patch)
from nihui/mesa-turnip-android-driver (MIT, copyright 2016 avs333). Its complete
permission notice is also retained in the header and generated amalgamation:

Copyright (c) 2016 avs333

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
