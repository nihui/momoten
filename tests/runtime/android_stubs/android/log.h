// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0
#ifndef MOMOTEN_TEST_ANDROID_LOG_H
#define MOMOTEN_TEST_ANDROID_LOG_H
#define ANDROID_LOG_DEBUG 3
extern "C" int __android_log_print(int priority, const char* tag, const char* format, ...);
#endif
