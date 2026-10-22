// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#ifndef MOMOTEN_BACKEND_REPLAY_H
#define MOMOTEN_BACKEND_REPLAY_H

#include "objects.h"

struct ReplayState
{
    explicit ReplayState(VkDevice value);

    VkDevice device;
    std::vector<std::shared_ptr<DeviceMemory> > touched_memories;
};

cl_int replay_copy(ReplayState& state, const RecordedCommand& command);
cl_int replay_barrier(ReplayState& state, const RecordedCommand& command);
cl_int replay_dispatch(ReplayState& state, const RecordedCommand& command);
cl_int finish_replay(ReplayState& state, cl_event& event);
void abort_replay(ReplayState& state);

#endif // MOMOTEN_BACKEND_REPLAY_H
