# Copyright 2018 The apache/tvm Authors. All Rights Reserved.
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
# Modifications Copyright (c) Microsoft.
"""Backward-compatible re-exports for the merged Reduction rule.

The standalone Reduction schedule rule has been merged into
GeneralReduction.  This module re-exports GeneralReduction as
Reduction and all shared helper functions so that existing imports
continue to work.
"""

from .general_reduction import GeneralReduction as Reduction  # noqa: F401

# Re-export all helpers that other modules historically imported from here.
from .reduction_utils import (  # noqa: F401
    _MIN_ELEMS_PER_THREAD,
    _as_const_int,
    _as_const_float,
    _is_accumulator_term,
    _analyze_reduction_update,
    _extract_single_input_buffer,
    _collect_input_buffers,
    _is_direct_buffer_load,
    _is_same_buffer_load,
    _is_square_of_buffer_load,
    _block_writes_buffer,
    _find_buffer_index,
    _infer_reduce_dim,
    _default_init_value,
    _infer_init_value,
    _choose_num_threads,
    _choose_reduction_step,
)
