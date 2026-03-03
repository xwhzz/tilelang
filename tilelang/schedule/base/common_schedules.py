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
# The code below is mostly copied from apache/tvm common_schedules.py in dlight.
"""Common schedule strategies for TIR."""
from typing import List
from tvm import tir
from tvm import IRModule
from .analysis import BlockInfo

def _retrieve_func_from_module(ir_module: IRModule) -> tir.PrimFunc:
    if not isinstance(ir_module, IRModule):
        raise ValueError("Not supported type: ", type(ir_module))
    assert len(ir_module.get_global_vars()) == 1, (
        "The optimized module should only have one global variable for default schedule.")
    func = list(ir_module.functions.values())[0]
    return func

def get_block(
    sch: tir.Schedule,
    blocks: List[BlockInfo],
    name: str,
):
    """Get the target block from a schedule.

    Parameters
    ----------
    sch : tir.Schedule
        The TIR schedule used to get target block.
    name : str
        The name of the target block.

    Returns
    -------
    target_block : BlockRV
        The target block.
    """

    target_block: tir.BlockRV = None
    for block_info in blocks:
        block = block_info.block_rv
        if sch.get(block).name_hint == name:
            target_block = block
    return target_block


def get_output_blocks(
    sch: tir.Schedule,
    blocks: List[BlockInfo],
):
    """Get the output blocks of a schedule.

    Parameters
    ----------
    sch : tir.Schedule
        The TIR schedule used to get output blocks.
    blocks : List[BlockInfo]
        The blocks to be analyzed.

    Returns
    -------
    output_blocks : List[BlockInfo]
        The output blocks.
    """

    # collect arguments buffer
    func = _retrieve_func_from_module(sch.mod)
    args = list(func.buffer_map.values())

    output_blocks = []
    for block_info in blocks:
        block = block_info.block_rv
        for write in sch.get(block).writes:
            if write.buffer in args:
                output_blocks.append(block)

    return output_blocks

