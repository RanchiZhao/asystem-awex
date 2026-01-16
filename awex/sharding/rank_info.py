# Licensed to the Awex developers under one
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

from dataclasses import dataclass


@dataclass(slots=True)
class RankInfo:
    """
    Holds information about the distributed training ranks and sizes for a model worker.

    Attributes:
        tp_rank (int): Tensor parallel rank.
        tp_size (int): Tensor parallel size.
        pp_rank (int): Pipeline parallel rank.
        pp_size (int): Pipeline parallel size.
        dp_size (int): Data parallel size.
        attn_tp_rank (int): Attention tensor parallel rank.
        attn_tp_size (int): Attention tensor parallel size.
        attn_dp_rank (int): Attention data parallel rank.
        world_size (int): Total world size (within engine for multi-engine mode).
        global_rank (int): Global rank within engine (0 to world_size-1 for multi-engine).
        local_rank (int): Local rank of the worker.
        engine_rank (int): Engine rank of the worker (0 to num_engines-1).
        is_infer (bool): Whether the worker is an inference worker.
        true_global_rank (int): True global rank across all engines (0 to total_world_size-1).
        num_engines (int): Total number of inference engines.
        infer_instance_world_size (int): World size per inference engine instance.
    """

    tp_rank: int
    tp_size: int
    pp_rank: int
    pp_size: int
    dp_size: int
    dp_rank: int
    ep_rank: int
    ep_size: int
    ep_tp_rank: int
    ep_tp_size: int
    attn_tp_rank: int
    attn_tp_size: int
    attn_dp_rank: int
    world_size: int
    global_rank: int
    local_rank: int
    engine_rank: int
    is_infer: bool
    true_global_rank: int = None  # Real global rank (0 to infer_world_size-1)
    num_engines: int = 1  # Total number of inference engines
    infer_instance_world_size: int = None  # World size per engine (tp_size * pp_size)

    def __post_init__(self):
        """Initialize derived fields if not provided."""
        if self.true_global_rank is None:
            # Default: true_global_rank equals global_rank for backward compatibility
            self.true_global_rank = self.global_rank
        if self.infer_instance_world_size is None:
            # Default: compute from tp_size and pp_size
            self.infer_instance_world_size = self.tp_size * self.pp_size
