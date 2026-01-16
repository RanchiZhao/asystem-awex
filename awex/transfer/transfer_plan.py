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

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch

from awex import logging
from awex.meta.meta_resolver import (
    ParameterMeta,
    ParameterReplicaMeta,
    ParameterShardMeta,
)
from awex.util.common import to_dict

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CommunicationOperation:
    """Represents a single communication operation in the weights exchange plan."""

    send_rank: int
    send_shard_meta: ParameterShardMeta
    send_offset: Tuple[int, ...]
    recv_rank: int
    recv_shard_meta: ParameterShardMeta
    recv_offset: Tuple[int, ...]
    overlap_shape: Tuple[int, ...]
    train_slices: Tuple[slice, ...]
    inf_slices: Tuple[slice, ...]


@dataclass(slots=True)
class TransferPlan:
    """Represents a transfer plan for a specific rank.

    The operations dictionary maps from the opposite rank (the rank we communicate with)
    to a list of communication operations:
    - For training ranks: maps inference rank -> operations where this training rank sends to that inference rank
    - For inference ranks: maps training rank -> operations where this inference rank receives from that training rank
    """

    # mapping from opposite rank to list of communication operations
    operations: Dict[int, List[CommunicationOperation]]


@dataclass(slots=True)
class ShardOffset:
    shard: Any
    start_offset: tuple
    end_offset: tuple
    shape: tuple
    rank: int


@dataclass(slots=True)
class OverlapRegion:
    inference_shard: ShardOffset
    training_shard: ShardOffset
    overlap_start: tuple
    overlap_end: tuple


class TransferPlanBuilder:
    def __init__(
        self,
        infer_world_size: int,
        train_world_size: int,
        num_infer_engines: int = 1,
        enable_debug_mode: bool = False,
        hf_config=None,
        enable_colocate_mode: bool = False,
    ):
        if num_infer_engines <= 0:
            raise ValueError("num_infer_engines must be positive")

        self.train_world_size = train_world_size
        self.infer_world_size = infer_world_size
        self.world_size = train_world_size + infer_world_size
        self.infer_instance_world_size = infer_world_size // num_infer_engines
        self.num_infer_engines = num_infer_engines
        self.enable_debug_mode = enable_debug_mode
        self.hf_config = hf_config
        self.enable_colocate_mode = enable_colocate_mode
        logger.info(
            f"TransferPlanBuilder: infer_world_size: {infer_world_size}, train_world_size: {train_world_size}, world_size: {self.world_size}, "
            f"infer_instance_world_size: {self.infer_instance_world_size}, num_infer_engines: {num_infer_engines}, "
            f"enable_debug_mode: {enable_debug_mode}, enable_colocate_mode: {enable_colocate_mode}"
        )

    def build_weights_mapping_operations(
        self,
        inference_weights_meta: List[ParameterMeta],
        training_weights_meta: List[ParameterMeta],
        global_transfer_rank: int = None,
    ) -> List[CommunicationOperation]:
        """
        Build the weights transfer plan.

        Args:
            inference_weights_meta: List of inference parameter metadata
            training_weights_meta: List of training parameter metadata
            global_transfer_rank: If provided, only build operations for this specific rank
            is_train: If provided, indicates whether this is a training rank
        Returns:
            TransferPlan containing communication operations
        """
        # Validate input parameters
        if not inference_weights_meta or not training_weights_meta:
            logger.warning("Empty weights meta provided, returning empty plan")
            return []

        # Create a mapping from parameter name to ParameterMeta for both inference and training
        inference_meta_dict = {meta.name: meta for meta in inference_weights_meta}
        training_meta_dict = {meta.name: meta for meta in training_weights_meta}

        # Get all parameter names that exist in both inference and training
        common_params = set(inference_meta_dict.keys()) & set(training_meta_dict.keys())

        # In colocate mode with EP sharding, inference side only has LOCAL parameters (e.g., its experts),
        # while training side has ALL parameters aggregated from all ranks.
        # We only require that all INFERENCE params exist in training (subset relationship).
        # The extra training params (for other inference ranks) are expected.
        if self.enable_colocate_mode:
            # Colocate mode: inference params must be subset of training params
            infer_only_params = set(inference_meta_dict.keys()) - set(training_meta_dict.keys())
            if infer_only_params:
                logger.error(
                    f"[COLOCATE] Inference has params not in training: {len(infer_only_params)} params"
                )
                logger.error(f"[COLOCATE] Inference-only params (first 10): {list(infer_only_params)[:10]}")
                raise ValueError(
                    f"[COLOCATE] Inference has {len(infer_only_params)} params not in training: {list(infer_only_params)[:5]}..."
                )
            # Log the difference for debugging
            train_only_count = len(training_meta_dict) - len(common_params)
            logger.info(
                f"[COLOCATE] Param matching: inference={len(inference_meta_dict)}, training={len(training_meta_dict)}, "
                f"common={len(common_params)}, train_only={train_only_count} (expected for EP/other ranks)"
            )
        else:
            # Non-colocate mode: require exact match
            if len(common_params) != len(inference_weights_meta) or len(
                common_params
            ) != len(training_weights_meta):
                logger.error(
                    f"inference weights and training weights are not consistent: {len(inference_meta_dict)}, {len(training_meta_dict)}"
                )
                logger.error(
                    f"inference weights and training weights are not consistent, inference weights: {list(inference_meta_dict.keys())}"
                )
                logger.error(
                    f"inference weights and training weights are not consistent, training weights: {list(training_meta_dict.keys())}"
                )
                diff_params = set(inference_meta_dict.keys()) - set(
                    training_meta_dict.keys()
                )
                raise ValueError(
                    f"inference weights and training weights are not consistent: {len(common_params)}, {len(inference_weights_meta)}, {len(training_weights_meta)}, {diff_params}"
                )

        communication_plan = []

        for param_name in sorted(common_params):
            inference_meta = inference_meta_dict[param_name]
            training_meta = training_meta_dict[param_name]

            # Build communication plan for this parameter
            param_plan = self._build_parameter_communication_plan(
                param_name,
                inference_meta,
                training_meta,
                global_transfer_rank=global_transfer_rank,
            )
            communication_plan.extend(param_plan)

        logger.info(
            f"Built communication plan with {len(communication_plan)} operations "
            f"for {len(common_params)} common parameters"
        )
        return communication_plan

    def _build_parameter_communication_plan(
        self,
        param_name: str,
        inference_meta: ParameterMeta,
        training_meta: ParameterMeta,
        global_transfer_rank: int = None,
    ) -> List[CommunicationOperation]:
        """
        Build communication plan for a single parameter for a specific rank.

        This method handles multiple replicas by:
        1. Getting all replicas from both inference and training ParameterMeta
        2. Assigning training replicas to inference replicas evenly using round-robin
        3. For each replica pair, finding overlapping regions and building communication operations
        4. Each inference replica receives tensors from one assigned training replica

        Args:
            param_name: Name of the parameter
            inference_meta: Inference parameter metadata with replicas
            training_meta: Training parameter metadata with replicas

        Returns:
            List of communication operations for this parameter
        """
        # Special handling for qkv_proj with KV head replication
        if self._is_qkv_param(param_name) and self._needs_qkv_head_mapping(inference_meta, training_meta):
            return self._build_qkv_communication_plan(
                param_name, inference_meta, training_meta, global_transfer_rank
            )

        # Special handling for MoE expert weights with EP→TP conversion
        if self._is_expert_param(param_name) and self._needs_expert_mapping(inference_meta, training_meta):
            return self._build_expert_communication_plan(
                param_name, inference_meta, training_meta, global_transfer_rank
            )

        plan = []

        # Get the replicas for both inference and training
        inference_replicas = []
        for engine_rank in range(self.num_infer_engines):
            inference_replicas.extend(
                (engine_rank, replica) for replica in inference_meta.replicas
            )
        training_replicas = training_meta.replicas

        if not inference_replicas or not training_replicas:
            raise ValueError(f"No replicas found for parameter {param_name}")

        # Assign training replicas to inference replicas using proper distribution
        num_inference_replicas = len(inference_replicas)
        num_training_replicas = len(training_replicas)

        # Ensure each inference replica is assigned to exactly one training replica
        # and each training replica gets an equal number of inference replicas
        replica_assignments = []

        # Use round-robin assignment to distribute inference replicas evenly
        for inf_replica_idx in range(num_inference_replicas):
            # Assign each inference replica to a training replica using modulo
            train_replica_idx = inf_replica_idx % num_training_replicas
            replica_assignments.append((inf_replica_idx, train_replica_idx))

        logger.debug(
            f"Parameter {param_name}: Assigned {num_inference_replicas} inference replicas "
            f"to {num_training_replicas} training replicas: {replica_assignments}"
        )

        # Build communication plan for each replica pair
        for inf_replica_idx, train_replica_idx in replica_assignments:
            engine_rank, inference_replica = inference_replicas[inf_replica_idx]
            training_replica = training_replicas[train_replica_idx]
            is_infer_rank = (
                global_transfer_rank is not None
                and global_transfer_rank < self.infer_world_size
            )
            is_train_rank = (
                global_transfer_rank is not None
                and global_transfer_rank >= self.infer_world_size
            )
            reused_shape_obj = [None] * len(training_replica.shards[0].shape)
            # Create a mapping from global offset ranges to shards for both inference and training replicas
            inference_shard_offsets = self._create_shard_offset_for_replica(
                inference_replica,
                engine_rank,
                is_infer=True,
                global_transfer_rank=global_transfer_rank if is_infer_rank else None,
                reused_shape_obj=reused_shape_obj,
            )
            training_shard_offsets = self._create_shard_offset_for_replica(
                training_replica,
                0,
                is_infer=False,
                global_transfer_rank=global_transfer_rank if is_train_rank else None,
                reused_shape_obj=reused_shape_obj,
            )

            # Find overlapping regions between inference and training shards
            overlapping_regions = self._find_overlapping_regions(
                inference_shard_offsets,
                training_shard_offsets,
            )
            for region in overlapping_regions:
                region_plan = self._build_region_communication_plan(
                    region, reused_shape_obj
                )
                plan.extend(region_plan)

        return plan

    def _create_shard_offset_for_replica(
        self,
        replica: ParameterReplicaMeta,
        engine_rank: int,
        is_infer: bool,
        global_transfer_rank: int = None,
        reused_shape_obj=None,
    ) -> List[ShardOffset]:
        """
        Create a mapping from offset ranges to shards for a specific replica.

        Args:
            replica: ParameterReplicaMeta containing a list of shards
            is_infer: Whether this is for inference replicas

        Returns:
            List of dictionaries containing shard information with offset ranges
        """
        assert is_infer is not None, (
            "is_infer must be provided if global_transfer_rank is provided"
        )
        shard_map = []
        reused_shape_obj = reused_shape_obj or [None] * len(replica.shards[0].shape)

        if not replica.shards:
            raise ValueError("No shards found for replica")

        for shard in replica.shards:
            # Validate that global_offset is not empty
            if not shard.global_offset:
                raise ValueError("Shard has empty global_offset")

            # Validate that global_offset and shape have the same number of dimensions
            if len(shard.global_offset) != len(shard.shape):
                raise ValueError(
                    f"Dimension mismatch for shard in replica: "
                    f"global_offset has {len(shard.global_offset)} dims, shape has {len(shard.shape)} dims"
                )
            shard_transfer_rank = self._compute_shard_transfer_rank(
                shard, engine_rank, is_infer
            )
            if (
                global_transfer_rank is not None
                and shard_transfer_rank != global_transfer_rank
            ):
                continue

            # Calculate the end offset for this shard
            for i in range(len(shard.global_offset)):
                reused_shape_obj[i] = shard.global_offset[i] + shard.shape[i]
            end_offset = tuple(reused_shape_obj)
            shard_map.append(
                ShardOffset(
                    shard=shard,
                    start_offset=shard.global_offset,
                    end_offset=end_offset,
                    shape=shard.shape,
                    rank=shard_transfer_rank,
                )
            )

        return shard_map

    def _compute_shard_transfer_rank(
        self, shard: ParameterShardMeta, engine_rank: int, is_infer: bool
    ) -> int:
        """
        Compute the transfer rank for a shard.

        For inference shards:
        - TP_SHARDING: shard.global_rank is virtual (0 to instance_world_size-1)
          → Need to add engine_rank * instance_world_size to get real rank
        - EP_SHARDING: shard.global_rank is already real (0 to infer_world_size-1)
          → In multi-engine colocate mode, already the correct rank

        For training shards:
        - global_rank is always virtual, need to add infer_world_size offset
        """
        if is_infer:
            # In colocate mode with multiple engines, EP_SHARDING params already have
            # real global_rank (because EP uses actual rank that owns each expert).
            # Don't add engine_rank offset for these.
            from awex.sharding.param_sharding import ShardingType
            if (self.enable_colocate_mode and
                self.num_infer_engines > 1 and
                hasattr(shard, 'sharding_type') and
                shard.sharding_type == ShardingType.EP_SHARDING):
                # EP sharding in multi-engine colocate mode: global_rank is already real
                return shard.global_rank
            else:
                # TP/other sharding: global_rank is virtual, needs engine_rank offset
                return shard.global_rank + engine_rank * self.infer_instance_world_size
        else:
            assert engine_rank == 0, "Training only has one engine instance"
            return shard.global_rank + self.infer_world_size

    def _find_overlapping_regions(
        self,
        inference_map: List[ShardOffset],
        training_map: List[ShardOffset],
    ) -> List[OverlapRegion]:
        """Find overlapping regions between inference and training shards."""
        overlapping_regions = []

        for inf_shard_offset in inference_map:
            for train_shard_offset in training_map:
                # Check if shards overlap in all dimensions
                overlap_start = []
                overlap_end = []
                has_overlap = True

                for dim in range(len(inf_shard_offset.start_offset)):
                    start = max(
                        inf_shard_offset.start_offset[dim],
                        train_shard_offset.start_offset[dim],
                    )
                    end = min(
                        inf_shard_offset.end_offset[dim],
                        train_shard_offset.end_offset[dim],
                    )

                    if start >= end:
                        has_overlap = False
                        break

                    overlap_start.append(start)
                    overlap_end.append(end)

                if has_overlap:
                    overlapping_regions.append(
                        OverlapRegion(
                            inference_shard=inf_shard_offset,
                            training_shard=train_shard_offset,
                            overlap_start=tuple(overlap_start),
                            overlap_end=tuple(overlap_end),
                        )
                    )

        return overlapping_regions

    def _build_region_communication_plan(
        self,
        region: OverlapRegion,
        reused_shape_obj: List = None,
    ) -> List[CommunicationOperation]:
        """Build communication plan for an overlapping region."""
        plan = []

        inf_shard_offset = region.inference_shard
        train_shard_offset = region.training_shard
        overlap_start = region.overlap_start
        overlap_end = region.overlap_end
        reused_shape_obj = reused_shape_obj or [None] * len(
            region.inference_shard.shape
        )

        # Calculate the shape of the overlapping region
        overlap_shape = tuple(
            overlap_end[i] - overlap_start[i] for i in range(len(overlap_start))
        )

        # Calculate relative offsets within each shard
        for i in range(len(overlap_start)):
            reused_shape_obj[i] = overlap_start[i] - inf_shard_offset.start_offset[i]
        inf_relative_offset = tuple(reused_shape_obj)

        for i in range(len(overlap_start)):
            reused_shape_obj[i] = overlap_start[i] - train_shard_offset.start_offset[i]
        train_relative_offset = tuple(reused_shape_obj)

        # Create conversion functions for both training (sender) and inference (receiver)
        # Each side may need to slice if their shard is larger than the overlapping region
        for i in range(len(reused_shape_obj)):
            reused_shape_obj[i] = slice(
                train_relative_offset[i], train_relative_offset[i] + overlap_shape[i]
            )
        train_slices = tuple(reused_shape_obj)
        for i in range(len(reused_shape_obj)):
            reused_shape_obj[i] = slice(
                inf_relative_offset[i], inf_relative_offset[i] + overlap_shape[i]
            )
        infer_slices = tuple(reused_shape_obj)

        # Add to communication plan
        # Training is sender, inference is receiver
        # Only transfer the overlapping region
        send_rank = train_shard_offset.rank
        recv_rank = inf_shard_offset.rank
        assert send_rank != recv_rank, "Send and recv rank cannot be the same"
        assert send_rank >= self.infer_world_size, (
            f"Send rank {send_rank} is not in training world {self.infer_world_size, self.world_size}"
        )
        assert recv_rank < self.infer_world_size, (
            f"Recv rank {recv_rank} is not in inference world {0, self.infer_world_size}"
        )
        plan.append(
            CommunicationOperation(
                send_rank=send_rank,
                send_shard_meta=train_shard_offset.shard,
                send_offset=train_relative_offset,
                recv_rank=recv_rank,
                recv_shard_meta=inf_shard_offset.shard,
                recv_offset=inf_relative_offset,
                overlap_shape=overlap_shape,
                train_slices=train_slices,
                inf_slices=infer_slices,
            )
        )
        # [VERIFY] Log TransferPlan operations for correctness verification
        op = plan[-1]
        if "embed_tokens" in train_shard_offset.shard.name or "qkv_proj" in train_shard_offset.shard.name:
            logger.info(
                f"[TRANSFER_OP] param={op.send_shard_meta.name} "
                f"send_rank={op.send_rank} recv_rank={op.recv_rank} "
                f"overlap_shape={op.overlap_shape} "
                f"train_slices={op.train_slices} inf_slices={op.inf_slices} "
                f"train_global_offset={train_shard_offset.shard.global_offset} "
                f"inf_global_offset={inf_shard_offset.shard.global_offset}"
            )

        return plan

    def _group_operations_by_rank(
        self, operations: List[CommunicationOperation], key_name: str
    ) -> Dict[int, List[CommunicationOperation]]:
        grouped_operations = {}
        for operation in operations:
            key = getattr(operation, key_name)
            if key not in grouped_operations:
                grouped_operations[key] = []
            grouped_operations[key].append(operation)

        # Sort operations within each group to ensure consistent ordering
        for key in grouped_operations:
            grouped_operations[key].sort(
                key=lambda op: (
                    op.send_shard_meta.name,  # Sort by parameter name first
                    op.send_offset,  # Then by send offset
                    op.recv_offset,  # Then by recv offset
                )
            )
        # Sort the keys to ensure consistent iteration order
        sorted_grouped_operations = {}
        for key in sorted(grouped_operations.keys()):
            sorted_grouped_operations[key] = grouped_operations[key]
        return sorted_grouped_operations

    def _is_qkv_param(self, param_name: str) -> bool:
        """Check if the parameter is a fused QKV projection."""
        return "qkv_proj" in param_name

    def _needs_qkv_head_mapping(
        self, inference_meta: ParameterMeta, training_meta: ParameterMeta
    ) -> bool:
        """
        Check if QKV parameter needs special head-level mapping.
        This is needed when inference has KV head replication (global shapes differ).
        """
        if not self.hf_config:
            return False

        infer_global_shape = inference_meta.global_shape
        train_global_shape = training_meta.global_shape

        # If shapes are equal, no special handling needed
        if infer_global_shape == train_global_shape:
            return False

        # Check if this is a GQA model with KV replication scenario
        num_attention_heads = getattr(self.hf_config, "num_attention_heads", None)
        num_kv_heads = getattr(self.hf_config, "num_key_value_heads", None) or getattr(
            self.hf_config, "num_query_groups", num_attention_heads
        )

        if num_kv_heads is None or num_attention_heads is None:
            return False

        # Need special handling if inference TP > num_kv_heads (causes KV replication)
        infer_tp_size = self.infer_instance_world_size
        if infer_tp_size > num_kv_heads:
            logger.info(
                f"QKV head mapping enabled: infer_tp={infer_tp_size} > num_kv_heads={num_kv_heads}, "
                f"train_shape={train_global_shape}, infer_shape={infer_global_shape}"
            )
            return True

        return False

    def _build_qkv_communication_plan(
        self,
        param_name: str,
        inference_meta: ParameterMeta,
        training_meta: ParameterMeta,
        global_transfer_rank: int = None,
    ) -> List[CommunicationOperation]:
        """
        Build communication plan for QKV projection with head-level mapping.

        This handles the case where inference has KV head replication:
        - Training: [Q|K|V] with unique K/V data
        - Inference: [Q|K|V] with replicated K/V data

        For example (training TP=4, inference TP=8, num_kv_heads=4):
        - Training rank 0: Q heads 0-7, K head 0, V head 0 (1280 elements)
        - Inference rank 0: Q heads 0-3, K head 0, V head 0 (768 elements)
        - Inference rank 1: Q heads 4-7, K head 0, V head 0 (768 elements)

        Train rank 0's K/V should be sent to BOTH inference ranks 0 and 1 (fan-out).
        """
        plan = []

        # Get model config
        num_q_heads = getattr(self.hf_config, "num_attention_heads", 32)
        num_kv_heads = getattr(self.hf_config, "num_key_value_heads", None) or getattr(
            self.hf_config, "num_query_groups", num_q_heads
        )
        head_dim = getattr(self.hf_config, "head_dim", None) or getattr(
            self.hf_config, "kv_channels", None
        )
        if head_dim is None:
            hidden_size = getattr(self.hf_config, "hidden_size", 2048)
            head_dim = hidden_size // num_q_heads

        # Compute TP sizes from world sizes
        # In colocate mode, training and inference share GPUs
        # Training world size includes DP replicas, so we need to infer TP from shards
        train_tp_size = len(training_meta.replicas[0].shards) if training_meta.replicas else 1
        infer_tp_size = self.infer_instance_world_size

        # For DP training, we only use one replica (DP replicas have same data)
        training_replica = training_meta.replicas[0]

        # KV replication factor
        kv_replicas = max(1, infer_tp_size // num_kv_heads)

        logger.info(
            f"QKV head mapping for {param_name}: num_q_heads={num_q_heads}, num_kv_heads={num_kv_heads}, "
            f"head_dim={head_dim}, train_tp={train_tp_size}, infer_tp={infer_tp_size}, kv_replicas={kv_replicas}"
        )

        # Compute per-rank head assignments
        train_q_heads_per_rank = num_q_heads // train_tp_size
        train_kv_heads_per_rank = num_kv_heads // train_tp_size
        infer_q_heads_per_rank = num_q_heads // infer_tp_size

        # Process each training shard
        for train_shard in training_replica.shards:
            train_tp_rank = train_shard.tp_rank
            train_transfer_rank = train_shard.global_rank + self.infer_world_size

            # Skip if filtering by rank and this isn't the right rank
            if global_transfer_rank is not None and train_transfer_rank != global_transfer_rank:
                # Also check if this rank is an inference rank that needs data from this training rank
                if global_transfer_rank >= self.infer_world_size:
                    continue

            # Training shard layout: [Q | K | V]
            train_q_size = train_q_heads_per_rank * head_dim
            train_k_size = train_kv_heads_per_rank * head_dim
            train_v_size = train_kv_heads_per_rank * head_dim

            # Q head range for this training rank
            train_q_start_head = train_tp_rank * train_q_heads_per_rank
            train_q_end_head = train_q_start_head + train_q_heads_per_rank

            # KV head range for this training rank
            train_kv_start_head = train_tp_rank * train_kv_heads_per_rank
            train_kv_end_head = train_kv_start_head + train_kv_heads_per_rank

            # Process each inference engine
            for engine_rank in range(self.num_infer_engines):
                for inf_replica in inference_meta.replicas:
                    for inf_shard in inf_replica.shards:
                        infer_tp_rank = inf_shard.tp_rank
                        infer_transfer_rank = inf_shard.global_rank + engine_rank * self.infer_instance_world_size

                        # Skip if filtering by rank
                        if global_transfer_rank is not None:
                            if global_transfer_rank < self.infer_world_size:
                                # Inference rank: only include ops where this rank is receiver
                                if infer_transfer_rank != global_transfer_rank:
                                    continue
                            else:
                                # Training rank: only include ops where this rank is sender
                                if train_transfer_rank != global_transfer_rank:
                                    continue

                        # Inference shard layout: [Q | K | V]
                        infer_q_size = infer_q_heads_per_rank * head_dim
                        infer_k_size = head_dim  # 1 KV head per rank (replicated)
                        infer_v_size = head_dim

                        # Q head range for this inference rank
                        infer_q_start_head = infer_tp_rank * infer_q_heads_per_rank
                        infer_q_end_head = infer_q_start_head + infer_q_heads_per_rank

                        # KV head for this inference rank (with replication)
                        infer_kv_head = infer_tp_rank // kv_replicas

                        # === Q transfer ===
                        # Find overlap between training Q heads and inference Q heads
                        q_overlap_start = max(train_q_start_head, infer_q_start_head)
                        q_overlap_end = min(train_q_end_head, infer_q_end_head)

                        if q_overlap_start < q_overlap_end:
                            num_overlap_q_heads = q_overlap_end - q_overlap_start
                            overlap_q_size = num_overlap_q_heads * head_dim

                            # Training offset: relative to Q start in training shard
                            train_q_offset = (q_overlap_start - train_q_start_head) * head_dim
                            # Inference offset: relative to Q start in inference shard
                            infer_q_offset = (q_overlap_start - infer_q_start_head) * head_dim

                            hidden_dim = train_shard.shape[1] if len(train_shard.shape) > 1 else 1

                            plan.append(
                                CommunicationOperation(
                                    send_rank=train_transfer_rank,
                                    send_shard_meta=train_shard,
                                    send_offset=(train_q_offset, 0),
                                    recv_rank=infer_transfer_rank,
                                    recv_shard_meta=inf_shard,
                                    recv_offset=(infer_q_offset, 0),
                                    overlap_shape=(overlap_q_size, hidden_dim),
                                    train_slices=(slice(train_q_offset, train_q_offset + overlap_q_size), slice(0, hidden_dim)),
                                    inf_slices=(slice(infer_q_offset, infer_q_offset + overlap_q_size), slice(0, hidden_dim)),
                                )
                            )

                        # === K transfer (with fan-out for replication) ===
                        # Check if this inference rank needs K data from this training rank
                        if train_kv_start_head <= infer_kv_head < train_kv_end_head:
                            # Training K offset: relative to K start in training shard
                            train_k_local_idx = infer_kv_head - train_kv_start_head
                            train_k_offset = train_q_size + train_k_local_idx * head_dim
                            # Inference K offset: relative to start of inference shard
                            infer_k_offset = infer_q_size

                            hidden_dim = train_shard.shape[1] if len(train_shard.shape) > 1 else 1

                            plan.append(
                                CommunicationOperation(
                                    send_rank=train_transfer_rank,
                                    send_shard_meta=train_shard,
                                    send_offset=(train_k_offset, 0),
                                    recv_rank=infer_transfer_rank,
                                    recv_shard_meta=inf_shard,
                                    recv_offset=(infer_k_offset, 0),
                                    overlap_shape=(head_dim, hidden_dim),
                                    train_slices=(slice(train_k_offset, train_k_offset + head_dim), slice(0, hidden_dim)),
                                    inf_slices=(slice(infer_k_offset, infer_k_offset + head_dim), slice(0, hidden_dim)),
                                )
                            )

                        # === V transfer (with fan-out for replication) ===
                        # Check if this inference rank needs V data from this training rank
                        if train_kv_start_head <= infer_kv_head < train_kv_end_head:
                            # Training V offset
                            train_v_local_idx = infer_kv_head - train_kv_start_head
                            train_v_offset = train_q_size + train_k_size + train_v_local_idx * head_dim
                            # Inference V offset
                            infer_v_offset = infer_q_size + infer_k_size

                            hidden_dim = train_shard.shape[1] if len(train_shard.shape) > 1 else 1

                            plan.append(
                                CommunicationOperation(
                                    send_rank=train_transfer_rank,
                                    send_shard_meta=train_shard,
                                    send_offset=(train_v_offset, 0),
                                    recv_rank=infer_transfer_rank,
                                    recv_shard_meta=inf_shard,
                                    recv_offset=(infer_v_offset, 0),
                                    overlap_shape=(head_dim, hidden_dim),
                                    train_slices=(slice(train_v_offset, train_v_offset + head_dim), slice(0, hidden_dim)),
                                    inf_slices=(slice(infer_v_offset, infer_v_offset + head_dim), slice(0, hidden_dim)),
                                )
                            )

        logger.info(f"QKV head mapping generated {len(plan)} operations for {param_name}")
        return plan

    def _is_expert_param(self, param_name: str) -> bool:
        """Check if the parameter is an MoE expert parameter."""
        # Detect expert params like: mlp.experts.0.gate_up_proj.weight, mlp.experts.0.down_proj.weight
        # But not shared_experts or router/gate
        if "shared_experts" in param_name:
            return False
        if "expert_bias" in param_name or "gate.weight" in param_name or "router" in param_name:
            return False
        return "experts." in param_name

    def _needs_expert_mapping(
        self, inference_meta: ParameterMeta, training_meta: ParameterMeta
    ) -> bool:
        """
        Check if expert parameter needs special EP→TP mapping.
        This is needed when training uses EP sharding (experts split by expert_id)
        but inference uses TP sharding (experts split by intermediate dimension).
        """
        if not self.hf_config:
            return False

        # Get sharding info from the first replica's first shard
        if not training_meta.replicas or not inference_meta.replicas:
            return False

        train_first_shard = training_meta.replicas[0].shards[0]
        infer_first_shard = inference_meta.replicas[0].shards[0]

        # Check if training uses EP sharding and inference uses TP sharding
        from awex.sharding.param_sharding import ShardingType
        train_sharding = train_first_shard.sharding_type
        infer_sharding = infer_first_shard.sharding_type

        # Need special handling when:
        # - Training has EP_SHARDING (experts split by expert_id across EP ranks)
        # - Inference has TP_SHARDING (experts split by intermediate dimension)
        needs_mapping = (
            train_sharding == ShardingType.EP_SHARDING and
            infer_sharding == ShardingType.TP_SHARDING
        )

        if needs_mapping:
            logger.info(
                f"Expert EP→TP mapping enabled for {training_meta.name}: "
                f"train_sharding={train_sharding}, infer_sharding={infer_sharding}, "
                f"train_shape={training_meta.global_shape}, infer_shape={inference_meta.global_shape}"
            )

        return needs_mapping

    def _build_expert_communication_plan(
        self,
        param_name: str,
        inference_meta: ParameterMeta,
        training_meta: ParameterMeta,
        global_transfer_rank: int = None,
    ) -> List[CommunicationOperation]:
        """
        Build communication plan for MoE expert weights with EP→TP conversion.

        This handles the case where:
        - Training: EP sharding (each GPU has N/EP_size complete experts)
        - Inference: TP sharding (each GPU has all N experts but only 1/TP_size of intermediate dim)

        For example (training EP=8, inference TP=8, num_experts=128):
        - Training rank 0: experts 0-15, complete intermediate_size (1536)
        - Inference rank 0: all 128 experts, intermediate_size[0:192]
        - Inference rank 1: all 128 experts, intermediate_size[192:384]
        ...

        Each training rank sends slices to ALL inference ranks.
        """
        plan = []

        # Determine expert intermediate_size from the shape
        # Training shape: (intermediate_size, hidden_size) for gate_up_proj
        # or (hidden_size, intermediate_size) for down_proj
        train_first_shard = training_meta.replicas[0].shards[0]
        infer_first_shard = inference_meta.replicas[0].shards[0]

        # Infer dimensions from actual shapes
        train_shape = train_first_shard.shape  # e.g., (1536, 2048) for gate_up_proj
        infer_shape = infer_first_shard.shape  # e.g., (192, 2048) for gate_up_proj

        # Get sharding dimension (0 for gate_up_proj, 1 for down_proj)
        sharding_dim = infer_first_shard.sharding_dim

        # Compute TP/EP sizes
        infer_tp_size = self.infer_instance_world_size

        # Size of each slice for inference
        full_dim_size = train_shape[sharding_dim]  # Complete dimension on training side
        slice_size = full_dim_size // infer_tp_size

        hidden_dim = train_shape[1 - sharding_dim] if len(train_shape) > 1 else 1

        # Get the absolute expert ID from the parameter name
        # e.g., "mlp.experts.15.gate_up_proj.weight" -> expert_id = 15
        expert_id = self._extract_expert_id(param_name)
        if expert_id is None:
            logger.warning(f"Could not extract expert_id from {param_name}")
            return []

        is_gate_up_param = "gate_up_proj" in param_name and sharding_dim == 0
        logger.info(
            f"Expert EP→TP mapping for {param_name}: expert_id={expert_id}, "
            f"infer_tp_size={infer_tp_size}, "
            f"full_dim={full_dim_size}, slice_size={slice_size}, sharding_dim={sharding_dim}, "
            f"is_gate_up={is_gate_up_param}, "
            f"train_replicas={len(training_meta.replicas)}, infer_replicas={len(inference_meta.replicas)}, "
            f"train_shards_per_replica={[len(r.shards) for r in training_meta.replicas]}, "
            f"infer_shards_per_replica={[len(r.shards) for r in inference_meta.replicas]}, "
            f"train_shape={train_shape}, infer_shape={infer_shape}"
        )

        # Process each training replica
        # For EP_SHARDING, each expert param should have exactly 1 training shard
        for train_replica in training_meta.replicas:
            for train_shard in train_replica.shards:
                train_transfer_rank = train_shard.global_rank + self.infer_world_size

                # Skip if filtering by rank and this isn't the right rank
                if global_transfer_rank is not None and global_transfer_rank >= self.infer_world_size:
                    if train_transfer_rank != global_transfer_rank:
                        continue

                # For each inference rank, send the corresponding slice
                for engine_rank in range(self.num_infer_engines):
                    for inf_replica in inference_meta.replicas:
                        # [DEBUG] Log shard details for first expert
                        if expert_id == 0 and "gate_up_proj" in param_name and "layers.0." in param_name:
                            shard_details = [(s.tp_rank, s.global_rank) for s in inf_replica.shards]
                            logger.info(
                                f"[EXPERT_SHARD_DEBUG] {param_name}: "
                                f"global_transfer_rank={global_transfer_rank}, "
                                f"infer_world_size={self.infer_world_size}, "
                                f"shard_details (tp_rank, global_rank)={shard_details}"
                            )
                        for inf_shard in inf_replica.shards:
                            infer_tp_rank = inf_shard.tp_rank
                            infer_transfer_rank = inf_shard.global_rank + engine_rank * self.infer_instance_world_size

                            # Skip if filtering by rank
                            if global_transfer_rank is not None:
                                if global_transfer_rank < self.infer_world_size:
                                    # Inference rank: only include ops where this rank is receiver
                                    if infer_transfer_rank != global_transfer_rank:
                                        continue
                                else:
                                    # Training rank: already filtered above
                                    pass

                            # Check if this is a gate_up_proj (fused gate+up with gated linear unit)
                            # For gate_up_proj:
                            #   Megatron layout: [Gate_all | Up_all] contiguous
                            #   SGLang layout: [Gate_slice | Up_slice] per TP rank
                            # Need to split into TWO transfers
                            is_gate_up = "gate_up_proj" in param_name and sharding_dim == 0

                            if is_gate_up:
                                # For gate_up_proj, need to send Gate and Up slices separately
                                # full_dim_size = 2 * intermediate_size (gate + up)
                                # half_dim = intermediate_size (gate only or up only)
                                # slice_size = full_dim_size / infer_tp_size = 2 * half_dim / TP
                                # half_slice = slice_size / 2 = half_dim / TP
                                half_dim = full_dim_size // 2  # e.g., 768
                                half_slice = slice_size // 2   # e.g., 96

                                # Gate slice: train[r*half_slice : (r+1)*half_slice] -> infer[0:half_slice]
                                gate_train_start = infer_tp_rank * half_slice
                                gate_train_end = gate_train_start + half_slice
                                gate_infer_start = 0
                                gate_infer_end = half_slice

                                plan.append(
                                    CommunicationOperation(
                                        send_rank=train_transfer_rank,
                                        send_shard_meta=train_shard,
                                        send_offset=(gate_train_start, 0),
                                        recv_rank=infer_transfer_rank,
                                        recv_shard_meta=inf_shard,
                                        recv_offset=(gate_infer_start, 0),
                                        overlap_shape=(half_slice, hidden_dim),
                                        train_slices=(slice(gate_train_start, gate_train_end), slice(0, hidden_dim)),
                                        inf_slices=(slice(gate_infer_start, gate_infer_end), slice(0, hidden_dim)),
                                    )
                                )

                                # Up slice: train[half_dim + r*half_slice : half_dim + (r+1)*half_slice] -> infer[half_slice:slice_size]
                                up_train_start = half_dim + infer_tp_rank * half_slice
                                up_train_end = up_train_start + half_slice
                                up_infer_start = half_slice
                                up_infer_end = slice_size

                                plan.append(
                                    CommunicationOperation(
                                        send_rank=train_transfer_rank,
                                        send_shard_meta=train_shard,
                                        send_offset=(up_train_start, 0),
                                        recv_rank=infer_transfer_rank,
                                        recv_shard_meta=inf_shard,
                                        recv_offset=(up_infer_start, 0),
                                        overlap_shape=(half_slice, hidden_dim),
                                        train_slices=(slice(up_train_start, up_train_end), slice(0, hidden_dim)),
                                        inf_slices=(slice(up_infer_start, up_infer_end), slice(0, hidden_dim)),
                                    )
                                )

                                # Log gate_up split for debugging
                                if len(plan) <= 6 or "layers.0." in param_name:
                                    logger.debug(
                                        f"[EXPERT_GATE_UP_SPLIT] {param_name}: "
                                        f"half_dim={half_dim}, half_slice={half_slice}, "
                                        f"gate: train[{gate_train_start}:{gate_train_end}] -> infer[{gate_infer_start}:{gate_infer_end}], "
                                        f"up: train[{up_train_start}:{up_train_end}] -> infer[{up_infer_start}:{up_infer_end}]"
                                    )
                            else:
                                # For down_proj or non-gated params: simple contiguous slice
                                # Calculate slice offsets
                                # Training: send slice [tp_rank * slice_size : (tp_rank + 1) * slice_size]
                                train_slice_start = infer_tp_rank * slice_size
                                train_slice_end = train_slice_start + slice_size

                                # Inference: receive at [0 : slice_size] (the full shard)
                                infer_slice_start = 0
                                infer_slice_end = slice_size

                                # Build slices based on sharding dimension
                                if sharding_dim == 0:
                                    # Non-gated column parallel: shape (intermediate_size, hidden_size)
                                    train_slices = (slice(train_slice_start, train_slice_end), slice(0, hidden_dim))
                                    inf_slices = (slice(infer_slice_start, infer_slice_end), slice(0, hidden_dim))
                                    overlap_shape = (slice_size, hidden_dim)
                                    train_offset = (train_slice_start, 0)
                                    infer_offset = (infer_slice_start, 0)
                                else:
                                    # down_proj: shape (hidden_size, intermediate_size)
                                    train_slices = (slice(0, hidden_dim), slice(train_slice_start, train_slice_end))
                                    inf_slices = (slice(0, hidden_dim), slice(infer_slice_start, infer_slice_end))
                                    overlap_shape = (hidden_dim, slice_size)
                                    train_offset = (0, train_slice_start)
                                    infer_offset = (0, infer_slice_start)

                                plan.append(
                                    CommunicationOperation(
                                        send_rank=train_transfer_rank,
                                        send_shard_meta=train_shard,
                                        send_offset=train_offset,
                                        recv_rank=infer_transfer_rank,
                                        recv_shard_meta=inf_shard,
                                        recv_offset=infer_offset,
                                        overlap_shape=overlap_shape,
                                        train_slices=train_slices,
                                        inf_slices=inf_slices,
                                    )
                                )

                                # Log for down_proj debugging
                                if len(plan) <= 3 or "layers.0." in param_name:
                                    logger.debug(
                                        f"[EXPERT_OP] {param_name}: "
                                        f"train_rank={train_transfer_rank} (global={train_shard.global_rank}) -> "
                                        f"infer_rank={infer_transfer_rank} (tp={infer_tp_rank}), "
                                        f"train_slice=[{train_slice_start}:{train_slice_end}] -> "
                                        f"infer_slice=[{infer_slice_start}:{infer_slice_end}], "
                                        f"overlap_shape={overlap_shape}"
                                    )

        logger.info(f"Expert EP→TP mapping generated {len(plan)} operations for {param_name}")

        # [EXPERT_TRANSFER_MAP] Log detailed transfer mapping for first expert of first layer
        if "layers.0." in param_name and "gate_up_proj" in param_name and expert_id < 16:
            # Group operations by sender rank to show all-to-all pattern
            sender_to_receivers = {}
            for op in plan:
                sender = op.send_rank
                receiver = op.recv_rank
                if sender not in sender_to_receivers:
                    sender_to_receivers[sender] = []
                sender_to_receivers[sender].append({
                    'recv_rank': receiver,
                    'send_offset': op.send_offset,
                    'recv_offset': op.recv_offset,
                    'overlap_shape': op.overlap_shape,
                })

            for sender, receivers in sorted(sender_to_receivers.items()):
                recv_ranks = sorted(set(r['recv_rank'] for r in receivers))
                logger.info(
                    f"[EXPERT_TRANSFER_MAP] {param_name}: "
                    f"train_rank={sender} -> infer_ranks={recv_ranks} "
                    f"(total {len(receivers)} ops, shape={receivers[0]['overlap_shape'] if receivers else 'N/A'})"
                )

        return plan

    def _extract_expert_id(self, param_name: str) -> int:
        """Extract expert ID from parameter name like 'mlp.experts.15.gate_up_proj.weight'."""
        import re
        match = re.search(r'experts\.(\d+)\.', param_name)
        if match:
            return int(match.group(1))
        return None

    def build_local_transfer_plan(
        self,
        inference_weights_meta: List[ParameterMeta],
        training_weights_meta: List[ParameterMeta],
        global_transfer_rank: int,
    ) -> TransferPlan:
        is_train = global_transfer_rank >= self.infer_world_size
        name = "train" if is_train else "infer"
        start_time = time.time()
        logger.info(
            f"Rank[{global_transfer_rank}] Building local transfer plan for {name}"
        )
        num_infer_shards = (
            sum(
                len(replica.shards)
                for param in inference_weights_meta
                for replica in param.replicas
            )
            * self.num_infer_engines
        )
        num_train_shards = sum(
            len(replica.shards)
            for param in training_weights_meta
            for replica in param.replicas
        )
        logger.info(
            f"Rank[{global_transfer_rank}] Number of inference shards: {num_infer_shards}, number of training shards: {num_train_shards}"
        )
        total_shards = num_infer_shards + num_train_shards
        prune_threshold = 10000
        if total_shards > prune_threshold:
            logger.info(
                f"Rank[{global_transfer_rank}] Pruning global transfer plan for {name} because number of shards is too large: {total_shards}"
            )
            operations = self.build_weights_mapping_operations(
                inference_weights_meta,
                training_weights_meta,
                global_transfer_rank=global_transfer_rank,
            )
        else:
            logger.info(
                f"Rank[{global_transfer_rank}] Building global transfer plan for {name} because number of shards is small: {total_shards}"
            )
            operations = self.build_weights_mapping_operations(
                inference_weights_meta, training_weights_meta
            )
        if self.enable_debug_mode:
            data = to_dict(operations)
            file_name = f"global_communication_plan_{name}_{global_transfer_rank}_{os.getpid()}.json"
            with open(file_name, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(
                f"Rank[{global_transfer_rank}] Saved communication plan to {os.path.abspath(file_name)}"
            )

        # Filter operations for this specific rank
        new_operations = []
        for operation in operations:
            if is_train:
                # For training ranks, only include operations where this rank is the sender
                if operation.send_rank == global_transfer_rank:
                    new_operations.append(operation)
            else:
                # For inference ranks, only include operations where this rank is the receiver
                if operation.recv_rank == global_transfer_rank:
                    new_operations.append(operation)

        # Group operations by the opposite rank (the rank we communicate with)
        # For training ranks: group by recv_rank (the inference rank we send to)
        # For inference ranks: group by send_rank (the training rank we receive from)
        key = "recv_rank" if is_train else "send_rank"
        grouped_operations = self._group_operations_by_rank(new_operations, key)

        # Validate that all operations in the grouped plan are for this specific rank
        for ops in grouped_operations.values():
            for op in ops:
                if is_train:
                    assert op.send_rank == global_transfer_rank, (
                        f"Training rank {global_transfer_rank} local plan contains operation "
                        f"with send_rank {op.send_rank} instead of {global_transfer_rank}"
                    )
                else:
                    assert op.recv_rank == global_transfer_rank, (
                        f"Inference rank {global_transfer_rank} local plan contains operation "
                        f"with recv_rank {op.recv_rank} instead of {global_transfer_rank}"
                    )
        duration = time.time() - start_time
        logger.info(
            f"Rank[{global_transfer_rank}] ({'training' if is_train else 'inference'}) "
            f"local plan has {len(new_operations)} operations grouped by {len(grouped_operations)} "
            f"opposite ranks: {list(grouped_operations.keys())}, "
            f"global operations across all ranks: {len(operations)}, "
            f"took time: {duration:.2f} seconds"
        )

        if self.enable_debug_mode:
            data = to_dict(grouped_operations)
            file_name = f"local_communication_plan_{name}_{global_transfer_rank}_{os.getpid()}.json"
            with open(file_name, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(
                f"Rank[{global_transfer_rank}] Saved communication plan to {os.path.abspath(file_name)}"
            )
        return TransferPlan(operations=grouped_operations)


@torch.no_grad()
def slice_tensor(
    tensor: torch.Tensor, op: CommunicationOperation, is_train, **kwargs
) -> torch.Tensor:
    """
    Slice the overlapping region from the source tensor.

    Args:
        tensor: The source tensor to slice

    Returns:
        The sliced tensor containing only the overlapping region
    """
    slices = op.train_slices if is_train else op.inf_slices
    sliced_tensor = tensor[slices]
    if not sliced_tensor.is_contiguous():
        param_name = op.send_shard_meta.name
        source_offset = op.send_offset if is_train else op.recv_offset
        if is_train:
            slice_context = kwargs.get("slice_context", {})
            key = f"{param_name}-{slices}"
            sliced = slice_context.get(key)
            if sliced is not None:
                return sliced
            sliced = sliced_tensor.contiguous()
            slice_context[key] = sliced
            return sliced
        else:
            msg = (
                f"Sliced tensor is not contiguous, param_name: {param_name}, "
                f"inference_meta: {op.recv_shard_meta}, training_meta: {op.send_shard_meta}, "
                f"source_offset: {source_offset}, overlap_shape: {op.overlap_shape}, "
                f"tensor shape: {tensor.shape}, slices: {slices}, contiguous: {tensor.is_contiguous()}"
            )
            raise ValueError(msg)
    return sliced_tensor
