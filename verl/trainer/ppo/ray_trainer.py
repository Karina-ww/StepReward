# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import uuid
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Type, Dict, List
from omegaconf import ListConfig
import numpy as np
import ray
import torch
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, RandomSampler, SequentialSampler
from tqdm import tqdm
import time
import copy
import json
import torch.nn.functional as F

from verl.utils.model import compute_position_id_with_mask
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
    reduce_metrics,
)
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger

WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """

    GAE = "gae"
    GAE_STEP = "gae_step"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REINFORCE_PLUS_PLUS_BASELINE = "reinforce_plus_plus_baseline"
    REMAX = "remax"
    RLOO = "rloo"
    OPO = "opo"
    GPO = "gpo"


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {node: node_info.get("GPU", 0) for node, node_info in node_available_resources.items()}

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )

        # check each resource pool can be satisfied, O(#resource_pools * #nodes)
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(
                    f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes}"
                    + "cannot be satisfied in this ray cluster"
                )


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    responses = data.batch["responses"]
    response_length = responses.size(1)
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch["attention_mask"]
    response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=0.95, num_repeat=1, norm_adv_by_std_in_grpo=True, actor_rollout_wg=None):
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == AdvantageEstimator.GAE:
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            eos_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.GAE_STEP:
        advantages, returns = core_algos.compute_gae_step_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            eos_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.GRPO:
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            eos_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE:
        advantages, returns = core_algos.compute_reinforce_plus_plus_baseline_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            eos_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            eos_mask=data.batch["response_mask"],
            gamma=gamma,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REMAX:
        advantages, returns = core_algos.compute_remax_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            reward_baselines=data.batch["reward_baselines"],
            eos_mask=data.batch["response_mask"],
        )

        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.RLOO:
        advantages, returns = core_algos.compute_rloo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.OPO:
        advantages, returns = core_algos.compute_opo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            eos_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            value_baseline="optimal"      
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        raise NotImplementedError
    return data


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    if name not in timing_raw:
        timing_raw[name] = 0
    timing_raw[name] += timer.last


class RayPPOTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        reward_fn=None,
        val_reward_fn=None,
    ):
        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.validation_generations_logger = ValidationGenerationsLogger()

        # define in-reward KL control
        # kl loss control currently not suppoorted
        # if not self.config.actor_rollout_ref.actor.get('use_kl_loss', False):
        #     self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)
        if self.use_reference_policy:
            if config.algorithm.kl_ctrl.type == 'fixed':
                self.kl_ctrl = core_algos.FixedKLController(kl_coef=config.algorithm.kl_ctrl.kl_coef)
            elif config.algorithm.kl_ctrl.type == 'adaptive':
                assert config.algorithm.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
                self.kl_ctrl = core_algos.AdaptiveKLController(init_kl_coef=config.algorithm.kl_ctrl.kl_coef,
                                                               target_kl=config.algorithm.kl_ctrl.target_kl,
                                                               horizon=config.algorithm.kl_ctrl.horizon)
            else:
                raise NotImplementedError
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.)


        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
            AdvantageEstimator.GRPO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS,
            AdvantageEstimator.REMAX,
            AdvantageEstimator.RLOO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
            AdvantageEstimator.OPO,
            AdvantageEstimator.GPO,
            AdvantageEstimator.GAE_STEP,
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_config()
        self._create_dataloader()

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % n_gpus == 0, (
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."
        )

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            settings = {
                "actor_rollout_ref.actor": "micro_batch_size",
                "critic": "micro_batch_size",
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(
                        f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'."
                    )

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(
                        f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. "
                        f"Please remove '{name}.{param}' because only '*_{param_per_gpu}'"
                        + "is supported (the former is deprecated)."
                    )

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.actor.ppo_micro_batch_size,
                config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                "actor_rollout_ref.actor",
            )

            if self.use_reference_policy:
                # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
                check_mutually_exclusive(
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                    "actor_rollout_ref.ref",
                )

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.rollout",
            )

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(
                config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu, "critic"
            )

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(
                config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu, "reward_model"
            )

        # Actor
        # check if train_batch_size is larger than ppo_mini_batch_size
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size
            sp_size = config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert (
                    config.actor_rollout_ref.actor.ppo_mini_batch_size
                    % config.actor_rollout_ref.actor.ppo_micro_batch_size
                    == 0
                )
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        assert config.actor_rollout_ref.actor.loss_agg_mode in [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ], f"Invalid loss_agg_mode: {config.actor_rollout_ref.actor.loss_agg_mode}"

        # if config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
        #     print("NOTICE: You have both enabled in-reward kl and kl loss.")
        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size
            sp_size = config.critic.get("ulysses_sequence_parallel_size", 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == "fsdp" and (
            config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1) > 1
            or config.actor_rollout_ref.ref.get("ulysses_sequence_parallel_size", 1) > 1
        ):
            assert config.actor_rollout_ref.model.use_remove_padding, (
                "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."
            )

        if self.use_critic and config.critic.strategy == "fsdp":
            if config.critic.get("ulysses_sequence_parallel_size", 1) > 1:
                assert config.critic.model.use_remove_padding, (
                    "When using sequence parallelism for critic, you must enable `use_remove_padding`."
                )

        if config.data.get("val_batch_size", None) is not None:
            print(
                "WARNING: val_batch_size is deprecated."
                + " Validation datasets are sent to inference engines as a whole batch,"
                + " which will schedule the memory themselves."
            )

        # check eval config
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, (
                "validation gen temperature should be greater than 0 when enabling do_sample"
            )

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self):
        from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
        # TODO: we have to make sure the batch size is divisible by the dp size


        self.train_dataset = RLHFDataset(parquet_files=self.config.data.train_files,
                                         tokenizer=self.tokenizer,
                                         prompt_key=self.config.data.prompt_key,
                                         max_prompt_length=self.config.data.max_prompt_length,
                                         filter_prompts=True,
                                         return_raw_chat=self.config.data.get('return_raw_chat', False),
                                         truncation='error')
        # use sampler for better ckpt resume
        if self.config.data.shuffle:
            print(f"We shuffle the training data...")
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.get('seed', 1))
            sampler = RandomSampler(data_source=self.train_dataset, generator=train_dataloader_generator)
        else:
            print(f"We do not shuffle the training data...")
            sampler = SequentialSampler(data_source=self.train_dataset)

        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                           batch_size=self.config.data.train_batch_size,
                                           drop_last=True,
                                           collate_fn=collate_fn,
                                           sampler=sampler)
        print(f"{self.config.data.val_files=} {type(self.config.data.val_files)=}")
        if not isinstance(self.config.data.val_files, (List, ListConfig)):
            parquet_files = [self.config.data.val_files]
        else:
            parquet_files = self.config.data.val_files

        parquet_files = copy.deepcopy(parquet_files)
        # self.config.data.val_files = self.config.data.val_files if isinstance(self.config.data.val_files, list) else [self.config.data.val_files]
        self.val_datasets = []
        self.val_dataloaders = []
        self.val_names = []
        for idx, val_file in enumerate(parquet_files):
            print(f"Working on {val_file=}")
            # Create dataset for current file
            val_dataset = RLHFDataset(
                parquet_files=val_file,  
                tokenizer=self.tokenizer,
                prompt_key=self.config.data.prompt_key,
                max_prompt_length=self.config.data.get('val_max_prompt_length', 3072),
                filter_prompts=False,
                return_raw_chat=self.config.data.get('return_raw_chat', False),
                truncation='error'
            )
            
            # Create dataloader for current dataset
            val_dataloader = DataLoader(
                dataset=val_dataset,
                batch_size=len(val_dataset),  # Use full dataset size as batch size
                shuffle=False,
                drop_last=False,
                collate_fn=collate_fn
            )
            
            self.val_datasets.append(val_dataset)
            self.val_dataloaders.append(val_dataloader)
            self.val_names.append(os.path.basename(val_file).replace('.', '_'))



        assert len(self.train_dataloader) >= 1
        for i in range(len(self.val_dataloaders)):
            assert len(self.val_dataloaders[i]) >= 1

        print(f'Size of train dataloader: {len(self.train_dataloader)}')
        print(f'Total number of training samples: {len(self.train_dataloader.dataset)}')
        for i, val_dataloader in enumerate(self.val_dataloaders):
            print(f'Size of val dataloader {i+1}/{len(self.val_dataloaders)}: {len(val_dataloader)}')

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps


    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = 10

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []
        for idx, val_dataloader in enumerate(self.val_dataloaders):
            for test_data in val_dataloader:
                test_batch = DataProto.from_single_dict(test_data) # RZ: The validation dataset has only one batch.

                # repeat test batch
                test_batch = test_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
                )

                # we only do validation on rule-based rm
                if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                    return {}

                # Store original inputs
                input_ids = test_batch.batch["input_ids"] # RZ: tensor. shape = bsz * seq length.
                # TODO: Can we keep special tokens except for padding tokens?
                input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
                sample_inputs.extend(input_texts) 

                test_gen_batch = test_batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=["raw_prompt_ids"],
                ) 
                test_gen_batch.meta_info = {
                    "eos_token_id": self.tokenizer.eos_token_id,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "recompute_log_prob": False,
                    "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                    "validate": True,
                }
                print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

                # pad to be divisible by dp_size
                test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded) 

                # unpad
                test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
                print("validation generation end")

                # Store generated outputs
                output_ids = test_output_gen_batch.batch["responses"]
                output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
                sample_outputs.extend(output_texts)

                test_batch = test_batch.union(test_output_gen_batch)

                # evaluate using reward_function
                result = self.val_reward_fn(test_batch, return_dict=True) # RZ: If we use default reward fuxtion for DAPO or AIME, the reward is the rule-based reward + length penalty.
                reward_tensor = result["reward_tensor"] # RZ: tensor. Shape = bsz * response's length.
                scores = reward_tensor.sum(-1).cpu().tolist() # RZ: length = bsz. In AIME, we already repeat every question for 30 times so here is 960.
                sample_scores.extend(scores)

                reward_extra_infos_dict["reward"].extend(scores)
                if "reward_extra_info" in result:
                    for key, lst in result["reward_extra_info"].items(): # RZ: keys = score, acc, pred
                        reward_extra_infos_dict[key].extend(lst)

                data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict) 
        # RZ: defaultdict(lambda: defaultdict(lambda: defaultdict(list))). data source ('math_dapo') -> variable ('acc', 'reward') -> metric ('mean@32', 'maj@32', 'best@32') -> value (list of float).
        metric_dict = {}

        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val
        return metric_dict

    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy], config=self.config.actor_rollout_ref, role="ref"
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls, **wg_kwargs
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

    def _save_checkpoint(self):
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir,
                                                f'global_step_{self.global_steps}')
        actor_local_path = os.path.join(local_global_step_folder, 'actor')

        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
            self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'actor')
        self.actor_rollout_wg.save_checkpoint(actor_local_path,
                                              actor_remote_path,
                                              self.global_steps,
                                              remove_previous_ckpt=self.config.trainer.remove_previous_ckpt_in_save)

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, 'critic')
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
                self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'critic')
            self.critic_wg.save_checkpoint(critic_local_path,
                                           critic_remote_path,
                                           self.global_steps,
                                           remove_previous_ckpt=self.config.trainer.remove_previous_ckpt_in_save)

        # save dataloader
        dataloader_local_path = os.path.join(local_global_step_folder, 'data.pt')
        import dill
        torch.save(self.train_dataloader, dataloader_local_path, pickle_module=dill)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir,
                                                           'latest_checkpointed_iteration.txt')
        with open(local_latest_checkpointed_iteration, 'w') as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.training_start_time = time.time()
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                batch: DataProto = DataProto.from_single_dict(batch_dict)
                # print("----batch----", batch)
                # pop those keys for generation # RZ: exclude the multi-modal inputs.
                gen_batch = batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=["raw_prompt_ids"],
                )
                is_last_step = self.global_steps >= self.total_training_steps

                with _timer("step", timing_raw):
                    # generate a batch
                    with _timer("gen", timing_raw):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with _timer("gen_max", timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                    )
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    batch.batch["response_mask"] = compute_response_mask(batch)
                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # recompute old_log_probs
                    with _timer("old_log_prob", timing_raw):

                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropy"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_loss = agg_loss(
                            loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode
                        )
                        old_log_prob_metrics = {"actor/entropy_loss": entropy_loss.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        ########################
                        prompt_len = batch.batch['prompts'].size(1)
                        resp_len = batch.batch['responses'].size(1)

                        
                        ########################
                        old_log_prob.batch.pop("entropy")
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer("ref", timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer("adv", timing_raw):
                        if self.config.algorithm.get("use_step_prob_reward", False):
                            k = int(self.config.algorithm.get("step_prob_reward_k", 4))
                            min_gap = int(self.config.algorithm.get("step_prob_reward_min_gap", 100))
                            assign_mode = self.config.algorithm.get("step_prob_reward_assign_mode", "segment_last_token")
                            # print("batch.non_tensor_batch", batch.non_tensor_batch.keys())
                            # print("non_tensor_batch", batch.non_tensor_batch)
                            
                            ground_truth_list = []
                            for rm in batch.non_tensor_batch['reward_model']:
                                if 'ground_truth' in rm:
                                    # print("DEBUG rm['ground_truth']:", rm['ground_truth'], type(rm['ground_truth']))
                                    ground_truth_list.append(str(rm['ground_truth']['target']))
                                else:
                                    raise ValueError("No ground_truth")
                            # breakpoint()
                            # 用上面 old_log_prob 段缓存的响应熵做切分
                            scores = self.compute_step_prob_rewards(
                                batch=batch,
                                gen_batch_output=gen_batch_output,
                                ground_truth_list=ground_truth_list,
                                response_entropy=entropys,  # [B, R]
                                k=k, min_gap=min_gap, assign_mode=assign_mode
                            )
                            batch.batch["values"] = scores

                            ### WWW:rule base reward ######
                            reward_result = self.reward_fn(batch, return_dict=True)
                            outcome_r = reward_result["reward_tensor"]
                            extra = reward_result.get("reward_extra_info", {})
                            batch.batch["token_level_scores"] = outcome_r 
                            print("*** token_level_scores ***") 
                            print(outcome_r.sum(dim=-1)[:100])
                            ### WWW:rule base reward ######
                    
                            if not self.config.actor_rollout_ref.actor.get('use_kl_loss', False):
                                batch, kl_metrics = apply_kl_penalty(
                                    batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                                )
                                metrics.update(kl_metrics)
                            else:
                                batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                            batch = compute_advantage(
                                batch,
                                adv_estimator=self.config.algorithm.adv_estimator,
                                gamma=self.config.algorithm.gamma,
                                lam=self.config.algorithm.lam,
                                num_repeat=self.config.actor_rollout_ref.rollout.n
                            )

                        else:
                            if self.use_rm:
                                # we first compute reward model score
                                reward_tensor = self.rm_wg.compute_rm_score(batch)
                                batch = batch.union(reward_tensor)

                            # we combine with rule-based rm
                            reward_extra_infos_dict: dict[str, list]
                            try:
                                reward_result = self.reward_fn(batch, return_dict=True)
                                reward_tensor = reward_result["reward_tensor"]
                                reward_extra_infos_dict = reward_result["reward_extra_info"]
                            except Exception as e:
                                print(f"Error in reward_fn: {e}")
                                reward_tensor = self.reward_fn(batch)
                                reward_extra_infos_dict = {}
                            batch.batch["token_level_scores"] = reward_tensor

                            print(f"{list(reward_extra_infos_dict.keys())=}")
                            if reward_extra_infos_dict:
                                batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                            # compute rewards. apply_kl_penalty if available
                            if not self.config.actor_rollout_ref.actor.get('use_kl_loss', False):
                                batch, kl_metrics = apply_kl_penalty(
                                    batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                                )
                                metrics.update(kl_metrics)
                            else:
                                batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                            # compute advantages, executed on the driver process
                        
                            batch = compute_advantage(
                                batch,
                                adv_estimator=self.config.algorithm.adv_estimator,
                                gamma=self.config.algorithm.gamma,
                                lam=self.config.algorithm.lam,
                                num_repeat=self.config.actor_rollout_ref.rollout.n
                            )

                    # update critic
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer("update_actor", timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # validate
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                    ):
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (
                        is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                    ):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                # collect metrics
                non_training_labels = ['testing', 'save_checkpoint']
                time_pure_training = timing_raw['step']
                for label in non_training_labels:
                    if label in timing_raw.keys():
                        time_pure_training -= timing_raw[label]
                timing_raw['time_pure_training'] = time_pure_training
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))

                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                timing_raw = defaultdict(float)  # clear timing

                batch = None
                num_prompt_in_batch = 0

                # Add total training time to metrics
                current_time = time.time()
                metrics["train/total_training_time_seconds"] = current_time - self.training_start_time
                metrics["train/global_steps"] = self.global_steps
                # metrics['train/num_prompts_in_batch'] = num_prompt_in_batch

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1

    def _pick_k_uniform_snap_peaks(self,
                               entropy_vec: torch.Tensor,
                               valid_len: int,
                               k: int,
                               snap_window: int | None = None,
                               prefer_near_anchor: bool = True) -> list[int]:
        """
        基于“均分锚点 + 附近吸附到高熵”的 k 段分割。
        返回升序段末索引（response 局部坐标），最后一个恒为 L-1。
        在实现上对以下情况做了鲁棒处理：
        - entropy_vec 实际长度 < valid_len
        - 窗口被截断导致局部长度 != (b-a)
        - tie-break 在局部坐标中进行，避免长度不一致
        """
        # 展平并确定真正可用长度
        e_full = entropy_vec.detach().flatten()
        e_len = int(e_full.shape[0])
        L = int(valid_len)
        if L <= 0 or k <= 0:
            return []
        if L <= k: # WWW:处理报错特殊情况
            return [0] * k
        k_eff = min(k, L)
        # 等间距段末锚点（含最后一个 L-1）
        anchors = [max(0, min(L - 1, round((j * L) / k_eff) - 1)) for j in range(1, k_eff + 1)]
        anchors[-1] = L - 1

        if snap_window is None:
            snap_window = max(1, L // (4 * max(1, k_eff)))

        cuts: list[int] = []
        taken = set()
        prev = -1

        for t in anchors:
            # 窗口在实际 L 内裁剪
            a = max(0, t - snap_window)
            b = min(L, t + snap_window + 1)
            local = e_full[a:b]              # [local_len]
            local_len = int(local.shape[0])

            if local_len == 0:
                j = t
            else:
                if prefer_near_anchor:
                    # 在局部坐标系进行 tie-break
                    t_local = min(max(t - a, 0), local_len - 1)
                    rel_idx = torch.arange(local_len, device=local.device)
                    dist = (rel_idx - t_local).abs().float() / (snap_window + 1e-6)
                    score = local - 1e-6 * dist
                    off = int(torch.argmax(score).item())
                    j = a + off
                else:
                    off = int(torch.argmax(local).item())
                    j = a + off

            # 单调性与边界修正
            j = int(j)
            if j <= prev:
                j = prev + 1
            if j >= L:
                j = L - 1

            is_last = (t == anchors[-1])
            if (not is_last) and (j == L - 1):
                j = max(prev + 1, L - 2)

            # 避免重复与非严格递增
            while (j in taken) or (j <= prev):
                j += 1
                if j >= (L - 1 if not is_last else L):
                    j = min(L - 1, max(prev + 1, j))
                    break

            cuts.append(j)
            taken.add(j)
            prev = j

        # 明确最后一个 cut 是 L-1
        cuts[-1] = L - 1

        # 去重 + 严格递增（极端情况下段数可能少于 k）
        uniq: list[int] = []
        for x in cuts:
            if len(uniq) == 0 or x > uniq[-1]:
                uniq.append(x)
        return uniq
    

    def _build_context_gt_batch(self,
                            gen_batch_output: DataProto,
                            ground_truth_list: list[str],
                            cut_counts: list[list[int]],
                            start_answer='<answer>',
                                end_answer='</answer>') -> DataProto:
        """
        构造 (B*(k+1)) 条样本：prompt + response_prefix + <answer> GT </answer> eos
        利用 replace_answer_with_gt_batch 批量生成，再封装为 DataProto 返回。
        """
        prompts_ids = gen_batch_output.batch['prompts']     # [B, P]
        responses_ids = gen_batch_output.batch['responses'] # [B, R]
        B = prompts_ids.size(0)
        R = responses_ids.size(1)

        pad_id =self.tokenizer.pad_token_id

        merged_prompts, merged_responses, gt_repeated, merged_input_ids = [], [], [], []


        for i in range(B):
            prompt_ids = prompts_ids[i]
            resp_ids = responses_ids[i]
            if all(c == 0 for c in cut_counts[i]):
                for _ in range(k + 1):
                    resp_prefix = resp_ids.clone()
                    input_ids = torch.cat([prompt_ids, resp_prefix], dim=0)
                    
                    merged_prompts.append(prompt_ids.clone())
                    merged_responses.append(resp_prefix)
                    gt_repeated.append(ground_truth_list[i])
                    merged_input_ids.append(input_ids)
            else:
                for c in cut_counts[i]:
                    c = int(c)
                    if c < 0:
                        c = 0
                    elif c > R:
                        c = R

                    # 取前缀并右补到 R
                    resp_prefix = resp_ids[:c]  # [c]
                    pad_right = R - resp_prefix.size(0)
                    if pad_right > 0:
                        resp_prefix = F.pad(resp_prefix, (0, pad_right), value=pad_id)  # -> [R]
                    
                    input_ids = torch.cat([prompt_ids, resp_prefix], dim=0)

                    merged_prompts.append(prompt_ids.clone())   # [P]
                    merged_responses.append(resp_prefix)        # [R]
                    gt_repeated.append(ground_truth_list[i])
                    merged_input_ids.append(input_ids)  

        
        stub_batch = DataProto.from_single_dict({
            'prompts': torch.stack(merged_prompts, dim=0),
            'responses': torch.stack(merged_responses, dim=0),
            'input_ids': torch.stack(merged_input_ids, dim=0), 
        })

        # 真正构造 “prompt + 前缀 + <answer> GT </answer> eos”
        batch_results = self.replace_answer_with_gt_batch(
            gen_ids_batch=stub_batch.batch['input_ids'],       # dummy, 仅用于 batch 维度
            gen_response_batch=stub_batch.batch['responses'],
            ground_truth_batch=gt_repeated,
            prompts_batch_shape=prompts_ids.shape[-1],
            eos_token_str=self.tokenizer.eos_token,
            pad_token_str=self.tokenizer.pad_token,
            max_length=self.config.data.max_prompt_length + self.config.data.max_response_length,
            suffix=''  # 使用默认字段名
        )

        # 封装为 DataProto
        pr_batch = DataProto.from_single_dict(batch_results)
        return pr_batch

    def compute_step_prob_rewards(self,
                              batch: DataProto,
                              gen_batch_output: DataProto,
                              ground_truth_list: list[str],
                              response_entropy: torch.Tensor,
                              k: int = 4,
                              min_gap: int = 16,
                              assign_mode: str = 'segment_last_token') -> torch.Tensor:
        """
        计算 step-level probability reward，返回与 batch.input_ids 对齐的 token_level_scores (float32)。

        参数：
        - response_entropy: [B, R] 的响应部分逐 token 熵（仅响应段）
        """
        device = batch.batch['input_ids'].device
        B = gen_batch_output.batch['responses'].size(0)
        prompt_len = gen_batch_output.batch['prompts'].size(1)
        resp_len   = gen_batch_output.batch['responses'].size(1)

        # 有效长度（部分响应可能被 padding）
        attn_resp  = gen_batch_output.batch['attention_mask'][:, -resp_len:]
        valid_lens = attn_resp.sum(-1).tolist()  # List[int], <= resp_len


        # 1) 选切点（以响应局部索引计），再转成“前缀 token 数”
        cut_counts: list[list[int]] = []
        for i in range(B):
            ends = self._pick_k_uniform_snap_peaks(
                entropy_vec=response_entropy[i],
                valid_len=int(valid_lens[i]),
                k=k,                      
                snap_window=None,         
                prefer_near_anchor=True   
            )
            cut_counts.append([0] + [e + 1 for e in ends]) 
        print("====cut_counts====",cut_counts)
        # 2) 构造 (B*(k+1)) 的 batch（每个样本包含 V0..Vk 的上下文）

        pr_batch = self._build_context_gt_batch(
            gen_batch_output=gen_batch_output,
            ground_truth_list=ground_truth_list,
            cut_counts=cut_counts
        )
        # print("**** pr_batch ****", pr_batch)
        # 3) 前向一次得到每条上下文的 GT token 对数概率均值 => V_j
        pr_old = self.actor_rollout_wg.compute_log_prob(pr_batch)
 
        log_probs = pr_old.batch['old_log_probs'].to(device)        # [B*(k+1), L]
        gt_mask   = pr_batch.batch['ground_truth_mask'].to(device)  # [B*(k+1), R_mask]
        # V: [B*(k+1)]
        V = (torch.exp(log_probs) * gt_mask).sum(-1) / (gt_mask.sum(-1) + 1e-8)

        Vs: list[list[float]] = [] ### WWW: [bs*(k+1)] -> [bs, k+1]
        idx = 0
        for i in range(B):
            cnt = len(cut_counts[i])
            Vs.append(V[idx:idx+cnt].tolist())
            idx += cnt
        print(f"*** STEP-Level Value:{Vs[:200]} ***")
        # 4) 回填 token-level scores 到原 batch.input_ids 对齐的二维张量 -->  WWW: token-level VALUE
        scores = torch.zeros_like(batch.batch['responses'], dtype=torch.float32, device=device)
        for i in range(B):
            # r_j = V_j - V_{j-1}, j = 1..k
            if all(c == 0 for c in cut_counts[i]):
                continue
            rs = [Vs[i][j] - Vs[i][j-1] for j in range(1, len(Vs[i]))]  # 长度 k
            seg_ends = [c - 1 for c in cut_counts[i][1:]]  # 响应局部索引
            prev = 0
            for r, end_idx in zip(rs, seg_ends):
                start =  prev
                end   =  end_idx + 1
                length = max(1, end - start)
                scores[i, start:end] = r 
                prev = end_idx + 1
        print("final scores", scores.size(), scores)
        # breakpoint()
        return scores



    def replace_answer_with_gt_batch(
        self,
        gen_ids_batch,
        gen_response_batch,
        ground_truth_batch,
        prompts_batch_shape,
        eos_token_str,
        pad_token_str,
        max_length,
        suffix
    ):
        batch_size = len(gen_ids_batch)

        gen_texts = self.tokenizer.batch_decode(gen_ids_batch, skip_special_tokens=False)
        gen_response_texts = self.tokenizer.batch_decode(gen_response_batch, skip_special_tokens=False)
        gen_texts = [replace_left_and_right_str(text, pad_token_str) for text in gen_texts]
        gen_response_texts = [replace_left_and_right_str(text, pad_token_str) for text in gen_response_texts]

        new_texts = []
        all_new_responses = []
        # total_ground_truth = []
        for i in range(batch_size):
            full_text = gen_texts[i]
            response_text = gen_response_texts[i].rstrip(eos_token_str)
            gt = ground_truth_batch[i].strip()
            if len(gt) < 3:  
                boxed_gt = f" \\boxed{{{ gt }}} "  
            else:
                boxed_gt = f"\\boxed{{{gt}}}"
            # total_ground_truth.append({"raw_gt": gt, "boxed_gt": boxed_gt})
            

            old_boxed, _ = extract_boxed_expr(response_text)
            if old_boxed:
                # print(f"[DEBUG] Found old boxed: {old_boxed}, replacing with: {boxed_gt}")
                new_response_text = response_text.replace(old_boxed, boxed_gt)
            else:
                # 控制拼接后的 response 长度不要超限
                response_ids = self.tokenizer.encode(response_text, add_special_tokens=False)
                boxed_gt_ids = self.tokenizer.encode(" " + boxed_gt, add_special_tokens=False)
                total_len = len(response_ids) + len(boxed_gt_ids)
                if total_len > self.config.data.max_response_length:
                    allowed_len = self.config.data.max_response_length - len(boxed_gt_ids)
                    response_ids = response_ids[:allowed_len]
                    response_text = self.tokenizer.decode(response_ids, skip_special_tokens=False)
                
                new_response_text = response_text + " " + boxed_gt
            
            # 拼接成完整新文本（替换原来的 response）
            assistant_tag = "<|im_start|>assistant\n"
            if assistant_tag not in full_text:
                raise ValueError("Missing assistant tag in generated full text.")
            split_pos = full_text.rfind(assistant_tag) + len(assistant_tag)
            new_text = full_text[:split_pos] + new_response_text + eos_token_str
            new_texts.append(new_text)
            all_new_responses.append(new_response_text)

        # print("=== Ground Truth Samples ===")
        # print(total_ground_truth)
        # --- Tokenize new_texts and build tensors ---
        batch_input_data = self.tokenizer(
            new_texts, return_tensors='pt', add_special_tokens=False,
            padding=True, truncation=True,
            max_length=max_length + self.config.data.max_response_length
        )
        batch_input_ids = batch_input_data['input_ids']
        batch_attention_mask = batch_input_data['attention_mask']

        # --- 定位 assistant 作为 response 起点 ---
        pos_start_resp_batch = self.batch_locate_substring_tokens(new_texts, "<|im_start|>assistant\n")

        # --- 定位 GT 的 boxed 出现位置（包含完整的\boxed{}）---
        pos_gt_batch = self.batch_locate_substring_tokens(new_texts, [f"\\boxed{{{gt}}}" for gt in ground_truth_batch])

        # --- 构建 batch 数据 ---
        results = {
            f'prompts{suffix}': [],
            f'responses{suffix}': [],
            f'input_ids{suffix}': [],
            f'attention_mask{suffix}': [],
            f'position_ids{suffix}': [],
            f'ground_truth_mask{suffix}': [],
        }

        # 预计算boxed前缀和后缀的token长度
        boxed_prefix = "\\boxed{"
        prefix_ids = self.tokenizer.encode(boxed_prefix, add_special_tokens=False)
        prefix_len = len(prefix_ids)
        
        boxed_suffix = "}"
        suffix_ids = self.tokenizer.encode(boxed_suffix, add_special_tokens=False)
        suffix_len = len(suffix_ids)
        bad_indices = []
        bad_responses = []
        for i in range(batch_size):
            input_ids = batch_input_ids[i]
            attn_mask = batch_attention_mask[i]
            pos_resp_tok = pos_start_resp_batch[i][-1] + 1
            pos_gt = pos_gt_batch[i]

            if not pos_gt:
                raise ValueError("Failed to locate ground truth box.")

            # 原始boxed的位置（包含前缀和后缀）
            boxed_start, boxed_end = pos_gt[0], pos_gt[-1]
            
            # # 计算实际GT内容的位置（排除\boxed{}部分）
            gt_start = boxed_start + prefix_len
            gt_end = boxed_end - suffix_len

           
            # # 验证GT位置的有效性
            # if gt_start > gt_end:
            #     raise ValueError(f"GT position error in sample {i}: start ({gt_start}) > end ({gt_end})")
            if not pos_gt:
                print(f"[WARN] Failed to locate ground truth box for sample {i}, masking as all zeros.")
                bad_indices.append(i)
                gt_start, gt_end = None, None
            else:
                boxed_start, boxed_end = pos_gt[0], pos_gt[-1]
                gt_start = boxed_start + prefix_len
                gt_end = boxed_end - suffix_len

                if gt_start > gt_end:
                    print(f"[WARN] GT position error in sample {i}: start ({gt_start}) > end ({gt_end}), masking as all zeros.GT:{ground_truth_batch[i].strip()},{ground_truth_batch[i]}")
                    bad_indices.append(i)
                    bad_responses.append(all_new_responses[i])
                    gt_start, gt_end = None, None

            prompts = input_ids[:pos_resp_tok]
            responses = input_ids[pos_resp_tok:]

            left_pad_tuple = (max_length - prompts.shape[0], 0)
            right_pad_tuple = (0, self.config.data.max_response_length - responses.shape[0])

            # Truncate / pad
            prompts = F.pad(prompts, left_pad_tuple,'constant', value=self.tokenizer.pad_token_id)
            responses =F.pad(responses, right_pad_tuple, 'constant', self.tokenizer.pad_token_id)

            full_input_ids = torch.cat([prompts, responses], dim=0)
            full_attention_mask = F.pad(attn_mask, right_pad_tuple, 'constant', 0)
            full_attention_mask = F.pad(full_attention_mask, left_pad_tuple, 'constant', 0)
            # position_ids = compute_position_id_with_mask(full_attention_mask.unsqueeze(0)).squeeze(0)
            position_ids = compute_position_id_with_mask(F.pad(attn_mask, right_pad_tuple, 'constant', 1))
            position_ids = F.pad(position_ids, left_pad_tuple, 'constant', 0)

            # 计算 ground_truth_mask（仅包含实际答案部分）
            ground_truth_mask = torch.zeros_like(responses)
            if gt_start is not None and gt_end is not None:
                gt_relative_start = max(0, gt_start - pos_resp_tok)
                gt_relative_end = min(len(ground_truth_mask) - 1, gt_end - pos_resp_tok)
                ground_truth_mask[gt_relative_start:gt_relative_end + 1] = 1
            # Append to result
            results[f'prompts{suffix}'].append(prompts)
            results[f'responses{suffix}'].append(responses)
            results[f'input_ids{suffix}'].append(full_input_ids)
            results[f'attention_mask{suffix}'].append(full_attention_mask)
            results[f'position_ids{suffix}'].append(position_ids)
            results[f'ground_truth_mask{suffix}'].append(ground_truth_mask)

        if bad_indices:
            print("\n=== Bad Indices and Corresponding Responses ===")
            for idx, resp in zip(bad_indices, bad_responses):
                print(f"Index {idx}:\nResponse: {resp}\n{'-'*50}")
        for k in results:
            results[k] = torch.stack(results[k])
        return results
    


    def batch_locate_substring_tokens(self, full_strings, substrings, ignore_end_text=None):
        """
        Locates the token IDs and positions corresponding to a substring in a full string.
        Args:
            full_string (List[str]): The full string to tokenize.
            substring (List[str]): The substring to locate in the full string.
            tokenizer_name (List[str]): The name of the tokenizer to use (default is "gpt2").
        """
        # Tokenize the full string and get byte-level offsets
        batch_encodings = self.tokenizer(full_strings, return_offsets_mapping=True, add_special_tokens=False)
        batch_offsets = batch_encodings["offset_mapping"]  # List of (start, end) byte positions for each token
        # Find the byte-level start and end positions of the substring in the full string
        batch_matching_token_indices = []
        for string_idx in range(len(full_strings)):
            full_string = full_strings[string_idx]
            if isinstance(substrings, str):
                # print(f"full_string: {full_string},substrings:{substrings} ")
                substring = substrings
            else:
                substring = substrings[string_idx]
            offsets = batch_offsets[string_idx]
            if ignore_end_text is not None:
                assert full_string.endswith(
                    ignore_end_text), f"{full_string=} given but {ignore_end_text=} not in the end of the full string"
                sub_start = full_string[:-len(ignore_end_text)].rfind(substring)
            else:
                sub_start = full_string.rfind(substring)
            if sub_start == -1:
                print(f"{full_string=}")
                raise ValueError(f"Substring `{substring}` not found in the full string.")
            sub_end = sub_start + len(substring)
            # Locate the tokens that overlap with the substring's byte range
            matching_token_indices = [
                i for i, (start, end) in enumerate(offsets)
                if start < sub_end and end > sub_start
            ]
            batch_matching_token_indices.append(matching_token_indices)
        return batch_matching_token_indices


    

def replace_left_and_right_str(text, left_str):
    while text.startswith(left_str):
        text = text[len(left_str):]
    while text.endswith(left_str):
        text = text[:-len(left_str)]
    return text

def extract_boxed_expr(s):
    """从字符串中提取第一个合法的 \boxed{...} 表达式，返回完整字符串和内容"""
    idx = s.find(r'\boxed{')
    if idx == -1:
        return None, None
    i = idx + len(r'\boxed{')
    stack = 1
    content = ''
    while i < len(s):
        if s[i] == '{':
            stack += 1
            content += s[i]
        elif s[i] == '}':
            stack -= 1
            if stack == 0:
                break
            content += s[i]
        else:
            content += s[i]
        i += 1
    if stack == 0:
        full = r'\boxed{' + content + '}'
        return full, content
    return None, None

def find_subsequence(tokens, subseq):
    """从token列表中查找子序列位置"""
    for i in range(len(tokens) - len(subseq) + 1):
        if tokens[i:i+len(subseq)] == subseq:
            return i, i + len(subseq) - 1
    return None, None