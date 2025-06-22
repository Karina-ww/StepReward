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

import io
import PIL
import re
import os
import uuid
from datetime import datetime
import pandas as pd
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Type, Dict, List
import matplotlib.pyplot as plt
from copy import deepcopy
from collections import defaultdict
import random
from tqdm import tqdm
from omegaconf import ListConfig
import copy
import json

import torch.nn.functional as F

import numpy as np
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask
from verl.utils.reward_score.repetition import detect_repetition_with_hash



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
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes,
                                            use_gpu=True,
                                            max_colocate_count=1,
                                            name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]


import torch
from verl.utils.torch_functional import masked_mean


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty='kl'):
    responses = data.batch['responses']
    response_length = responses.size(1)
    token_level_scores = data.batch['token_level_scores']
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch['attention_mask']
    response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    if 'ref_log_prob' in data.batch.keys():
        kld = core_algos.kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'],
                                    kl_penalty=kl_penalty)  # (batch_size, response_length)
        kld = kld * response_mask
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch['token_level_rewards'] = token_level_rewards

    metrics = {'critic/kl': current_kl, 'critic/kl_coeff': beta}

    return data, metrics


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1):
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == 'gae':
        values = data.batch['values']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        token_level_rewards = data.batch['token_level_rewards']
        advantages, returns = core_algos.compute_gae_advantage_return(token_level_rewards=token_level_rewards,
                                                                      values=values,
                                                                      eos_mask=response_mask,
                                                                      gamma=gamma,
                                                                      lam=lam)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'grpo':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns, filter_rate, stds = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
        data.batch['final_reward_stds'] = stds
    elif adv_estimator == 'grpo_nostd':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_grpo_nostd_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'reinforce_plus_plus':
        token_level_rewards = data.batch['token_level_rewards']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=token_level_rewards, eos_mask=response_mask, gamma=gamma)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'remax':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]

        reward_baselines = data.batch['reward_baselines']

        advantages, returns = core_algos.compute_remax_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                         reward_baselines=reward_baselines,
                                                                         eos_mask=response_mask)

        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'rloo':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_rloo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    else:
        raise NotImplementedError
    return data, filter_rate


def reduce_metrics(metrics: dict):
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics


def _compute_response_info(batch):
    response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-response_length]
    response_mask = batch.batch['attention_mask'][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def compute_data_metrics(batch, use_critic=True):
    # TODO: add response length
    sequence_score = batch.batch['token_level_scores'].sum(-1)
    sequence_reward = batch.batch['token_level_rewards'].sum(-1)

    advantages = batch.batch['advantages']
    returns = batch.batch['returns']

    max_response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-max_response_length].bool()
    response_mask = batch.batch['attention_mask'][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info['prompt_length']
    response_length = response_info['response_length']

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch['values']
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # score
        'critic/score/mean':
            torch.mean(sequence_score).detach().item(),
        'critic/score/max':
            torch.max(sequence_score).detach().item(),
        'critic/score/min':
            torch.min(sequence_score).detach().item(),
        # reward
        'critic/rewards/mean':
            torch.mean(sequence_reward).detach().item(),
        'critic/rewards/max':
            torch.max(sequence_reward).detach().item(),
        'critic/rewards/min':
            torch.min(sequence_reward).detach().item(),
        # adv
        'critic/advantages/mean':
            torch.mean(valid_adv).detach().item(),
        'critic/advantages/max':
            torch.max(valid_adv).detach().item(),
        'critic/advantages/min':
            torch.min(valid_adv).detach().item(),
        # returns
        'critic/returns/mean':
            torch.mean(valid_returns).detach().item(),
        'critic/returns/max':
            torch.max(valid_returns).detach().item(),
        'critic/returns/min':
            torch.min(valid_returns).detach().item(),
        **({
            # values
            'critic/values/mean': torch.mean(valid_values).detach().item(),
            'critic/values/max': torch.max(valid_values).detach().item(),
            'critic/values/min': torch.min(valid_values).detach().item(),
            # vf explained var
            'critic/vf_explained_var': (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
        } if use_critic else {}),

        # response length
        'response_length/mean':
            torch.mean(response_length).detach().item(),
        'response_length/max':
            torch.max(response_length).detach().item(),
        'response_length/min':
            torch.min(response_length).detach().item(),
        'response_length/clip_ratio':
            torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        # prompt length
        'prompt_length/mean':
            torch.mean(prompt_length).detach().item(),
        'prompt_length/max':
            torch.max(prompt_length).detach().item(),
        'prompt_length/min':
            torch.min(prompt_length).detach().item(),
        'prompt_length/clip_ratio':
            torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }
    return metrics


def compute_timing_metrics(batch, timing_raw):
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info['prompt_length']).item()
    num_response_tokens = torch.sum(response_info['response_length']).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        'gen': num_response_tokens,
        **{
            name: num_overall_tokens for name in ['ref', 'values', 'adv', 'update_critic', 'update_actor']
        },
    }

    return {
        **{
            f'timing_s/{name}': value for name, value in timing_raw.items()
        },
        **{
            f'timing_per_token_ms/{name}': timing_raw[name] * 1000 / num_tokens_of_section[name] for name in set(num_tokens_of_section.keys(
            )) & set(timing_raw.keys())
        },
    }


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last


class RayPPOTrainer(object):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 reward_fn=None,
                 val_reward_fn=None):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
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

        if self.config.algorithm.adv_estimator == 'gae':
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in ['grpo', 'grpo_nostd', 'reinforce_plue_plus', 'remax', 'rloo']:
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
        assert real_train_batch_size % n_gpus == 0, \
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            if mbs is None and mbs_per_gpu is None:
                raise ValueError(f"[{name}] Please set at least one of '{name}.micro_batch_size' or "
                                 f"'{name}.micro_batch_size_per_gpu'.")

            if mbs is not None and mbs_per_gpu is not None:
                raise ValueError(f"[{name}] You have set both '{name}.micro_batch_size' AND "
                                 f"'{name}.micro_batch_size_per_gpu'. Please remove '{name}.micro_batch_size' "
                                 f"because only '*_micro_batch_size_per_gpu' is supported (the former is deprecated).")

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.actor.ppo_micro_batch_size,
                                     config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.actor")

            # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                                     config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.ref")

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                                     config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.rollout")

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu,
                                     "critic")

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu,
                                     "reward_model")

        # Actor
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            sp_size = config.actor_rollout_ref.actor.get('ulysses_sequence_parallel_size', 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            sp_size = config.critic.get('ulysses_sequence_parallel_size', 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == 'fsdp':
            if config.actor_rollout_ref.actor.get('ulysses_sequence_parallel_size', 1) > 1 or \
                    config.actor_rollout_ref.ref.get('ulysses_sequence_parallel_size', 1) > 1:
                assert config.actor_rollout_ref.model.use_remove_padding, \
                    "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

        if self.use_critic and config.critic.strategy == 'fsdp':
            if config.critic.get('ulysses_sequence_parallel_size', 1) > 1:
                assert config.critic.model.use_remove_padding, \
                    "When using sequence parallelism for critic, you must enable `use_remove_padding`."

        if config.data.get('val_batch_size', None) is not None:
            print(
                f"WARNING: val_batch_size is deprecated. Validation datasets are sent to inference engines as a whole batch, which will schedule the memory themselves."
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
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.get('seed', 1))
            sampler = RandomSampler(data_source=self.train_dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(data_source=self.train_dataset)

        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                           batch_size=self.config.data.train_batch_size,
                                           drop_last=True,
                                           collate_fn=collate_fn,
                                           sampler=sampler)

        use_new = True
        if use_new:
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

        else:
            self.val_dataset = RLHFDataset(parquet_files=self.config.data.val_files,
                                        tokenizer=self.tokenizer,
                                        prompt_key=self.config.data.prompt_key,
                                        max_prompt_length=self.config.data.get('val_max_prompt_length', 3072),
                                        filter_prompts=False,
                                        return_raw_chat=self.config.data.get('return_raw_chat', False),
                                        truncation='error')
            self.val_dataloader = DataLoader(
                dataset=self.val_dataset,
                # Validation datasets are sent to inference engines as a whole batch,
                # which will schedule the memory themselves.
                batch_size=len(self.val_dataset),
                # shuffle=True,
                shuffle=False,
                drop_last=False,
                collate_fn=collate_fn)

        assert len(self.train_dataloader) >= 1
        if use_new:
            for i in range(len(self.val_dataloaders)):
                assert len(self.val_dataloaders[i]) >= 1
        else:
            assert len(self.val_dataloader) >= 1

        print(f'Size of train dataloader: {len(self.train_dataloader)}')
        print(f'Total number of training samples: {len(self.train_dataloader.dataset)}')
        if use_new:
            for i, val_dataloader in enumerate(self.val_dataloaders):
                print(f'Size of val dataloader {i+1}/{len(self.val_dataloaders)}: {len(val_dataloader)}')
        else:
            print(f'Size of val dataloader: {len(self.val_dataloader)}')

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

    def _maybe_log_val_generations_to_wandb(self, inputs, outputs, scores, accs, format_rewards, data_sources, 
                                            extracted_answers, ground_truths,
                                            table_attr_name, table_name):
        """Log a table of validation samples to wandb"""

        generations_to_log = self.config.trainer.val_generations_to_log_to_wandb

        if generations_to_log == 0:
            return

        if generations_to_log > 0 and 'wandb' not in self.config.trainer.logger:
            print(
                'WARNING: `val_generations_to_log_to_wandb` is set to a positive value, but no wandb logger is found. ')
            return

        import wandb
        import numpy as np

        grouped_samples = defaultdict(list)
        for sample in zip(data_sources, inputs, outputs, extracted_answers, ground_truths, scores, accs, format_rewards):
            grouped_samples[sample[0]].append(sample)

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)

        # Randomly select `generations_to_log` samples from each data_source
        selected_samples = []
        for data_source, samples in grouped_samples.items():
            # if 'Math-500' in data_source or 'gpqa_diamond' in data_source or 'mmlu_pro' in data_source or 'HellaSwag' in data_source:
            rng.shuffle(samples)
            selected_samples.extend(samples[:generations_to_log])
        samples = selected_samples


        # Create column names for all samples
        columns = ["step"] + sum([[f"{i+1}_data_source", f"{i+1}_inputs", f"{i+1}_outputs", f"{i+1}_extracted_answer", f"{i+1}_ground_truth", f"{i+1}_score", f"{i+1}_acc", f"{i+1}_format_reward"] for i in range(len(samples))], [])

        # if not hasattr(self, 'validation_table'):
        #     # Initialize the table on first call
        #     self.validation_table = wandb.Table(columns=columns)

        if not hasattr(self, table_attr_name):
            # Initialize the table on first call
            setattr(self, table_attr_name, wandb.Table(columns=columns))

        # Create a new table with same columns and existing data
        # Workaround for https://github.com/wandb/wandb/issues/2981#issuecomment-1997445737
        # new_table = wandb.Table(columns=columns, data=self.validation_table.data)
        new_table = wandb.Table(columns=columns, data=getattr(self, table_attr_name).data)


        # Add new row with all data
        row_data = []
        row_data.append(self.global_steps)
        for sample in samples:
            row_data.extend(sample)

        new_table.add_data(*row_data)

        # Update reference and log
        wandb.log({f"val/{table_name}": new_table}, step=self.global_steps)
        setattr(self, table_attr_name, new_table)


    def _maybe_log_train_generations_to_wandb(self, sequences, advantages, scores, scoreAs, scoreBs, 
                                              format_rewards, final_reward_stds, entropies, data_sources, extracted_answers, ground_truths,
                                              table_attr_name, table_name):
        """Log a table of validation samples to wandb"""

        generations_to_log = self.config.trainer.get("train_generations_to_log_to_wandb", 0)

        if generations_to_log == 0:
            return

        if generations_to_log > 0 and 'wandb' not in self.config.trainer.logger:
            print(
                'WARNING: `train_generations_to_log_to_wandb` is set to a positive value, but no wandb logger is found. ')
            return

        import wandb
        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(data_sources, sequences, extracted_answers, ground_truths, advantages, scores, scoreAs, scoreBs, format_rewards, final_reward_stds, entropies))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Create column names for all samples
        # columns = ["step"] + sum([[f"data_source_{i+1}", f"sequence_{i+1}", f"extracted_answer_{i+1}", f"ground_truth_{i+1}", f"advantage_{i+1}", f"score_{i+1}", f"scoreB_{i+1}", f"format_reward_{i+1}", f"entropy_{i+1}"] for i in range(len(samples))], [])
        columns = ["step"] + sum([[f"{i+1}_data_source", f"{i+1}_sequence", f"{i+1}_extracted_answer", f"{i+1}_ground_truth", f"{i+1}_advantage", f"{i+1}_score", f"{i+1}_scoreA", f"{i+1}_scoreB", f"{i+1}_format_reward", f"{i+1}_final_reward_std", f"{i+1}_entropy"] for i in range(len(samples))], [])

        # if not hasattr(self, 'train_table'):
        if not hasattr(self, table_attr_name):
            # Initialize the table on first call
            # getattr(self, table_attr_name) = 
            setattr(self, table_attr_name, wandb.Table(columns=columns))
            # self.train_table = wandb.Table(columns=columns)

        # Create a new table with same columns and existing data
        # Workaround for https://github.com/wandb/wandb/issues/2981#issuecomment-1997445737
        # new_table = wandb.Table(columns=columns, data=self.train_table.data)
        new_table = wandb.Table(columns=columns, data=getattr(self, table_attr_name).data)

        # Add new row with all data
        row_data = []
        row_data.append(self.global_steps)
        for sample in samples:
            row_data.extend(sample)

        new_table.add_data(*row_data)

        # Update reference and log
        wandb.log({f"train/{table_name}": new_table}, step=self.global_steps)
        setattr(self, table_attr_name, new_table)

    def _maybe_log_histogram_to_wandb(self, values, histogram_name, column_name, title, bins=20):
        import wandb

        if len(values) > 10000:
            values = random.sample(values, 10000)

        # data = [[s] for s in values]
        # table = wandb.Table(data=data, columns=[column_name])
        # debug_todo = False
        # if debug_todo:
        #     print(f"{table=}")
        #     print(f"{data=}")
        plt.figure(figsize=(8, 6))
        weights = np.ones_like(values) / len(values) * 100  # Calculate percentages
        plt.hist(values, bins=bins, edgecolor='black', alpha=0.7, weights=weights)
        plt.title(title)
        plt.xlabel("Values")
        plt.ylabel(f"Percentage (%) of {column_name}")
        plt.grid(axis='y', alpha=0.5)
        plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f%%'))
        img_buf = io.BytesIO()
        plt.savefig(img_buf)
        image = PIL.Image.open(img_buf)
        wandb.log({histogram_name + f"_step{self.global_steps}": [wandb.Image(image)]}, step=self.global_steps)
        plt.close()

        
        # Save histogram to /data/logs/val_results
        save_dir = "/data/logs/val_results"
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, histogram_name.split('/')[0]), exist_ok=True)
        
        # Create and save the histogram using matplotlib
        plt.figure()
        plt.hist(values, bins=20)
        plt.title(title + f'_step{self.global_steps}')
        plt.xlabel(column_name)
        plt.ylabel('Frequency')
        
        # Save the plot
        file_name = f"{histogram_name}_step{self.global_steps}.png"
        file_path = os.path.join(save_dir, file_name)
        plt.savefig(file_path)
        plt.close()

    def _validate(self):
        result = {}
        debug = False
        if debug:
            pass
        else:
            result.update(self._validate_inner(decoding_strategy='sampling'))
        result.update(self._validate_inner(decoding_strategy='greedy'))
        return result



    def _validate_inner(self, decoding_strategy):
        acc_tensor_lst = []
        data_source_lst = []
        data_source2count = defaultdict(int)
        data_source_set = set()

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []
        sample_accs = []
        sample_format_rewards = []
        sample_extracted_answers = []
        sample_ground_truths = []
        sample_data_sources = []

        if not hasattr(self, 'val_dataloaders'):
            self.val_dataloaders = [self.val_dataloader]

        for idx, val_dataloader in enumerate(self.val_dataloaders):
            val_name = self.val_names[idx]
            # for test_data in self.val_dataloader:
            for test_data in val_dataloader:
                test_batch = DataProto.from_single_dict(test_data)
                debug = False
                if debug:
                    print(f"{idx=} In validation: {test_batch=}", flush=True) # bs=1021
                    # test_batch = test_batch.slice(list(range(7444)))
                    # print(f"{idx=} After slicing In validation: {test_batch=}", flush=True) # bs=1021
                    print(f"{idx=} {self.actor_rollout_wg.world_size=}")

                # we only do validation on rule-based rm
                if self.config.reward_model.enable and test_batch[0].non_tensor_batch['reward_model']['style'] == 'model':
                    return {}

                # Store original inputs
                input_ids = test_batch.batch['input_ids']
                input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
                sample_inputs.extend(input_texts)

                test_gen_batch = test_batch.pop(['input_ids', 'attention_mask', 'position_ids'])
                if decoding_strategy == 'greedy':
                    test_gen_batch.meta_info = {
                        'eos_token_id': self.tokenizer.eos_token_id,
                        'pad_token_id': self.tokenizer.pad_token_id,
                        'recompute_log_prob': False,
                        'do_sample': False,
                        'validate': True,
                    }
                elif decoding_strategy == 'sampling':
                    test_gen_batch.meta_info = {
                        'eos_token_id': self.tokenizer.eos_token_id,
                        'pad_token_id': self.tokenizer.pad_token_id,
                        'recompute_log_prob': False,
                        'do_sample': True,
                        'validate': True,
                        'n': 1,
                    }
                else:
                    raise ValueError

                print(f"In validation: {test_gen_batch.meta_info=}", flush=True)

                # pad to be divisible by dp_size
                test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)

                debug = False
                if debug:
                    print(f"{test_gen_batch_padded=}")
                    print(f"{pad_size=}")
                    print(f"{self.actor_rollout_wg.world_size=}")
                
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
                test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)


                # test_output_gen_batch.batch['prompts']: contains multiple prompts. Each ids correspond to "'system\\n\\n... We bought three types of gift items for 720 Ft, a total of 20 pieces, with unit prices of 60 Ft, 50 Ft, and 10 Ft. How many pieces of each type did we buy?\\n\\nPresent the answer in LaTex format: \\\\boxed{Your answer}\\nassistant\\n'"

                # Store generated outputs
                output_ids = test_output_gen_batch.batch['responses']
                output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
                sample_outputs.extend(output_texts)
                
                test_batch = test_batch.union(test_output_gen_batch)

                # evaluate using reward_function
                # reward_tensor, acc_tensor, format_reward_tensor = self.val_reward_fn(test_batch)
                # _, acc_tensor, __ = self.val_reward_fn(test_batch)
                reward_tensor, acc_tensor, scoreA_tensor, format_reward_tensor, extracted_answer_list = self.val_reward_fn(test_batch, name=f"{self.config.trainer.experiment_name}-{val_name}-{decoding_strategy}-global_step_{self.global_steps}")

                # Store scores
                scores = reward_tensor.sum(-1).cpu().tolist()
                accs = acc_tensor.sum(-1).cpu().tolist()
                format_rewards = format_reward_tensor.sum(-1).cpu().tolist()
                sample_scores.extend(scores)
                sample_accs.extend(accs)
                sample_format_rewards.extend(format_rewards)
                # print(f"{extracted_answer_list=} {scores=}")
                sample_extracted_answers.extend(extracted_answer_list)
                for i_ in range(len(test_batch)):
                    sample_ground_truths.append(test_batch[i_].non_tensor_batch['reward_model']['ground_truth'])
                    sample_data_sources.append(test_batch[i_].non_tensor_batch['data_source'])

                # reward_tensor_lst.append(reward_tensor)
                acc_tensor_lst.append(acc_tensor)
                # data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))
                data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * acc_tensor.shape[0]))
                for i in range(len(test_batch)):
                    data_source2count[test_batch[i].non_tensor_batch.get('data_source', 'unknown')] += 1
                    data_source_set.add(test_batch[i].non_tensor_batch.get('data_source', 'unknown'))
        print(f"{data_source2count=}")

        self._maybe_log_val_generations_to_wandb(inputs=sample_inputs, outputs=sample_outputs, 
                                                 scores=sample_scores, accs=sample_accs, 
                                                 format_rewards=sample_format_rewards, data_sources=sample_data_sources, 
                                                 extracted_answers=sample_extracted_answers, 
                                                 ground_truths=sample_ground_truths,
                                                 table_attr_name=f'val_table_{decoding_strategy}',
                                                 table_name=f"val_generations_{decoding_strategy}")

 
        # reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        acc_tensor = torch.cat(acc_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)

        # evaluate test_score based on data source
        # data_source_reward = {}
        # for i in range(reward_tensor.shape[0]):
        #     data_source = data_sources[i]
        #     if data_source not in data_source_reward:
        #         data_source_reward[data_source] = []
        #     data_source_reward[data_source].append(reward_tensor[i].item())
        data_source_acc = {}
        for i in range(acc_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_acc:
                data_source_acc[data_source] = []
            data_source_acc[data_source].append(acc_tensor[i].item())



        metric_dict = {}
        for data_source, accs in data_source_acc.items():
            metric_dict[f'val_{decoding_strategy}/test_acc/{data_source}'] = np.mean(accs)

        # Compute overall mean score across all data sources
        if False:
            overall_mean = np.mean(list(metric_dict.values()))
            metric_dict['val/test_acc/overall_mean'] = overall_mean

        # # List of specific data sources to compute a separate mean
        # specific_sources = ['numina_synthetic_math', 'numina_olympiads', 'numina_cn_k12']

        suffixes = ['R1', 'OC', 'SysR1', 'UserR1']  # Add or remove suffixes as needed
        
        data_source_set = list(sorted(set(['-'.join(data_source.split('-')[:-1]) for data_source in data_source_set if any(data_source.endswith(f'-{suffix}') for suffix in suffixes)]))) # set(['-'.join(data_source.split('-')[:-1]) for data_source in data_source_set])
        print(f"{data_source_set=}")
        data_source_not_counted = ['numina_synthetic_math', 'numina_olympiads', 'numina_cn_k12', 'numina_synthetic_amc', 'numina_aops_forum', 'numina_amc_aime']
        for ds in data_source_not_counted:
            if ds in data_source_set:
                data_source_set.remove(ds)

        for suffix in suffixes:
            self.compute_and_store_means(metric_dict, data_source_set, suffix, decoding_strategy)

        # Compute the best accuracy for each data source and store the mean
        acc_list = []
        for val_set in data_source_set:
            for suffixes_log in [['OC', 'SysR1'], ['OC', 'UserR1']]:
                if all(f'val_{decoding_strategy}/test_acc/{val_set}-{suffix}' in metric_dict for suffix in suffixes_log):
                    acc_list.append(max(metric_dict[f'val_{decoding_strategy}/test_acc/{val_set}-{suffix}'] for suffix in suffixes_log))
                if len(acc_list) > 0:
                    metric_dict[f'val_mean_{decoding_strategy}/test_acc/best-mean_{"+".join([source.split("-")[0] for source in data_source_set])}'] = np.mean(acc_list)

        return metric_dict

    def compute_and_store_means(self, metric_dict, data_source_set, suffix, decoding_strategy=None):
        specific_mean = self.compute_specific_means(metric_dict, [f'{source}-{suffix}' for source in data_source_set], decoding_strategy=decoding_strategy)
        if specific_mean is not None:
            if decoding_strategy:
                metric_dict[f'val_mean_{decoding_strategy}/test_acc/{suffix}-mean_{"+".join([source.split("-")[0] for source in data_source_set])}'] = specific_mean
            else:
                metric_dict[f'val_mean/test_acc/{suffix}-mean_{"+".join([source.split("-")[0] for source in data_source_set])}'] = specific_mean

    @staticmethod
    def compute_specific_means(metric_dict, specific_sources, decoding_strategy=None):
        """
        Compute the mean score across specific data sources.

        Args:
            metric_dict (dict): A dictionary containing mean scores for each data source.
            specific_sources (list): A list of specific data source keys to compute the mean for.

        Returns:
            float: The mean score across the specified data sources, or None if none are found.
        """
        # Filter the specific means from the metric_dict
        if decoding_strategy:
            specific_means = [metric_dict[f'val_{decoding_strategy}/test_acc/{source}'] for source in specific_sources if f'val_{decoding_strategy}/test_acc/{source}' in metric_dict]
        else:
            specific_means = [metric_dict[f'val/test_acc/{source}'] for source in specific_sources if f'val/test_acc/{source}' in metric_dict]
        
        # Compute the mean if specific means are found
        if len(specific_means) == len(specific_sources):
            return np.mean(specific_means)
        else:
            return None

    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout],
                                                     config=self.config.actor_rollout_ref,
                                                     role='actor_rollout')
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
                                                  config=self.config.actor_rollout_ref,
                                                  role='ref')
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]['rm'] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg['critic']
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg['ref']
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg['rm']
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg['actor_rollout']
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
        if self.config.trainer.resume_mode == 'disable':
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            NotImplementedError('load from hdfs is not implemented yet')
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == 'auto':
            if global_step_folder is None:
                print('Training from scratch')
                return 0
        elif self.config.trainer.resume_mode == 'resume_path':
            print(f"Resume path")
            assert isinstance(self.config.trainer.resume_mode, str), "resume ckpt must be str type"
            assert 'global_step_' in self.config.trainer.resume_from_path, "resume ckpt must specify the global_steps"
            global_step_folder = self.config.trainer.resume_from_path
            if not os.path.isabs(global_step_folder):
                working_dir = os.getcwd()
                global_step_folder = os.path.join(working_dir, global_step_folder)
        else:
            if not (self.config.trainer.resume_from_path and global_step_folder is not None):
                assert isinstance(self.config.trainer.resume_mode, str), "resume ckpt must be str type"
                assert 'global_step_' in self.config.trainer.resume_mode, "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_mode
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)


        print(f'Load from checkpoint folder: {global_step_folder}')
        # set global step
        self.global_steps = int(global_step_folder.split('global_step_')[-1])

        print(f'Setting global step to {self.global_steps}')
        print(f'Resuming from {global_step_folder}') # Example: /data/checkpoints/verl/qwen2_esl_th0.10_e1_D1/global_step_50



        actor_path = os.path.join(global_step_folder, 'actor')
        critic_path = os.path.join(global_step_folder, 'critic')
        # load actor
        self.actor_rollout_wg.load_checkpoint(actor_path,
                                              del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(critic_path,
                                           del_local_after_load=self.config.trainer.del_local_ckpt_after_load)

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, 'data.pt')
        self.train_dataloader = torch.load(dataloader_local_path)
        if isinstance(self.train_dataloader.dataset, RLHFDataset):
            self.train_dataloader.dataset.resume_dataset_state()

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix='global_seqlen'):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch['attention_mask']
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch['attention_mask'].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst,
                                                              k_partitions=world_size,
                                                              equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        if metrics:
            global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst,
                                                        partitions=global_partition_lst,
                                                        prefix=logging_prefix)
            metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf

        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate()
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return

        # we start from step 1
        self.global_steps += 1

        for epoch in range(self.config.trainer.total_epochs):
            if epoch == 0 and 'prob' in self.config.reward_model.reward_manager:
                promptgt2scoreA = self.compute_promptgt2scoreA(epoch)

            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                batch: DataProto = DataProto.from_single_dict(batch_dict)

                if 'prob' in self.config.reward_model.reward_manager:

                    # Decode all input IDs in the batch at once
                    prompts = self.tokenizer.batch_decode(
                        batch.batch['input_ids'], 
                        skip_special_tokens=False
                    )
                    prompts = [prompt.replace(self.tokenizer.pad_token, '') for prompt in prompts]

                    # Extract ground truths for the entire batch
                    ground_truths = [item.non_tensor_batch['reward_model']['ground_truth'] for item in batch]

                    # Combine prompts and ground truths to create keys for lookup
                    prompt_gt_keys = [prompt + gt for prompt, gt in zip(prompts, ground_truths)]

                    # Check if any prompt_gt_key is missing in promptgt2scoreA
                    if any(key not in promptgt2scoreA for key in prompt_gt_keys):
                        print("Skipping batch due to missing scoreA.")  # Log for robustness
                        continue

                    # Assign scoreA to each item in the batch
                    for i, key in enumerate(prompt_gt_keys):
                        batch[i].non_tensor_batch['reward_model']['scoreA'] = promptgt2scoreA[key]

                gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])

                with _timer('step', timing_raw):

                    # pre-generate a batch
                    with _timer('gen', timing_raw):
                        print(f"{gen_batch.meta_info=}")
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch) # gen_batch: 256 batch size --> gen_batch_output: 256*5

                    metrics.update(self.compute_think_answer_length_metrics(gen_batch_output))

                    if self.config.algorithm.adv_estimator == 'remax':
                        raise NotImplementedError
                        with _timer('gen_max', timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info['do_sample'] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch['reward_baselines'] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    if 'ce' in self.config.reward_model.reward_manager:
                        print(f"Using cross entropy reward...")
                        ground_truth_list = [batch[i_].non_tensor_batch['reward_model']['ground_truth'] for i_ in range(len(batch))]
                        ground_truth_list = [item for item in ground_truth_list for _ in range(self.config.actor_rollout_ref.rollout.n)]
                        if 'binary' in self.config.reward_model.reward_manager:
                            start_answer_pos = '<answer> Therefore, the correct answer is:'
                            start_answer_neg = '<answer> Therefore, the wrong answer is:'
                            gen_batch_output_posce = self.construct_new_batch(gen_batch_output, ground_truth_list, start_answer=start_answer_pos, suffix='_posce')
                            gen_batch_output_negce = self.construct_new_batch(gen_batch_output, ground_truth_list, start_answer=start_answer_neg, suffix='_negce')
                        else:
                            gen_batch_output_ce = self.construct_new_batch(gen_batch_output, ground_truth_list)

                    batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                             dtype=object)
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True) # batch: 256 batch_Size --> 256*5
                    batch = batch.union(gen_batch_output)

                    if 'ce' in self.config.reward_model.reward_manager:
                        if 'binary' in self.config.reward_model.reward_manager:
                            batch = batch.union(gen_batch_output_posce)
                            batch = batch.union(gen_batch_output_negce)
                        else:
                            batch = batch.union(gen_batch_output_ce)

                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                    # recompute old_log_probs
                    # Actually, below is the log_prob towards the GT
                    with _timer('old_log_prob', timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        response_entropy = old_log_prob.pop(batch_keys=['entropy']).batch['entropy']
                        row_mean = self.compute_row_mean(response_entropy, batch.batch['attention_mask']) # (N,)
                        batch = batch.union(old_log_prob)

                        metrics.update({
                            'entropy/response_entropy/mean': torch.mean(row_mean).item(),
                            'entropy/response_entropy/max': torch.max(row_mean).item(),
                            'entropy/response_entropy/min': torch.min(row_mean).item(),
                        }) # We should use `update`. Do not use metrics[kk] = vv

                    if 'ce' in self.config.reward_model.reward_manager:
                        with _timer('old_log_prob_ce', timing_raw):
                            # "1 + 2 = <|im_end|><|im_start|>assistant\n<think> aaa </think>" --> Model "<answer> The answer is 3. </answer>"
                            if 'binary' in self.config.reward_model.reward_manager:
                                old_log_prob_posce = self.actor_rollout_wg.compute_log_prob_posce(batch)
                                old_log_prob_negce = self.actor_rollout_wg.compute_log_prob_negce(batch)
                                batch = batch.union(old_log_prob_posce)
                                batch = batch.union(old_log_prob_negce)
                            else:
                                old_log_prob_ce = self.actor_rollout_wg.compute_log_prob_ce(batch)
                                batch = batch.union(old_log_prob_ce)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer('ref', timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer('values', timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer('adv', timing_raw):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)


                        reward_tensor, scoreB_tensor, scoreA_tensor, format_reward_tensor, extracted_answer_list = self.reward_fn(batch) # reward_tensor.shape: torch.Size([40, 1024])
                        # Each row in reward_tensor contains at most 1 element.

                        # plot scoreB-scoreA
                        scoreA_list, scoreB_list, scoreB_minus_scoreA_list = [], [], []
                        for i_ in range(len(batch)):
                            # scoreA_ = batch[i_].non_tensor_batch['reward_model'].get('scoreA', 0.0)
                            scoreA_ = scoreA_tensor[i_].sum().item()
                            scoreB_ = scoreB_tensor[i_].sum().item()
                            scoreA_list.append(scoreA_)
                            scoreB_list.append(scoreB_)
                            scoreB_minus_scoreA_list.append(scoreB_ - scoreA_)
                        if (self.global_steps - 1) % 50 == 0:
                            self._maybe_log_histogram_to_wandb(scoreA_list, f'figures/scoreA', 'scoreA', 'Score A Distribution')
                            self._maybe_log_histogram_to_wandb(scoreB_list, f'figures/scoreB', 'scoreB', 'Score B Distribution')
                            self._maybe_log_histogram_to_wandb(scoreB_minus_scoreA_list, f'figures/scoreB-scoreA', 'scoreB-scoreA', 'ScoreB - ScoreA Distribution')

                        if self.config.reward_model.get("repetition_penalty", False):
                            # Decode all responses in a batch
                            responses = self.tokenizer.batch_decode(batch.batch['responses'], skip_special_tokens=True)
                            for i_, response_i in enumerate(responses):
                                # Apply repetition penalty
                                non_zero_indices = reward_tensor[i_].nonzero(as_tuple=True)
                                reward_tensor[i_][non_zero_indices] += detect_repetition_with_hash(response_i, window_size=10)


                        batch.batch['token_level_scores'] = reward_tensor

                        # compute rewards. apply_kl_penalty if available
                        if not self.config.actor_rollout_ref.actor.get('use_kl_loss', False):
                            batch, kl_metrics = apply_kl_penalty(batch,
                                                                 kl_ctrl=self.kl_ctrl,
                                                                 kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

                        # compute advantages, executed on the driver process
                        batch, filter_rate = compute_advantage(batch,
                                                  adv_estimator=self.config.algorithm.adv_estimator,
                                                  gamma=self.config.algorithm.gamma,
                                                  lam=self.config.algorithm.lam,
                                                  num_repeat=self.config.actor_rollout_ref.rollout.n)
                    metrics.update({"critic/filter_rate": filter_rate})
                    # compute reward distribution inside a batch
                    uid_to_distribution = self.compute_reward_distributions(reward_tensor, batch)
                    std_per_group, gap_per_group, mean_per_group = [], [], []
                    for group_idx, (uid, distribution) in enumerate(uid_to_distribution.items()):
                        mean_per_group.append(distribution['mean'])
                        std_per_group.append(distribution['std'])
                        gap_per_group.append(distribution['gap'])
                    
                    metrics.update({
                        "critic/rewards/mean_per_group/mean": sum(mean_per_group) / len(mean_per_group),
                        "critic/rewards/mean_per_group/max": max(mean_per_group),
                        "critic/rewards/mean_per_group/min": min(mean_per_group),
                        "critic/rewards/std_per_group/mean": sum(std_per_group) / len(std_per_group),
                        "critic/rewards/std_per_group/max": max(std_per_group),
                        "critic/rewards/std_per_group/min": min(std_per_group),
                    })
                    # uid_to_rewards = defaultdict(list)

                    log_to_wandb = {
                        "scores": [],
                        "scoreAs": [],
                        "scoreBs": [],
                        "advantages": [],
                        "entropies": [],
                        "format_rewards": [],
                        "data_sources": [],
                        "sequences": [],
                        "extracted_answers": [],
                        "ground_truths": [],
                        "final_reward_stds": [],
                    }
                    entropy_list, advantage_list = [], []
                    for i_ in range(len(batch)):
                        advantage_ = torch.masked_select(batch[i_].batch['advantages'], batch[i_].batch['attention_mask'][-self.config.data.max_response_length:].bool()).mean().item()
                        advantage_list.append(advantage_)
                        entropy_ = response_entropy[i_].mean().item()
                        entropy_list.append(entropy_)
                        if batch[i_].non_tensor_batch['uid'] == batch[0].non_tensor_batch['uid']:
                            score_ = reward_tensor[i_].sum().item()
                            scoreA_ = scoreA_tensor[i_].sum().item()
                            scoreB_ = scoreB_tensor[i_].sum().item()
                            # original_scoreB_ = original_scoreB_tensor[i_].sum().item()
                            format_reward_ = format_reward_tensor[i_].sum().item()
                            data_source_ = batch[i_].non_tensor_batch['data_source']
                            sequence_ = self.tokenizer.decode(batch.batch['input_ids'][i_][batch.batch['attention_mask'][i_].bool()], skip_special_tokens=False)
                            extracted_answer_ = extracted_answer_list[i_]
                            ground_truth_ = batch[i_].non_tensor_batch['reward_model']['ground_truth']
                            final_reward_std_ = batch[i_].batch['final_reward_stds'].item()

                            log_to_wandb['scores'].append(score_)
                            log_to_wandb['scoreAs'].append(scoreA_)
                            log_to_wandb['scoreBs'].append(scoreB_)
                            log_to_wandb['advantages'].append(advantage_)
                            log_to_wandb['entropies'].append(entropy_)
                            log_to_wandb['format_rewards'].append(format_reward_)
                            log_to_wandb['data_sources'].append(data_source_)
                            log_to_wandb['sequences'].append(sequence_)
                            log_to_wandb['extracted_answers'].append(extracted_answer_)
                            log_to_wandb['ground_truths'].append(ground_truth_)
                            log_to_wandb['final_reward_stds'].append(final_reward_std_)

                    if (self.global_steps - 1) % 2 == 0:
                        self._maybe_log_train_generations_to_wandb(table_attr_name='train_table', table_name='generations_same_instruction', **log_to_wandb)
                    if (self.global_steps - 1) % 10 == 0:
                        N = self.config.trainer.get("train_generations_to_log_to_wandb_2", 0)
                        if N > 0:
                            # scoreA_tensor = torch.tensor([batch[i_].non_tensor_batch['reward_model'].get("scoreA", 0.0) for i_ in range(len(batch))])
                            self._maybe_log_train_generations_to_wandb(table_attr_name="train_table_2", table_name="generations_varied_instruction", 
                                                                       **self.sample_batch_data(batch, reward_tensor=reward_tensor, scoreA_tensor=scoreA_tensor,
                                                                       scoreB_tensor=scoreB_tensor, advantage_list=advantage_list, entropy_list=entropy_list,
                                                                       format_reward_tensor=format_reward_tensor,
                                                                       extracted_answer_list=extracted_answer_list, 
                                                                       N=N))
                    # Log to /data/logs/train_generations/xxx.csv
                    self.log_train_generations(batch, reward_tensor=reward_tensor, scoreA_tensor=scoreA_tensor,
                                                scoreB_tensor=scoreB_tensor, advantage_list=advantage_list, entropy_list=entropy_list,
                                                format_reward_tensor=format_reward_tensor,
                                                extracted_answer_list=extracted_answer_list)

                    # update critic
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)

                    # compute nonzero scores
                    score_list = []
                    for i_ in range(batch.batch['input_ids'].shape[0]):
                        score_list.append(batch.batch['token_level_scores'][i_].sum().item())

                    metrics.update(self.bin_scores(score_list))

                    debug = False
                    if debug:
                        print(f"We wil debug all components here", flush=True)
                        print(f"{batch=}", flush=True)
                        # batch=DataProto(batch=TensorDict(
                        # fields={
                        # advantages: Tensor(shape=torch.Size([1280, 1024]), device=cpu, dtype=torch.float32, is_shared=False),
                        # attention_mask: Tensor(shape=torch.Size([1280, 1536]), device=cpu, dtype=torch.int64, is_shared=False),
                        # attention_mask_ce: Tensor(shape=torch.Size([1280, 2560]), device=cpu, dtype=torch.int64, is_shared=False),
                        # ground_truth_mask_ce: Tensor(shape=torch.Size([1280, 1024]), device=cpu, dtype=torch.int64, is_shared=False),
                        # input_ids: Tensor(shape=torch.Size([1280, 1536]), device=cpu, dtype=torch.int64, is_shared=False),
                        # input_ids_ce: Tensor(shape=torch.Size([1280, 2560]), device=cpu, dtype=torch.int64, is_shared=False),
                        # old_log_probs: Tensor(shape=torch.Size([1280, 1024]), device=cpu, dtype=torch.float32, is_shared=False),
                        # old_log_probs_ce: Tensor(shape=torch.Size([1280, 1024]), device=cpu, dtype=torch.float32, is_shared=False),
                        # position_ids: Tensor(shape=torch.Size([1280, 1536]), device=cpu, dtype=torch.int64, is_shared=False),
                        # position_ids_ce: Tensor(shape=torch.Size([1280, 2560]), device=cpu, dtype=torch.int64, is_shared=False),
                        # prompts: Tensor(shape=torch.Size([1280, 512]), device=cpu, dtype=torch.int64, is_shared=False),
                        # prompts_ce: Tensor(shape=torch.Size([1280, 1536]), device=cpu, dtype=torch.int64, is_shared=False),
                        # ref_log_prob: Tensor(shape=torch.Size([1280, 1024]), device=cpu, dtype=torch.float32, is_shared=False),
                        # responses: Tensor(shape=torch.Size([1280, 1024]), device=cpu, dtype=torch.int64, is_shared=False),
                        # responses_ce: Tensor(shape=torch.Size([1280, 1024]), device=cpu, dtype=torch.int64, is_shared=False),
                        # returns: Tensor(shape=torch.Size([1280, 1024]), device=cpu, dtype=torch.float32, is_shared=False),
                        batch_keys = batch.batch.keys()
                        for i_debug in range(batch.batch['input_ids'].shape[0]):
                            sample = batch.batch[i_debug]
                            print(f"{i_debug=} {repr(self.tokenizer.decode(batch.batch['input_ids'][i_debug][batch.batch['attention_mask'][i_debug].bool()], skip_special_tokens=False))=}\n\n", flush=True)
                            print(f"{i_debug=} {repr(self.tokenizer.decode(batch.batch['input_ids_ce'][i_debug][batch.batch['attention_mask_ce'][i_debug].bool()], skip_special_tokens=False))=}\n\n", flush=True)
                            print(f"{i_debug=} {repr(self.tokenizer.decode(batch.batch['responses'][i_debug], skip_special_tokens=True))=}\n\n", flush=True)
                            print(f"{i_debug=} {repr(self.tokenizer.decode(batch.batch['responses_ce'][i_debug], skip_special_tokens=True))=}\n\n", flush=True)
                            print(f"{i_debug=} {repr(self.tokenizer.decode(batch.batch['responses_ce'][i_debug][batch.batch['ground_truth_mask_ce'][i_debug].bool()], skip_special_tokens=False))=}\n\n", flush=True)
                            # print(f"{i_debug=} {repr(self.tokenizer.decode(batch.batch['input_ids'][i_debug][batch.batch['attention_mask'][i_debug]!=0], skip_special_tokens=False))=}\n\n", flush=True)
                            # print(f"{i_debug=} {repr(self.tokenizer.decode(batch.batch['input_ids_ce'][i_debug][batch.batch['attention_mask_ce'][i_debug]!=0], skip_special_tokens=False))=}\n\n", flush=True)
                            for k_debug in ['token_level_scores']: #, 'token_level_rewards', 'token_level_scores', 'returns', 'advantages']:
                                # print(f"{i_debug=} {k_debug=} {batch.batch[k_debug][i_debug]=}")
                                # print(f"{i_debug=} {k_debug=} {batch.batch[k_debug][i_debug][:20]=}")
                                # print(f"{i_debug=} {k_debug=} {batch.batch[k_debug][i_debug][-20:]=}")
                                print(f"{i_debug=} {k_debug=} {batch.batch[k_debug][i_debug][batch.batch['attention_mask'][i_debug][self.config.data.max_prompt_length:]!=0]=}")
                            # assert False
                            # if i_debug:
                            #     break


                    if 'ce' in self.config.reward_model.reward_manager:
                        if 'binary' in self.config.reward_model.reward_manager:
                            # escape_keys = ['response_mask_posce', 'response_mask_negce']
                            debug = False
                            if debug:
                                print(f"{batch.batch['response_mask_posce']=}")
                                print(f"{batch.batch['response_mask_negce']=}")
                                print(f"{batch.batch['response_mask_posce'].sum()=}")
                                print(f"{batch.batch['response_mask_negce'].sum()=}")
                                print(f"{(batch.batch['response_mask_posce'] - batch.batch['response_mask_negce']).sum()=}") # 0
                                print(f"We change response_mask_posce to response_mask_ce")
                            batch.pop(batch_keys=['response_mask_posce'])
                            batch.batch['response_mask_ce'] = batch.batch.pop('response_mask_negce')
                            if self.config.reward_model.get("optimize_think_only", True):
                                escape_keys = ['response_mask_ce']
                            else:
                                escape_keys = []
                            batch_keys_rm = [batch_k for batch_k in batch.batch.keys() if ((batch_k.endswith('_posce') or batch_k.endswith('_negce')) and batch_k not in escape_keys)]
                            debug = False
                            if debug:
                                print(f"{batch_keys_rm=}")
                                print(f"{batch=}")
                        else:
                            # escape_keys = ['response_mask_ce']
                            if self.config.reward_model.get("optimize_think_only", True):
                                escape_keys = ['response_mask_ce']
                            else:
                                escape_keys = []
                            batch_keys_rm = [batch_k for batch_k in batch.batch.keys() if (batch_k.endswith('_ce') and batch_k not in escape_keys)]
                        print(f"{batch_keys_rm=}")
                        batch.pop(batch_keys=batch_keys_rm)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer('update_actor', timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch) # , tokenizer_debug=self.tokenizer)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)
                    if 'ce' in self.config.reward_model.reward_manager:
                        if 'binary' in self.config.reward_model.reward_manager:
                            batch_keys_rm = [batch_k for batch_k in batch.batch.keys() if batch_k.endswith('_posce') or batch_k.endswith('_negce') or batch_k.endswith('_ce')]
                        else:
                            batch_keys_rm = [batch_k for batch_k in batch.batch.keys() if batch_k.endswith('_ce')]
                        if len(batch_keys_rm) != 0:
                            batch.pop(batch_keys=batch_keys_rm)

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                        self.global_steps % self.config.trainer.test_freq == 0:
                        with _timer('testing', timing_raw):
                            val_metrics: dict = self._validate()
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and \
                            self.global_steps % self.config.trainer.save_freq == 0:
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()

                # collect metrics
                # acc_reward = acc_tensor.sum(-1)
                scoreB = scoreB_tensor.sum(-1)
                scoreA = scoreA_tensor.sum(-1)
                format_reward = format_reward_tensor.sum(-1)
                metrics.update({# reward
                    'critic/scoreB/mean':
                        torch.mean(scoreB).detach().item(),
                    'critic/scoreB/max':
                        torch.max(scoreB).detach().item(),
                    'critic/scoreB/min':
                        torch.min(scoreB).detach().item(),
                    'critic/scoreA/mean':
                        torch.mean(scoreA).detach().item(),
                    'critic/scoreA/max':
                        torch.max(scoreA).detach().item(),
                    'critic/scoreA/min':
                        torch.min(scoreA).detach().item(),
                    'critic/format_rewards/mean':
                        torch.mean(format_reward).detach().item(),
                    'critic/format_rewards/max':
                        torch.max(format_reward).detach().item(),
                    'critic/format_rewards/min':
                        torch.min(format_reward).detach().item(),
                })
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                metrics.update(self.compute_scoreB_by_data_source_metrics(batch=batch, scoreB_tensor=scoreB_tensor, name='scoreB'))
                metrics.update(self.compute_scoreB_by_data_source_metrics(batch=batch, scoreB_tensor=scoreA_tensor, name='scoreA'))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                self.global_steps += 1

                if self.global_steps >= self.total_training_steps:

                    # perform validation after training
                    if self.val_reward_fn is not None:
                        val_metrics = self._validate()
                        pprint(f'Final validation metrics: {val_metrics}')
                        logger.log(data=val_metrics, step=self.global_steps)
                    if self.config.trainer.save_freq > 0 and \
                            (self.global_steps - 1) % self.config.trainer.save_freq != 0:
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()
                    return


    def collate_fn(self, data_list: list[dict]) -> dict:
        tensors = {}
        non_tensors = {}

        for data in data_list:
            for key, val in data.items():
                if isinstance(val, torch.Tensor):
                    if key not in tensors:
                        tensors[key] = []
                    tensors[key].append(val)
                else:
                    if key not in non_tensors:
                        non_tensors[key] = []
                    non_tensors[key].append(val)

        for key, val in tensors.items():
            tensors[key] = torch.stack(val, dim=0)

        for key, val in non_tensors.items():
            non_tensors[key] = np.array(val, dtype=object)

        output = {}
        output.update(tensors)
        output.update(non_tensors)
        return output
    
    def count_pad_tokens(self, s, pad_token_str):
        # Count the number of pad tokens on the left
        left_count = 0
        while s.startswith(pad_token_str):
            left_count += 1
            s = s[len(pad_token_str):]
        
        # Count the number of pad tokens on the right
        right_count = 0
        while s.endswith(pad_token_str):
            right_count += 1
            s = s[:-len(pad_token_str)]
        
        return left_count, right_count

    def get_scoreA(self, data):
        batch_input_ids = data.batch['input_ids'] # [256, 512]
        pad_token_str = self.tokenizer.pad_token
        eos_token_str = self.tokenizer.eos_token
        max_prompt_length, max_response_length = self.config.data.max_prompt_length, self.config.data.max_response_length
        data_list = []
        prompt_str_list, ground_truth_list = [], []
        add_answer_tags = False
        if add_answer_tags:
            start_answer, end_answer = '<answer>', '</answer>'
        for i in range(len(batch_input_ids)):
            data_item = data[i]
            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            # ... ion?\n<|im_end|>\n<|im_start|>assistant\n'
            prompt_str = self.tokenizer.decode(batch_input_ids[i], skip_special_tokens=False)
            if add_answer_tags:
                new_text = prompt_str + ' ' + start_answer + ' ' + ground_truth + ' ' + end_answer + ' ' + eos_token_str
            else:
                new_text = prompt_str + ' ' + ground_truth + ' ' + eos_token_str
            new_text_rmpad = new_text.replace(self.tokenizer.pad_token, '')
            if eos_token_str not in new_text_rmpad: # For a base model, the eos_token_str is the same as pad_token_str
                new_text_rmpad += eos_token_str
            outputs = self.tokenizer(new_text_rmpad, return_tensors='pt', add_special_tokens=False)
            input_ids = outputs['input_ids']
            attention_mask = outputs['attention_mask']
            if add_answer_tags:
                sep_str = start_answer
            else:
                sep_str = '<|im_start|>assistant' + '\n'
            pos = self.locate_substring_tokens(new_text_rmpad, sep_str, self.tokenizer)

            prompts = input_ids[:, :pos[-1] + 1]
            responses = input_ids[:, pos[-1] + 1:]



            if add_answer_tags:
                pos_eos = self.locate_substring_tokens(new_text_rmpad, end_answer, self.tokenizer) # list
            else:
                pos_eos = self.locate_substring_tokens(new_text_rmpad, eos_token_str, self.tokenizer) # list

            ground_truth_ids = input_ids[:, prompts.shape[1]:pos_eos[0]]

            # Pad prompts and responses for future packing
            left_pad_tuple = (max_prompt_length- prompts.shape[-1], 0)
            right_pad_tuple = (0, max_response_length - responses.shape[-1])

            prompts = F.pad(prompts, left_pad_tuple, 'constant', self.tokenizer.pad_token_id) # pad to be max_length before collate_fn
            responses = F.pad(responses, right_pad_tuple, 'constant', self.tokenizer.pad_token_id) # pad to be max_response_length before collate_fn

            input_ids = torch.cat([prompts, responses], dim=-1)

            # pad right first
            position_ids = compute_position_id_with_mask(F.pad(attention_mask, right_pad_tuple, 'constant', 1))
            attention_mask = F.pad(attention_mask, right_pad_tuple, 'constant', 0)
            # then pad left
            attention_mask = F.pad(attention_mask, left_pad_tuple, 'constant', 0)
            position_ids = F.pad(position_ids, left_pad_tuple, 'constant', 0)

            ground_truth_mask = torch.zeros_like(responses)
            ground_truth_mask[:, :ground_truth_ids.shape[-1]] = 1



            debug = False
            if debug:
                print(f"{new_text_rmpad=}")
                print(f"{prompts.shape=}")
                print(f"{ground_truth=}") # Madog ap Llywelyn
                print(f"{input_ids.shape=}")
                print(f"{new_text_rmpad=}")
                print(f"{ground_truth_ids=}")# tensor([[ 5672, 12738, 26278]])
                # response_mask_ce_shift_left = torch.zeros_like(gen_responses[ii])
                # response_mask_ce_shift_left[:response_mask_ce_length-1] = 1
                # response_mask_ce_shift_right = torch.zeros_like(gen_responses[ii])
                # response_mask_ce_shift_right[:response_mask_ce_length+1] = 1
                # print(f"{self.tokenizer.decode(gen_responses[ii][response_mask_ce_shift_left.bool()])=}") # '<think> The question is asking for ...  it. </think'
                # print(f"{self.tokenizer.decode(gen_responses[ii][response_mask_ce.bool()])=}") # '<think> The question is asking about the architect who designed the Shard Building, which is the tallest building in London. I need to recall the architect associated with this major landmark. \n\nThe Shard in London is designed by architect Renzo Piano. </think>\n\n'
                # print(f"{self.tokenizer.decode(gen_responses[ii][response_mask_ce_shift_right.bool()])=}") # '<think> The question is asking for ... it. </think>\n\n<'
                # print(f"{self.tokenizer.decode(gen_responses[ii]).replace(pad_token_str, '')=}") # '<think> The question is asking about the architect who designed the Shard Building, which is the tallest building in London. I need to recall the architect associated with this major landmark. \n\nThe Shard in London is designed by architect Renzo Piano. </think>\n\n<answer> The architect who designed the Shard Building in London is Renzo Piano. </answer><|im_end|>
                print(f"{self.tokenizer.decode(prompts[0], skip_special_tokens=True)=}")# 'system\nA conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\nuser\nPlease answer this question: Which architect designed the 87-storey Shard Building in London?\nassistant\n<think> The question is asking for the architect of the Shard Building in London. I will need to recall information about famous architecture in London and the Shard Building specifically.\n\nThe Shard Building in London was designed by the Italian architect Renzo Piano. \n\n </think>'
                print(f"{self.tokenizer.decode(responses[0], skip_special_tokens=True)=}") # ' <answer> renzo piano </answer>'
                print(f"{self.tokenizer.decode(responses[ground_truth_mask.bool()], skip_special_tokens=False)=}") # ' renzo piano'

                print('-'*50)
                # assert False

            row_dict = {
                'prompts': prompts[0],
                'responses': responses[0],
                'input_ids': input_ids[0],
                'attention_mask': attention_mask[0],
                'position_ids': position_ids[0],
                'ground_truth_mask': ground_truth_mask[0],
            }

            prompt_str_list.append(prompt_str.replace(pad_token_str, ''))
            ground_truth_list.append(ground_truth)
            debug = False
            if debug:
                for k, v in row_dict.items():
                    print(f"{k=} {v.shape=}")
            data_list.append(row_dict)

        data_new: DataProto = DataProto.from_single_dict(self.collate_fn(data_list))
        old_log_probs = self.actor_rollout_wg.compute_log_prob(data_new)['old_log_probs'].batch
        scoreAs_list = []
        old_log_probs_in_gt_list = []
        for i in range(len(batch_input_ids)):
            ground_truth_mask = data_new[i].batch['ground_truth_mask']
            old_log_prob = old_log_probs[i]
            scoreAs = float(torch.exp(old_log_prob[ground_truth_mask.bool()].mean(dim=-1)))
            scoreAs_list.append(scoreAs)
            old_log_probs_in_gt_list.append(old_log_prob[ground_truth_mask.bool()])
        debug = False
        if debug:
            print(f"{old_log_probs.shape=}")
            print(f"{scoreAs_list=}")

        return scoreAs_list, prompt_str_list, ground_truth_list, old_log_probs_in_gt_list




    def construct_new_batch(self, gen_batch_output, ground_truth_list, 
                                  start_think='<think>', end_think='</think>', 
                                  start_answer='<answer>', end_answer='</answer>',
                                  suffix='_ce'):
        """
            We convert the `input_ids` to a new batch including IAP, `prompts_ce`, `responses_ce` and `ground_truth_mask_ce`.
        """

        gen_ids = gen_batch_output.batch['input_ids']
        gen_responses = gen_batch_output.batch['responses']

        data_list = []
        pad_token_str = self.tokenizer.pad_token
        eos_token_str = self.tokenizer.eos_token

        max_length = self.config.data.max_prompt_length + self.config.data.max_response_length
        for ii in range(len(gen_ids)):
            gen_text = self.tokenizer.decode(gen_ids[ii], skip_special_tokens=False) # '<|endoftext|>...<|endoftext|><|im_start|>system\n\nWhen tackling complex reasoning tasks, you have access to the following actions. Use them as needed to progress through your thought process.\n\n[ASSESS]\n\n[ADVANCE]\n\n[VERIFY]\n\n[SIMPLIFY]\n\n[SYNTHESIZE]\n\n[PIVOT]\n\n[OUTPUT]\n\nYou should strictly follow the format below:\n\n[ACTION NAME]\n\n# Your action step 1\n\n# Your action step 2\n\n# Your action step 3\n\n...\n\nNext action: [NEXT ACTION NAME]\n\n<|im_end|>\n<|im_start|>user\nCalculate $\\sqrt{48}-6\\sqrt{\\frac{1}{3}}-\\sqrt{18}÷\\sqrt{6}$.\n\nPresent the answer in LaTex format: \\boxed{Your answer}<|im_end|>\n<|im_start|>assistant\n[ACTION NAME]\n\n# First, calculate the square roots and simplify where possible.\n\n\\[\n\\sqrt{48}=\\sqrt{16\\cdot3}=4\\sqrt{3}\n\\]\n\n\\[\n-\\sqrt{18}÷\\sqrt{6}=-\\sqrt{18÷6}=-\\sqrt{3}\n\\]\n\nNext, simplify \\(6\\sqrt{\\frac{1}{3}}\\):\n\n\\[\n6\\sqrt{\\frac{1}{3}}=6\\cdot\\frac{1}{\\sqrt{3}}=6\\cdot\\frac{\\sqrt{3}}{3}=2\\sqrt{3}\n\\]\n\n# Now, put it all together:\n\n\\[\n\\sqrt{48}-6\\sqrt{\\frac{1}{3}}-\\sqrt{18}÷\\sqrt{6} = 4\\sqrt{3} - 2\\sqrt{3} - (-\\sqrt{3})\n\\]\n\n# Simplify the expression:\n\n\\[\n4\\sqrt{3} - 2\\sqrt{3} + \\sqrt{3} = (4-2+1)\\sqrt{3} = 3\\sqrt{3}\n\\]\n\n# Present the answer in LaTex format:\n\n\\[\n\\boxed{3\\sqrt{3}}\n\\]\n\nNext action: [OUTPUT]\n\n**OUTPUT**: The answer in LaTeX format is \\(\\boxed{3\\sqrt{3}}\\).<|im_end|><|endoftext|>...<|endoftext|>'
            gen_text_rmpad = gen_text.replace(pad_token_str, '')
            if eos_token_str not in gen_text_rmpad:
                gen_text_rmpad += eos_token_str
            gen_responses_text = self.tokenizer.decode(gen_responses[ii], skip_special_tokens=False)
            gen_responses_text_rmpad = gen_responses_text.replace(pad_token_str, '')
            if eos_token_str not in gen_responses_text_rmpad:
                gen_responses_text_rmpad += eos_token_str

            ground_truth = str(ground_truth_list[ii])

            start_think_count = gen_responses_text_rmpad.count(start_think)
            end_think_count = gen_responses_text_rmpad.count(end_think)
            middle_content, leading_whitespace, trailing_whitespace = ' ', ' ', ' ' # content between `</think>` and `<answer>`
            if self.config.reward_model.get("version", None) == 'v2':
                start_answer_tag = '<answer>' # The contrastive reward sets the start_answer as "<answer> The correct answer is:", but we only detect "<answer>"
                start_answer_count = gen_responses_text_rmpad.count(start_answer_tag)
                valid_flag = (
                    start_think_count == 1 and 
                    end_think_count == 1 and 
                    start_answer_count == 1 and
                    start_think in gen_responses_text_rmpad and 
                    end_think in gen_responses_text_rmpad.split(start_think)[-1] and
                    start_answer_tag in gen_responses_text_rmpad.split(end_think)[-1]
                )
                if valid_flag:
                    middle_content = gen_responses_text_rmpad.split(end_think)[1].split(start_answer_tag)[0]
            elif self.config.reward_model.get("version", None) == 'v2.1':

                start_answer_tag = '<answer>' # The contrastive reward sets the start_answer as "<answer> The correct answer is:", but we only detect "<answer>"
                # end_answer_tag = '</answer>'
                start_answer_count = gen_responses_text_rmpad.count(start_answer_tag)
                # # end_answer_count = gen_responses_text_rmpad.count(end_answer_tag)
                pattern = r'^.*' + start_think + r'.*' + end_think +  r'.*' + start_answer_tag + r'.*$'
                valid_flag = (
                    start_think_count == 1 and
                    end_think_count == 1 and
                    start_answer_count == 1 and
                    (re.fullmatch(pattern, gen_responses_text_rmpad, re.DOTALL) is not None)
                )
                # if valid_flag:
                #     middle_content = gen_responses_text_rmpad.split(end_think)[1].split(start_answer_tag)[0]

                if valid_flag:
                    middle_content = gen_responses_text_rmpad.split(end_think)[1].split(start_answer_tag)[0]
       
                    # Extract answer section and its whitespace
                    answer_section = gen_responses_text_rmpad.split(start_answer_tag)[1]
                    
                    # Check if answer section contains only whitespace
                    if not answer_section.strip():  # This checks if string is empty or only whitespace
                        valid_flag = False
                    else:
                        # Extract leading whitespace (spaces and newlines) after <answer>
                        leading_whitespace = ''
                        for i, char in enumerate(answer_section):
                            if char in [' ', '\n', '\t', '\r']:
                                leading_whitespace += char
                            else:
                                break
                debug = False
                if debug:
                    if valid_flag:
                        print(f"{valid_flag=} {middle_content=} {leading_whitespace=} || {repr(gen_responses_text_rmpad)=}")
                    else:
                        print(f"{valid_flag=} || {repr(gen_responses_text_rmpad)=}")

            elif self.config.reward_model.get("version", None) == 'v2.2':
                valid_flag = (self.format_reward(gen_responses_text_rmpad) == 1.0)
                
                start_answer_tag = '<answer>' # The contrastive reward sets the start_answer as "<answer> The correct answer is:", but we only detect "<answer>"
                if valid_flag:
                    middle_content = gen_responses_text_rmpad.split(end_think)[1].split(start_answer_tag)[0]
            elif self.config.reward_model.get("version", None) == 'v3':
                start_answer_tag = '<answer>' # The contrastive reward sets the start_answer as "<answer> The correct answer is:", but we only detect "<answer>"
                end_answer_tag = '</answer>'
                start_answer_count = gen_responses_text_rmpad.count(start_answer_tag)
                end_answer_count = gen_responses_text_rmpad.count(end_answer_tag)
                valid_flag = (
                    start_think_count == 1 and 
                    end_think_count == 1 and 
                    start_answer_count == 1 and
                    end_answer_count == 1 and
                    start_think in gen_responses_text_rmpad and 
                    end_think in gen_responses_text_rmpad.split(start_think)[-1] and
                    start_answer_tag in gen_responses_text_rmpad.split(end_think)[-1] and
                    end_answer_tag in gen_responses_text_rmpad.split(start_answer_tag)[-1]
                )
                if valid_flag:
                    middle_content = gen_responses_text_rmpad.split(end_think)[1].split(start_answer_tag)[0]
       
                    # Extract answer section and its whitespace
                    answer_section = gen_responses_text_rmpad.split(start_answer_tag)[1].split(end_answer_tag)[0]
                    
                    # Check if answer section contains only whitespace
                    if not answer_section.strip():  # This checks if string is empty or only whitespace
                        valid_flag = False
                    else:
                        # Extract leading whitespace (spaces and newlines) after <answer>
                        leading_whitespace = ''
                        for i, char in enumerate(answer_section):
                            if char in [' ', '\n', '\t', '\r']:
                                leading_whitespace += char
                            else:
                                break
                                
                        # Extract trailing whitespace (spaces and newlines) before </answer>
                        trailing_whitespace = ''
                        reversed_section = answer_section[::-1]
                        for i, char in enumerate(reversed_section):
                            if char in [' ', '\n', '\t', '\r']:
                                trailing_whitespace += char
                            else:
                                break
                        trailing_whitespace = trailing_whitespace[::-1]  # Reverse back to original order
                    

            else:
                valid_flag = (
                    start_think_count == 1 and 
                    end_think_count == 1 and 
                    start_think in gen_responses_text_rmpad and 
                    end_think in gen_responses_text_rmpad.split(start_think)[-1]
                )
            middle_content = middle_content if middle_content != '' else ' '
            leading_whitespace = leading_whitespace if leading_whitespace != '' else ' '
            trailing_whitespace = trailing_whitespace if trailing_whitespace != '' else ' '
            if valid_flag:
                pos_endthink_in_original_ids = self.locate_substring_tokens(gen_text, end_think, self.tokenizer) # list

                # NOTE: the calculation below has one issue: if </think> is closely connected to the next string, then it might not be accurately calculated.
                # For example, <think> aaa </think><answer> xxx </answer>. It is possible we end with "</think><"
                response_mask_ce_length = pos_endthink_in_original_ids[-1] + 1 - gen_batch_output.batch['prompts'].shape[-1]
                new_text = end_think.join(gen_text_rmpad.split(end_think)[:-1]) + end_think + middle_content + start_answer + leading_whitespace + ground_truth + trailing_whitespace + end_answer + eos_token_str
                # new_text = start_answer.join(gen_text_rmpad.split(start_answer)[:-1]) + start_answer + ' ' + ground_truth + ' ' + end_answer + eos_token_str
            else:
                # response_mask_ce_length = gen_batch_output.batch['responses'].shape[-1]
                indices = (gen_responses[ii] == self.tokenizer.eos_token_id).nonzero()
                if indices.numel() > 0:
                    response_mask_ce_length = indices[0].item()
                else:
                    response_mask_ce_length = gen_responses[ii].shape[-1]
                # No matter how we construct this new text, the final score is 0, including the format reward
                # transition_str = end_think + ' The answer is '
                # end_text = transition_str + start_answer + ground_truth + end_answer + eos_token_str # \nThe answer is \\boxed{xx}<|im_end|>
                end_text = end_think + middle_content + start_answer + leading_whitespace + ground_truth + trailing_whitespace + end_answer + eos_token_str # \nThe answer is \\boxed{xx}<|im_end|>
                end_text_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(end_text))
                new_text = self.tokenizer.decode(gen_ids[ii][:-len(end_text_ids) - 5], skip_special_tokens=False).replace(self.tokenizer.pad_token, "") + end_text

            response_mask_ce = torch.zeros_like(gen_responses[ii])
            response_mask_ce[:response_mask_ce_length] = 1


            input_data = self.tokenizer(new_text, return_tensors='pt', add_special_tokens=False)
            input_ids = input_data['input_ids']
            attention_mask = input_data['attention_mask']
            if len(input_ids) > max_length: # It is possible that both start answer and end answer exist but it exceeds the max length
                valid_flag = False

            pos_endthink = self.locate_substring_tokens(new_text, end_think, self.tokenizer) # list
            prompts = input_ids[:, :pos_endthink[-1]+1] # ids that ends to the </think>
            responses = input_ids[:, pos_endthink[-1]+1:]

            pos_startanswer = self.locate_substring_tokens(new_text, start_answer, self.tokenizer) # list
            pos_endanswer = self.locate_substring_tokens(new_text, end_answer, self.tokenizer) # list

            ground_truth_ids = input_ids[:, pos_startanswer[-1]+1:pos_endanswer[0]]

            # Pad prompts and responses for future packing
            left_pad_tuple = (max_length - prompts.shape[-1], 0)
            right_pad_tuple = (0, self.config.data.max_response_length - responses.shape[-1])

            prompts = F.pad(prompts, left_pad_tuple, 'constant', self.tokenizer.pad_token_id) # pad to be max_length before collate_fn
            responses = F.pad(responses, right_pad_tuple, 'constant', self.tokenizer.pad_token_id) # pad to be max_response_length before collate_fn

            input_ids = torch.cat([prompts, responses], dim=-1)

            # pad right first
            position_ids = compute_position_id_with_mask(F.pad(attention_mask, right_pad_tuple, 'constant', 1))
            attention_mask = F.pad(attention_mask, right_pad_tuple, 'constant', 0)
            # then pad left
            attention_mask = F.pad(attention_mask, left_pad_tuple, 'constant', 0)
            position_ids = F.pad(position_ids, left_pad_tuple, 'constant', 0)

            assert input_ids.ndim == 2
            sequence_length = input_ids.shape[-1]
            if sequence_length > max_length + self.config.data.max_response_length: # truncation error
                raise NotImplementedError(f'{sequence_length=} is larger than {max_length=}')

            ground_truth_mask = torch.zeros_like(responses)
            if valid_flag:
                # (pos_startanswer[-1] + 1) - (pos_endthink[-1] + 1)
                # 0 1 2 3 4 5 6 7 8 9 10 => [0 1 2 3 4 5] [6 7 8 9 10]
                start = (pos_startanswer[-1] + 1) - (pos_endthink[-1] + 1)
                ground_truth_mask[:, start:start + ground_truth_ids.shape[-1]] = 1 # Suppose the response is <think> ABC </think> <answer> DEF </answer>. Then the mask is on " DEF ".

            debug = False
            if debug:
            # if ii % 200 == 0 or ii == 0:
                print(f"{new_text=}")
                print(f"{ground_truth_ids=}")# tensor([[ 5672, 12738, 26278]])
                print(f"{valid_flag=}")
                # print(f"Original response || {gen_responses_text_rmpad=}")
                response_mask_ce_shift_left = torch.zeros_like(gen_responses[ii])
                response_mask_ce_shift_left[:response_mask_ce_length-1] = 1
                response_mask_ce_shift_right = torch.zeros_like(gen_responses[ii])
                response_mask_ce_shift_right[:response_mask_ce_length+1] = 1
                print(f"Original response (shifted left) that will be updated || {self.tokenizer.decode(gen_responses[ii][response_mask_ce_shift_left.bool()])=}") # '<think> The question is asking for ...  it. </think'
                print(f"Original response that will be updated || {self.tokenizer.decode(gen_responses[ii][response_mask_ce.bool()])=}") # '<think> The question is asking about the architect who designed the Shard Building, which is the tallest building in London. I need to recall the architect associated with this major landmark. \n\nThe Shard in London is designed by architect Renzo Piano. </think>\n\n'
                print(f"Original response (shifted right) that will be updated || {self.tokenizer.decode(gen_responses[ii][response_mask_ce_shift_right.bool()])=}") # '<think> The question is asking for ... it. </think>\n\n<'
                print(f"Original response || {self.tokenizer.decode(gen_responses[ii]).replace(pad_token_str, '')=}") # '<think> The question is asking about the architect who designed the Shard Building, which is the tallest building in London. I need to recall the architect associated with this major landmark. \n\nThe Shard in London is designed by architect Renzo Piano. </think>\n\n<answer> The architect who designed the Shard Building in London is Renzo Piano. </answer><|im_end|>
                print(f"New Prompt || {self.tokenizer.decode(prompts[0], skip_special_tokens=True)=}")# 'system\nA conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\nuser\nPlease answer this question: Which architect designed the 87-storey Shard Building in London?\nassistant\n<think> The question is asking for the architect of the Shard Building in London. I will need to recall information about famous architecture in London and the Shard Building specifically.\n\nThe Shard Building in London was designed by the Italian architect Renzo Piano. \n\n </think>'
                print(f"New Response || {self.tokenizer.decode(responses[0], skip_special_tokens=True)=}") # ' <answer> renzo piano </answer>'
                print(f"Ground-truth in the new response || {self.tokenizer.decode(responses[ground_truth_mask.bool()], skip_special_tokens=False)=}") # ' renzo piano'

                original_response_mask = gen_batch_output.batch['attention_mask'][:, -gen_responses.size(1):] # This will be originally used in updating actor
                print(f"Original response || {self.tokenizer.decode(gen_responses[ii][original_response_mask[ii].bool()])=}")  # <think> I need to remember that these seem to be names, and I should use a translation tool or my own knowledge of languages to translate them. Unfortunately, as a text-based AI, I might not be able to accurately translate names from one language to another. However, given my own knowledge, it could seem like these are Indian names, with both likely being given names. </think>\n<answer> Tandon, Ravana </answer><|im_end|>
                print('-'*50)
                    


            
            row_dict = {
                f'prompts{suffix}': prompts[0],
                f'responses{suffix}': responses[0],
                f'input_ids{suffix}': input_ids[0],
                f'attention_mask{suffix}': attention_mask[0],
                f'position_ids{suffix}': position_ids[0],
                f'ground_truth_mask{suffix}': ground_truth_mask[0],
                f'response_mask{suffix}': response_mask_ce,
            }
            data_list.append(row_dict)
        gen_batch_output: DataProto = DataProto.from_single_dict(self.collate_fn(data_list))
        return gen_batch_output


    def split_input_ids(self, input_ids, start_answer_ids):
        # Convert input_ids to a list for easier manipulation
        input_ids_list = input_ids.tolist()[0]
        
        # Convert start_answer_ids to a list if it's not already
        start_answer_ids_list = start_answer_ids if isinstance(start_answer_ids, list) else start_answer_ids.tolist()
        
        # Find the rightmost starting index of start_answer_ids in input_ids
        start_index = None
        for i in range(len(input_ids_list) - len(start_answer_ids_list), -1, -1):  # Iterate from the end
            if input_ids_list[i:i+len(start_answer_ids_list)] == start_answer_ids_list:
                start_index = i
                break
        
        if start_index is None:
            raise ValueError("start_answer_ids not found in input_ids")
        
        # Split the input_ids into prompts and responses
        prompts = input_ids_list[:start_index]
        responses = input_ids_list[start_index:]
        
        # Convert back to tensors
        prompts = torch.tensor([prompts])
        responses = torch.tensor([responses])
        
        return prompts, responses


    def locate_substring_tokens(self, full_string, substring, tokenizer):
        """
        Locates the token IDs and positions corresponding to a substring in a full string.

        Args:
            full_string (str): The full string to tokenize.
            substring (str): The substring to locate in the full string.
            tokenizer_name (str): The name of the tokenizer to use (default is "gpt2").
        """
        # Tokenize the full string and get byte-level offsets
        encoding = tokenizer(full_string, return_offsets_mapping=True, add_special_tokens=False)
        offsets = encoding["offset_mapping"]  # List of (start, end) byte positions for each token

        # Find the byte-level start and end positions of the substring in the full string
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

        return matching_token_indices

    def bin_scores(self, score_list):
        # Initialize a list to store the counts for each bin (11 bins now)
        bin_counts = [0] * 12  # 10 bins for [0, 1.0) + 1 bin for out-of-range values

        # Iterate over the scores in score_list
        num_zero_score, num_one_score = 0, 0
        for score in score_list:
            if score == 0:
                num_zero_score += 1
            elif score == 1:
                num_one_score += 1
            elif 0 < score < 1.0:  # Check if the score is within [0, 1.0)
                bin_index = int(score * 10)  # Determine the bin index
                bin_counts[bin_index] += 1
            elif score < 0:
                bin_counts[10] += 1  # The 11th bin is for out-of-range values
            elif score > 1.0: # score > 1.0
                # Handle out-of-range values (negative or >= 1.0)
                bin_counts[11] += 1  # The 11th bin is for out-of-range values

        # Create a metrics dictionary to store the bin counts
        metrics = {}
        metrics['final_reward_dist/final_reward<0'] = bin_counts[10] / len(score_list)
        metrics['final_reward_dist/final_reward=0'] = num_zero_score / len(score_list)
        for i in range(10):
            metrics[f'final_reward_dist/final_reward_in_{i/10:.1f}_to_{(i+1)/10:.1f}'] = bin_counts[i] / len(score_list)
        metrics['final_reward_dist/final_reward=1'] = num_one_score / len(score_list)
        metrics['final_reward_dist/final_reward>1'] = bin_counts[11] / len(score_list)


        return metrics



    @staticmethod
    def compute_reward_distributions(reward_tensor, batch):
        """
        Compute reward distributions for each group based on `uid`.

        Args:
            reward_tensor (torch.Tensor): A tensor of shape (batch_size, sequence_length) containing rewards.
            batch (list): A list of batch elements where each element has a `non_tensor_batch` attribute
                        containing the `uid` for grouping.

        Returns:
            dict: A dictionary mapping each `uid` to its reward distribution statistics (mean, std, min, max, and raw rewards).
        """
        # Step 1: Collect rewards for each group
        uid_to_rewards = defaultdict(list)

        for i in range(reward_tensor.shape[0]):
            uid = batch[i].non_tensor_batch['uid']
            # Since each reward tensor contains at most 1 nonzero value, we use `sum()` to extract that
            rewards = reward_tensor[i].sum().item()  # Sum of rewards for the current sequence.
            uid_to_rewards[uid].append(rewards)

        # Step 2: Compute the distribution for each group
        uid_to_distribution = {}

        for uid, rewards in uid_to_rewards.items():
            rewards_tensor = torch.tensor(rewards)
            mean = rewards_tensor.mean().item()
            std = rewards_tensor.std().item()
            min_val = rewards_tensor.min().item()
            max_val = rewards_tensor.max().item()
            
            uid_to_distribution[uid] = {
                'mean': mean,
                'std': std,
                'min': min_val,
                'max': max_val,
                'gap': max_val - min_val,
                'rewards': rewards  # Optionally, keep the raw rewards for further analysis
            }

        return uid_to_distribution
    
    @staticmethod
    def calculate_stats_by_score_bin(data):
        # Initialize bins for scores in [0, 0.1), [0.1, 0.2), ..., [0.9, 1.0]
        bins = {i: [] for i in range(10)}
        
        # Initialize lists for scores strictly equal to 0 and 1
        zero_scores = []
        one_scores = []
        
        for score, length in data:
            if score == 0:
                zero_scores.append(length)
            elif score == 1:
                one_scores.append(length)
            else:
                # Determine which bin the score falls into
                bin_index = int(score * 10)
                bins[bin_index].append(length)

        # Calculate the mean, max, and min length for each bin
        metrics = {}
        metric_name = 'score_to_gt_length'
        for bin_index, lengths in bins.items():
            bin_range = f"{(bin_index / 10):.1f}_to_{((bin_index + 1) / 10):.1f}"
            if lengths:
                metrics[f"{metric_name}/bin_{bin_range}/mean"] = sum(lengths) / len(lengths)
                metrics[f"{metric_name}/bin_{bin_range}/max"] = max(lengths)
                metrics[f"{metric_name}/bin_{bin_range}/min"] = min(lengths)
            else:
                metrics[f"{metric_name}/bin_{bin_range}/mean"] = 0
                metrics[f"{metric_name}/bin_{bin_range}/max"] = 0
                metrics[f"{metric_name}/bin_{bin_range}/min"] = 0
        
        # Calculate the mean, max, and min length for scores strictly equal to 0 and 1
        if zero_scores:
            metrics[f"{metric_name}/score_zero/mean"] = sum(zero_scores) / len(zero_scores)
            metrics[f"{metric_name}/score_zero/max"] = max(zero_scores)
            metrics[f"{metric_name}/score_zero/min"] = min(zero_scores)
        else:
            metrics[f"{metric_name}/score_zero/mean"] = 0
            metrics[f"{metric_name}/score_zero/max"] = 0
            metrics[f"{metric_name}/score_zero/min"] = 0
        
        if one_scores:
            metrics[f"{metric_name}/score_one/mean"] = sum(one_scores) / len(one_scores)
            metrics[f"{metric_name}/score_one/max"] = max(one_scores)
            metrics[f"{metric_name}/score_one/min"] = min(one_scores)
        else:
            metrics[f"{metric_name}/score_one/mean"] = 0
            metrics[f"{metric_name}/score_one/max"] = 0
            metrics[f"{metric_name}/score_one/min"] = 0
        return metrics
        
        # Calculate the mean, max, and min length for each bin
        # bin_stats = {}
        # for bin_index, lengths in bins.items():
        #     if lengths:
        #         mean_length = sum(lengths) / len(lengths)
        #         max_length = max(lengths)
        #         min_length = min(lengths)
        #         bin_stats[(bin_index / 10, (bin_index + 1) / 10)] = {
        #             'mean': mean_length,
        #             'max': max_length,
        #             'min': min_length
        #         }
        #     else:
        #         bin_stats[(bin_index / 10, (bin_index + 1) / 10)] = {
        #             'mean': 0,
        #             'max': 0,
        #             'min': 0
        #         }  # No data in this bin
        
        # # Calculate the mean, max, and min length for scores strictly equal to 0 and 1
        # mean_zero = sum(zero_scores) / len(zero_scores) if zero_scores else 0
        # max_zero = max(zero_scores) if zero_scores else 0
        # min_zero = min(zero_scores) if zero_scores else 0
        
        # mean_one = sum(one_scores) / len(one_scores) if one_scores else 0
        # max_one = max(one_scores) if one_scores else 0
        # min_one = min(one_scores) if one_scores else 0
        
        # return bin_stats, (mean_zero, max_zero, min_zero), (mean_one, max_one, min_one)

    def compute_gt_length_wrt_reward(self, reward_tensor, batch):
        # Step 1: Collect rewards for each group
        # uid_to_rewards = defaultdict(list)
        # uid_to_gt_len = defaultdict(int)

        reward_and_gtlen = list()

        for i in range(reward_tensor.shape[0]):
            reward = reward_tensor[i].sum().item()  # Sum of rewards for the current sequence.

            ground_truth = batch[i].non_tensor_batch['reward_model']['ground_truth'] 
            ground_truth_ids = self.tokenizer(ground_truth, return_tensors='pt', add_special_tokens=False)['input_ids'].tolist()[0]
            # uid_to_gt_len[uid] = len(ground_truth_ids)
            reward_and_gtlen.append((reward, len(ground_truth_ids)))

        return self.calculate_stats_by_score_bin(reward_and_gtlen)


    @staticmethod
    def compute_row_mean(responses, attention_mask):
        row_mean = []
        response_mask = attention_mask[:, -responses.shape[1]:]
        for i in range(responses.size(0)):
            # Get non-padded tokens for the current row
            non_padded_tokens = responses[i][response_mask[i] == 1]
            # Compute std for non-padded tokens
            if non_padded_tokens.numel() > 0:  # Ensure there are non-padded tokens
                row_mean.append(non_padded_tokens.mean())
            else:
                row_mean.append(torch.tensor(0.0))  # If all tokens are padded, std is 0
        return torch.stack(row_mean)
    # Function to compute row-wise std, ignoring padded values
    @staticmethod
    def compute_row_std(responses, attention_mask):
        row_std = []
        response_mask = attention_mask[:, -responses.shape[1]:]
        for i in range(responses.size(0)):
            # Get non-padded tokens for the current row
            non_padded_tokens = responses[i][response_mask[i] == 1]
            # Compute std for non-padded tokens
            if non_padded_tokens.numel() > 0:  # Ensure there are non-padded tokens
                row_std.append(non_padded_tokens.std())
            else:
                row_std.append(torch.tensor(0.0))  # If all tokens are padded, std is 0
        return torch.stack(row_std)

    @staticmethod
    def percentage_less_than_threshold(row_std, thresholds, prefix=""):
        results = {}
        for threshold in thresholds:
            percentage = (row_std < threshold).float().mean().item() # Convert to percentage
            results[prefix + f"std < {threshold:.2f}"] = percentage
        return results

    def sample_batch_data(self, batch, reward_tensor, scoreA_tensor, scoreB_tensor, advantage_list, entropy_list, format_reward_tensor, extracted_answer_list, N):
        # Extract all unique uids from the batch
        unique_uids = list(set(item.non_tensor_batch['uid'] for item in batch))
        
        # Ensure N is not greater than the number of unique uids
        N = min(N, len(unique_uids))
        
        # Randomly select N unique uids
        selected_uids = random.sample(unique_uids, N)
        

        # Initialize the log dictionary
        log_to_wandb = {
            "scores": [],
            "scoreAs": [],
            "scoreBs": [],
            "advantages": [],
            "entropies": [],
            "format_rewards": [],
            "data_sources": [],
            "sequences": [],
            "extracted_answers": [],
            "ground_truths": [],
            "final_reward_stds": [],
        }
        
        # Iterate over the batch and collect data for the selected uids
        for i_ in range(len(batch)):
            if batch[i_].non_tensor_batch['uid'] in selected_uids:
                score_ = reward_tensor[i_].sum().item()
                scoreA_ = scoreA_tensor[i_].sum().item()
                scoreB_ = scoreB_tensor[i_].sum().item()
                advantage_ = advantage_list[i_]
                entropy_ = entropy_list[i_]
                format_reward_ = format_reward_tensor[i_].sum().item()
                data_source_ = batch[i_].non_tensor_batch['data_source']
                sequence_ = self.tokenizer.decode(batch.batch['input_ids'][i_][batch.batch['attention_mask'][i_].bool()], skip_special_tokens=False)
                extracted_answer_ = extracted_answer_list[i_]
                ground_truth_ = batch[i_].non_tensor_batch['reward_model']['ground_truth']
                final_reward_std_ = batch[i_].batch['final_reward_stds'].item()
                
                # Append the data to the log dictionary
                log_to_wandb['scores'].append(score_)
                log_to_wandb['scoreAs'].append(scoreA_)
                log_to_wandb['scoreBs'].append(scoreB_)
                log_to_wandb['advantages'].append(advantage_)
                log_to_wandb['entropies'].append(entropy_)
                log_to_wandb['format_rewards'].append(format_reward_)
                log_to_wandb['data_sources'].append(data_source_)
                log_to_wandb['sequences'].append(sequence_)
                log_to_wandb['extracted_answers'].append(extracted_answer_)
                log_to_wandb['ground_truths'].append(ground_truth_)
                log_to_wandb['final_reward_stds'].append(final_reward_std_)
                selected_uids.remove(batch[i_].non_tensor_batch['uid'])
        
        return log_to_wandb



    def log_train_generations(self, batch, reward_tensor, scoreA_tensor, scoreB_tensor, advantage_list, entropy_list, format_reward_tensor, extracted_answer_list):
        uid_to_items = {}
        for i, item in enumerate(batch):
            uid = item.non_tensor_batch['uid']
            if uid not in uid_to_items:
                uid_to_items[uid] = []
            uid_to_items[uid].append(i)

        # Process items grouped by uid
        records = []
        for uid, item_indices in uid_to_items.items():
            for i in item_indices:
                record = {
                    'uid': uid,
                    'score': reward_tensor[i].sum().item(),
                    'scoreA': scoreA_tensor[i].sum().item(),
                    "final_reward_std": batch[i].batch['final_reward_stds'].item(),
                    'scoreB': scoreB_tensor[i].sum().item(),
                    'advantage': advantage_list[i],
                    'entropy': entropy_list[i],
                    'format_reward': format_reward_tensor[i].sum().item(),
                    'data_source': batch[i].non_tensor_batch['data_source'],
                    'sequence': self.tokenizer.decode(
                        batch.batch['input_ids'][i][batch.batch['attention_mask'][i].bool()], 
                        skip_special_tokens=False),
                    'extracted_answer': extracted_answer_list[i],
                    'ground_truth': batch[i].non_tensor_batch['reward_model']['ground_truth']
                }
                records.append(record)

        df = pd.DataFrame(records)
        df.sort_values('uid', inplace=True)

        # Create directory if it doesn't exist
        save_path = f"/data/logs/train_generations/{self.config.trainer.experiment_name}_step{self.global_steps}.csv"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save to CSV
        try:
            print(f"{save_path=}")
            df.to_csv(save_path, index=False, escapechar='\\')
        except Exception as e: # we meet errors like _csv.Error: need to escape, but no escapechar set
            print(f"Error saving train_generations: {e}")

    @staticmethod
    def sample_rewards(rewards, prompts, ground_truths, num_samples=1000):
        """
        Sample rewards based on cumulative probabilities and print sampled rewards with info.

        Parameters:
            rewards (list): List of reward values.
            num_samples (int): Number of samples to draw.

        Returns:
            sampled_rewards (list): List of sampled rewards.
        """
        rewards = [float(reward) for reward in rewards]
        # Calculate cumulative sum of rewards
        cumulative_rewards = np.cumsum(rewards)

        # Normalize the cumulative rewards to get probabilities
        probabilities = cumulative_rewards / cumulative_rewards[-1]

        # Perform sampling based on the cumulative probabilities
        samples = np.searchsorted(probabilities, np.random.random(num_samples))

        # Map the samples back to the original rewards
        sampled_rewards = [rewards[i] for i in samples]

        # Print sampled rewards with additional info
        print("Sampled Rewards with Info:")
        for i, reward in enumerate(sampled_rewards[:10]):  # Print first 10 samples for brevity
            idx = rewards.index(reward)  # Get the index of the reward in the original list
            prob = probabilities[idx] - (probabilities[idx - 1] if idx > 0 else 0)  # Calculate individual probability
            print(f"Sample {i + 1}: Reward = {reward}, Index = {idx}, Probability = {prob:.4f}")

        import matplotlib.pyplot as plt
        # Plot the histogram of the sampled rewards
        plt.hist(sampled_rewards, bins=len(rewards), edgecolor='black')
        plt.title('Histogram of Sampled Rewards')
        plt.xlabel('Reward')
        plt.ylabel('Frequency')
        plt.savefig("tmp/1.png")

        return sampled_rewards


    @staticmethod
    def sample_rewards_with_prompts_and_truths(rewards, prompts, ground_truths, num_samples=1000):
        """
        Sample rewards, prompts, and ground truths based on cumulative probabilities of rewards.
        Log the sampled data to a file.

        Parameters:
            rewards (list): List of reward values.
            prompts (list): List of prompts corresponding to the rewards.
            ground_truths (list): List of ground truths corresponding to the rewards.
            num_samples (int): Number of samples to draw.

        Returns:
            sampled_rewards (list): List of sampled rewards.
            sampled_prompts (list): List of sampled prompts.
            sampled_truths (list): List of sampled ground truths.
        """
        # Ensure all lists are of the same length
        if not (len(rewards) == len(prompts) == len(ground_truths)):
            raise ValueError("All input lists (rewards, prompts, ground_truths) must be of the same length.")

        # Calculate cumulative sum of rewards
        cumulative_rewards = np.cumsum(rewards)

        # Normalize the cumulative rewards to get probabilities
        probabilities = cumulative_rewards / cumulative_rewards[-1]

        # Perform sampling based on the cumulative probabilities
        samples = np.searchsorted(probabilities, np.random.random(num_samples))

        # Map the samples back to the original rewards, prompts, and ground truths
        sampled_rewards = [rewards[i] for i in samples]
        sampled_prompts = [prompts[i] for i in samples]
        sampled_truths = [ground_truths[i] for i in samples]
        sampled_probs = [probabilities[i] - (probabilities[i - 1] if i > 0 else 0) for i in samples]

        # Create a pandas DataFrame
        df = pd.DataFrame({
            "reward": sampled_rewards,
            "prompt": sampled_prompts,
            "ground_truth": sampled_truths,
            "probability": sampled_probs
        })
        # Save the DataFrame to a Parquet file
        parquet_file = "tmp/scoreAs_prompts_ground_truths.csv"
        df.to_csv(parquet_file, index=False, escapechar='\\')

        # Create a DataFrame for the original data
        original_df = pd.DataFrame({
            "reward": rewards,
            "prompt": prompts,
            "ground_truth": ground_truths,
            "probability": [probabilities[i] - (probabilities[i - 1] if i > 0 else 0) for i in range(len(rewards))],
            "type": "original"  # Add a column to indicate this is original data
        })
        # Save the DataFrame to a Parquet file
        original_parquet_file = "tmp/scoreAs_prompts_ground_truths_full.csv"
        original_df.to_csv(original_parquet_file, index=False, escapechar='\\')

        # Create the `tmp` directory if it doesn't exist
        os.makedirs("tmp", exist_ok=True)

        # Log the sampled data to a file
        log_file = "tmp/scoreAs_prompts_ground_truths.txt"
        with open(log_file, "w") as f:
            f.write("Sampled Rewards, Prompts, and Ground Truths:\n")
            for i in range(num_samples):
                idx = samples[i]  # Get the index of the sample
                reward = rewards[idx]
                prompt = prompts[idx]
                truth = ground_truths[idx]
                prob = probabilities[idx] - (probabilities[idx - 1] if idx > 0 else 0)  # Calculate individual probability
                f.write(f"Sample {i + 1}: Reward = {reward}, Prompt = '{repr(prompt)}', Ground Truth = '{repr(truth)}', Probability = {prob:.4f}\n")
            f.write("\n")  # Add a newline for separation between runs

        # Print the first 10 samples for quick verification
        print("First 10 Sampled Rewards, Prompts, and Ground Truths:")
        for i in range(min(10, num_samples)):
            idx = samples[i]
            reward = rewards[idx]
            prompt = prompts[idx]
            truth = ground_truths[idx]
            prob = probabilities[idx] - (probabilities[idx - 1] if idx > 0 else 0)
            print(f"Sample {i + 1}: Reward = {reward}, Prompt = '{repr(prompt)}', Ground Truth = '{truth}', Probability = {prob:.4f}")

        import matplotlib.pyplot as plt
        # Plot the histogram of the sampled rewards
        # plt.hist(sampled_rewards, bins=50, edgecolor='black')
        plt.hist(rewards, bins=50, edgecolor='black')
        plt.title(f'Histogram of Rewards (total={len(rewards)})')
        plt.xlabel('Reward')
        plt.ylabel('Frequency')
        plt.savefig('tmp/scoreAs_prompts_ground_truths.jpg')

        return sampled_rewards, sampled_prompts, sampled_truths




    @staticmethod
    def compute_scoreB_by_data_source_metrics(batch, scoreB_tensor, name='scoreB'):
        # Initialize a dictionary to store scoreB_reward values for each data_source
        scoreB_by_source = defaultdict(list)
        
        # Iterate over the batch and collect scoreB_reward for each data_source
        for i in range(len(batch)):
            data_source = batch[i].non_tensor_batch['data_source']
            scoreB = scoreB_tensor[i].sum(-1).detach().item()
            scoreB_by_source[data_source].append(scoreB)
        
        # Calculate mean, max, and min for each data_source and format the keys
        metrics_by_source = {}
        for data_source, rewards in scoreB_by_source.items():
            rewards_tensor = torch.tensor(rewards)
            metrics_by_source[f'critic_wrt_data_source/{name}/{data_source}/mean'] = torch.mean(rewards_tensor).item()
            # metrics_by_source[f'critic_wrt_data_source/scoreB/{data_source}/max'] = torch.max(rewards_tensor).item()
            # metrics_by_source[f'critic_wrt_data_source/scoreB/{data_source}/min'] = torch.min(rewards_tensor).item()
        
        return metrics_by_source

    def compute_promptgt2scoreA(self, epoch: int) -> None:
        """
        Processes and logs the distribution of scoreA for the given epoch.
        
        Args:
            epoch (int): The current epoch number.
        """
        # Check if probabilistic reward is enabled in the configuration
        if 'prob' not in self.config.reward_model.reward_manager:
            return

        # Set the seed for reproducibility
        current_seed = self.config.data.get('seed', 1) if epoch == 0 else random.randint(0, 2**32 - 1)
        if self.config.data.shuffle:
            self.train_dataloader.sampler.generator.manual_seed(current_seed)

        # Initialize containers for storing data
        promptgt2scoreA = {}
        promptgt2datasource = {}
        promptgt2probsingt = {}
        scoreA_list, prompt_str_list, ground_truth_list = [], [], []

        total_train_samples = len(self.train_dataloader.dataset)

        # Iterate through the training data
        if total_train_samples < 100000:
            print(f"The total training samples is small, we will compute scoreAs for all of them")
            # while len(scoreA_list) < len(self.train_dataloader.dataset):
            # max_try = 3
            # current_try = 0
            while True:
                break_flag = True
                for idx, batch_dict in tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader)):
                    print(f"{len(promptgt2scoreA)=} {len(scoreA_list)=}. The goal is {total_train_samples}")
                    batch: DataProto = DataProto.from_single_dict(batch_dict)
                    scoreAs, prompt_strs, ground_truths, old_log_probs_in_gt_list = self.get_scoreA(batch)

                    # Debugging block (optional)
                    debug_todo = False
                    if debug_todo:
                        print(f"{len(scoreAs)=} {len(prompt_strs)=} {len(ground_truths)=}")

                    # Process each item in the batch
                    for i in range(len(batch)):
                        prompt = self.tokenizer.decode(
                            batch.batch[i]['input_ids'], 
                            skip_special_tokens=False
                        ).replace(self.tokenizer.pad_token, '')
                        ground_truth = batch[i].non_tensor_batch['reward_model']['ground_truth']
                        key = prompt + ground_truth
                        if key not in promptgt2scoreA:
                            break_flag = False
                            promptgt2scoreA[prompt + ground_truth] = scoreAs[i]
                            scoreA_list.append(scoreAs[i])
                    # scoreA_list.extend(scoreAs)
                    if len(promptgt2scoreA) >= total_train_samples:
                        break
                if len(promptgt2scoreA) >= total_train_samples:
                    break
                if break_flag:
                    print(f"No update, exiting...")
                    break
                # current_try += 1
                # if current_try >= max_try:
                #     break
        else:
            for idx, batch_dict in tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader)):
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                scoreAs, prompt_strs, ground_truths, old_log_probs_in_gt_list = self.get_scoreA(batch)

                # Debugging block (optional)
                debug_todo = False
                if debug_todo:
                    print(f"{len(scoreAs)=} {len(prompt_strs)=} {len(ground_truths)=}")

                # Process each item in the batch
                for i in range(len(batch)):
                    prompt = self.tokenizer.decode(
                        batch.batch[i]['input_ids'], 
                        skip_special_tokens=False
                    ).replace(self.tokenizer.pad_token, '')
                    ground_truth = batch[i].non_tensor_batch['reward_model']['ground_truth']
                    promptgt2scoreA[prompt + ground_truth] = scoreAs[i]
                    # For flan only
                    debug_todo1 = False
                    if debug_todo1:
                        # promptgt2datasource[prompt + ground_truth] = batch[i].non_tensor_batch['extra_info']['task']
                        promptgt2datasource[prompt + ground_truth] = batch[i].non_tensor_batch['data_source']
                        promptgt2probsingt[prompt + ground_truth] = np.exp(old_log_probs_in_gt_list[i].numpy()).tolist()
                scoreA_list.extend(scoreAs)

        # Debugging block (optional)
        debug_todo1 = False
        if debug_todo1:
            print(f"{len(promptgt2scoreA)=}")
            print(f"{len(promptgt2datasource)=}")
            print(f"{len(promptgt2probsingt)=}")
            with open('tmp/qwen2_tulunosingle_promptgt2scoreA.json', 'w') as file:
                json.dump(promptgt2scoreA, file)
            with open('tmp/qwen2_tulunosingle_promptgt2datasource.json', 'w') as file:
                json.dump(promptgt2datasource, file)
            with open('tmp/qwen2_tulunosingle_promptgt2probsingt.json', 'w') as file:
                json.dump(promptgt2probsingt, file)

        save_path = '/data/logs/promptgt2scoreA.json'
        with open(save_path, 'w') as file:
            print(f"We dump to {save_path}")
            json.dump(promptgt2scoreA, file)
            # assert False
        # Log the scoreA distribution to WandB
        self._maybe_log_histogram_to_wandb(
            scoreA_list, 
            'figures/scoreA', 
            'scoreA', 
            'Score A Distribution'
        )

        # Reset the seed to ensure consistent data order for training
        if self.config.data.shuffle:
            self.train_dataloader.sampler.generator.manual_seed(current_seed)

        print(f"{len(promptgt2scoreA)=}")
        return promptgt2scoreA

    def save_rollout(self, batch):
        print(f"{batch=}")

        prompts = self.tokenizer.batch_decode(
            batch.batch['prompts'], 
            skip_special_tokens=True
        )
        responses = self.tokenizer.batch_decode(
            batch.batch['responses'], 
            skip_special_tokens=True
        )

        results = []
        for i in range(len(prompts)):
            results.append({
                "Index": i,
                "Prompt": prompts[i],
                "Response": responses[i],
            })
        
        # Create directory if it doesn't exist
        save_path = f"/data/logs/rollout/{self.config.trainer.experiment_name}_step{self.global_steps}.csv"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        df = pd.DataFrame(results)
        # Save to CSV
        print(f"{save_path=}")
        df.to_csv(save_path, index=False, escapechar='\\')
        
    @staticmethod
    def format_reward(predict_str: str) -> float:
        def _validate_tags(input_string):
            tags = ['<think>', '</think>', '<answer>', '</answer>']
            for tag in tags:
                if input_string.count(tag) != 1:
                    return 0.0
            return 1.0

        if _validate_tags(predict_str) == 0.0:
            return 0.0
        pattern = re.compile(r'<think>.*</think>.*<answer>.*</answer>.*', re.DOTALL)
        match_result = re.fullmatch(pattern, predict_str)

        return 1.0 if match_result else 0.0

    def compute_think_answer_length_metrics(self, batch) -> Dict[str, float]:
        """
        Compute token length statistics for text within <think> and <answer> tags.
        If the text does not match the expected pattern, counts are set to 0.
        
        Args:
            batch: Contains response data with 'responses' field of token IDs
            
        Returns:
            Dictionary with mean/max/min token lengths for think and answer sections.
            If no matches exist, all metrics will be 0.
        """
        output_ids = batch.batch['responses']
        output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
        
        think_lengths = []
        answer_lengths = []
        
        pattern = re.compile(r'.*<think>(.*)</think>.*<answer>(.*)</answer>.*', re.DOTALL)
        
        for text in output_texts:
            match = pattern.fullmatch(text)
            
            if match:
                think_text, answer_text = match.groups()
                think_tokens = self.tokenizer.tokenize(think_text.strip())
                answer_tokens = self.tokenizer.tokenize(answer_text.strip())
                
                think_lengths.append(len(think_tokens))
                answer_lengths.append(len(answer_tokens))
            else:
                think_lengths.append(0)
                answer_lengths.append(0)
        
        # Compute statistics (if no entries, np.mean/max/min will return NaN, so we handle that)
        def safe_stats(values: List[int]) -> Dict[str, float]:
            if not values:  # Empty list
                return {"mean": 0.0, "max": 0.0, "min": 0.0}
            return {
                "mean": float(np.mean(values)),
                "max": float(np.max(values)),
                "min": float(np.min(values)),
            }
        
        think_stats = safe_stats(think_lengths)
        answer_stats = safe_stats(answer_lengths)
        
        return {
            "think_length/mean": think_stats["mean"],
            "think_length/max": think_stats["max"],
            "think_length/min": think_stats["min"],
            "answer_length/mean": answer_stats["mean"],
            "answer_length/max": answer_stats["max"],
            "answer_length/min": answer_stats["min"],
        }
