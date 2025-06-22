import os
import re
import copy
import pandas as pd
import torch
from verl import DataProto
from datetime import datetime
from typing import Literal, Tuple
from functools import partial
from .ce import *
from .naive import *


class MixRewardManager(object):

    def __init__(
        self, tokenizer, num_examine, compute_exact_score_func=None,
        compute_fuzzy_score_name=None, shaping_function_name=None, discrete_function_name=None, format_coefficient=0.1,
        n_rollouts=None, mix_type: Literal['soft', 'hard']='hard', pr_weight=0.5, vr_weight=1.0
    ) -> None:
        self.vr_manager = NaiveRewardManager(
            tokenizer=tokenizer,
            num_examine=num_examine,
            compute_score=compute_exact_score_func,
        )
        self.pr_manager = CERewardManager(
            tokenizer=tokenizer,
            num_examine=num_examine,
            compute_score_name=compute_fuzzy_score_name,
            shaping_function_name=shaping_function_name,
            discrete_function_name=discrete_function_name,
            format_coefficient=format_coefficient,
        )
        self.n_rollouts = n_rollouts
        self.mix_type = mix_type
        self.pr_weight = pr_weight
        self.vr_weight = vr_weight

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # import debugpy
        # try:
            # debugpy.listen(("localhost", 3000))
            # print("Waiting for debugger attach")
            # debugpy.wait_for_client()
        # except Exception as e:
            # pass

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        # if 'rm_scores' in data.batch.keys():
            # return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        format_reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        exact_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        fuzzy_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        pr_scoreB_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        pr_scoreA_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        pr_reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        vr_reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        extracted_answer_list = [None] * len(data)
        max_response_length = data.batch['responses'].shape[-1]

        vr_reward_tensor_, exact_tensor_, _, format_reward_tensor_, extracted_answer_list_ = self.vr_manager(data)
        pr_reward_tensor_, pr_scoreB_tensor_, pr_scoreA_tensor_, _, _ = self.pr_manager(data)
        fuzzy_tensor_ = pr_scoreB_tensor_ - pr_scoreA_tensor_

        assert len(data) % self.n_rollouts == 0, 'len(data) is need to be divisible by n_rollouts in MixRewardManager'
        grouped_index = sorted(list(range(len(data))), key=lambda x: data[x].non_tensor_batch['uid'])
        inv_grouped_index = [None] * len(data)
        for i in range(len(data)):
            inv_grouped_index[grouped_index[i]] = i
            format_reward_tensor[i] = format_reward_tensor_[grouped_index[i]]
            exact_tensor[i] = exact_tensor_[grouped_index[i]]
            fuzzy_tensor[i] = fuzzy_tensor_[grouped_index[i]]
            pr_scoreB_tensor[i] = pr_scoreB_tensor_[grouped_index[i]]
            pr_scoreA_tensor[i] = pr_scoreA_tensor_[grouped_index[i]]
            extracted_answer_list[i] = extracted_answer_list_[grouped_index[i]]
            pr_reward_tensor[i] = pr_reward_tensor_[grouped_index[i]]
            vr_reward_tensor[i] = vr_reward_tensor_[grouped_index[i]]

        straightA_tensor = ( # [bs], 1 for all correct, -1 for all wrong, 0 for mixed
            exact_tensor.sum(dim=-1).view(-1, self.n_rollouts).sum(dim=-1) == float(self.n_rollouts)
        ).float() - (
            exact_tensor.sum(dim=-1).view(-1, self.n_rollouts).sum(dim=-1) == 0.
        ).float()
        straightA_tensor = straightA_tensor.broadcast_to(self.n_rollouts, max_response_length, -1)
        straightA_tensor = straightA_tensor.permute(2, 0, 1).contiguous().view(-1, max_response_length) # [bs * n_rollouts, max_res_len]

        if self.mix_type == 'hard':
            assert False
            need_to_use_pr = (exact_tensor.sum(dim=-1).view(-1, self.n_rollouts).sum(dim=-1) == 0.) # [bs]
            need_to_use_pr = need_to_use_pr.broadcast_to(self.n_rollouts, max_response_length, -1)
            need_to_use_pr = need_to_use_pr.permute(2, 0, 1).contiguous().view(-1, max_response_length) # [bs * n_rollouts, max_res_len]
            reward_tensor = torch.where(need_to_use_pr, pr_reward_tensor, vr_reward_tensor) # [bs * n_rollouts, max_res_len]
            scoreB_tensor = torch.where(need_to_use_pr, pr_scoreB_tensor, exact_tensor)
            scoreA_tensor = torch.where(need_to_use_pr, pr_scoreA_tensor, torch.zeros_like(data.batch['responses']))
        elif self.mix_type == 'soft':
            # exact_tensor: 0/1, fuzzy_tensor: scoreB - scoreA
            reward_tensor = self.vr_weight * vr_reward_tensor + self.pr_weight * pr_reward_tensor
            scoreB_tensor = pr_scoreB_tensor
            scoreA_tensor = pr_scoreA_tensor
        else:
            raise NotImplementedError

        # if self.format_coefficient == -1:
            # reward_tensor = torch.where(
                # score_tensor != 0,   # only one pos has value in single rollout
                # torch.where(
                    # format_reward_tensor == 1,
                    # score_tensor,
                    # -torch.ones_like(data.batch['responses']),
                # ),
                # torch.zeros_like(data.batch['responses'])
            # )
        # else:
            # reward_tensor = torch.where(
                # score_tensor != 0,    # only one pos has value in single rollout
                # (1 - self.format_coefficient) * score_tensor + self.format_coefficient * format_reward_tensor,
                # torch.zeros_like(data.batch['responses'])
            # )

        def inv_grouped_tensor(*grouped_tensors):
            for t in grouped_tensors:
                if isinstance(t, torch.Tensor):
                    grouped_t = torch.empty_like(t).copy_(t)
                else:
                    grouped_t = copy.deepcopy(t)
                for i in range(len(t)):
                    t[i] = grouped_t[inv_grouped_index[i]]
            return grouped_tensors

        # reward_tensor: score(vr | pr) + format    [bs * n_rollouts, max_res_len]
        # scoreB_tensor: vr_Acc | pr_scoreB         [bs * n_rollouts, max_res_len]
        # scoreA_tensor: 0 | pr_scoreA              [bs * n_rollouts, max_res_len]
        # format_reward_tensor                      [bs * n_rollouts, max_res_len]
        # extracted_answer_list: str list           [bs * n_rollouts]
        # straightA_tensor: 1/-1/0                  [bs * n_rollouts, max_res_len]
        # exact_tensor: acc, 0/1                    [bs * n_rollouts, max_res_len]
        # pr_scoreB_tensor                          [bs * n_rollouts, max_res_len]
        # pr_scoreA_tensor                          [bs * n_rollouts, max_res_len]

        # maintaining that final_score = scoreB_tensor - scoreA_tensor
        return inv_grouped_tensor(
            reward_tensor, scoreB_tensor, scoreA_tensor, format_reward_tensor,
            extracted_answer_list, straightA_tensor, exact_tensor, pr_scoreB_tensor, pr_scoreA_tensor,
            pr_reward_tensor, vr_reward_tensor
        )
