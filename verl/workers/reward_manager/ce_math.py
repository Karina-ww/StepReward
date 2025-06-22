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

from verl import DataProto
from verl.utils.reward_score import _default_compute_score
import torch
import numpy as np
from functools import partial


def sigmoid_k(x, k=6):
    """
    Maps a value x in [0, 1] to [0, 1] using a sigmoid-like S-shaped curve.

    Parameters:
        x (float): Input value in the range [0, 1].
        k (float): Steepness parameter for the sigmoid curve. Higher values make the curve steeper.

    Returns:
        float: Mapped value in the range [0, 1].
    """
    if k == 0: # vanilla sigmoid
        return 1 / (1 + np.exp(-x))

    # Shift and scale x to the range [-k, k]
    x_scaled = k * (2 * x - 1)
    
    # Apply the sigmoid function
    sigmoid = 1 / (1 + np.exp(-x_scaled))
    
    # Scale the output to [0, 1]
    return sigmoid

def threshold_t_sigmoid_k(x, t, k=6):
    """
    Maps a value x in [0, 1] to [0, 1] using a sigmoid-like S-shaped curve.

    Parameters:
        x (float): Input value in the range [0, 1].
        k (float): Steepness parameter for the sigmoid curve. Higher values make the curve steeper.

    Returns:
        float: Mapped value in the range [0, 1].
    """
    # Shift and scale x to the range [-k, k]
    x_scaled = k * (2 * x - 1)
    
    # Apply the sigmoid function
    sigmoid = 1 / (1 + np.exp(-x_scaled))

    result = 0 if sigmoid < t else sigmoid
    
    return result


def threshold_t_sigmoidv2_k(x, t, k=6):
    # concave curve
    if x < t:
        result = 0
    else:
        x = x - t
        x = x * k
        result = 1 / (1 + np.exp(-x))
    return result


def threshold_t_sigmoidv2fixed_k(x, t, k=6):
    if x < t:
        result = 0
    else:
        x = (x- t) * k
        result = 1 / (1 + np.exp(-x))  * ((1 - t) / 0.5) - ((1-t) - t) 
    
    return result


def threshold_t_sigmoidv3_k(x, t, k=6):
    # convex curve
    if x < t:
        result = 0
    else:
        x = (x - 1) * k
        result = 1 / (1 + np.exp(-x)) + 0.5
    return result


def leaky_relu_like(score, threshold, alpha=0.01):
    """
    Maps a score from [0, 1] to [0, 1] using a Leaky ReLU-like function.

    Parameters:
    - score: The input score in the range [0, 1].
    - threshold: The threshold below which the score is scaled.
    - alpha: The slope for scores below the threshold (default is 0.01).

    Returns:
    - The transformed score in the range [0, 1].
    """
    if score < threshold:
        return alpha * score
    else:
        return score


def threshold_t_tanh_k(score, t, k=6):
    # Apply tanh transformation with a configurable scaling factor
    transformed_score = (np.tanh(score * k - k / 2) + 1) / 2
    
    # Threshold values smaller than 0.05 to 0
    if transformed_score < t:
        transformed_score = 0
    
    return transformed_score

class CEMathRewardManager:
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, compute_score, compute_score_name=None, shaping_function_name=None, save_results_dir=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        # assert compute_score is None
        self.compute_score = compute_score or _default_compute_score
        # self.compute_score = compute_score or _ce_compute_score
        self.compute_score_name = compute_score_name
        print(f"{shaping_function_name=}")
        if shaping_function_name == 'identity':
            self.shaping_function = lambda x:x
        elif shaping_function_name.startswith('threshold'):
            threshold = float(shaping_function_name.split('_')[-1])
            self.shaping_function = lambda x: 0 if x < threshold else x
        elif shaping_function_name.startswith('sigmoid_'):
            print(f"Selecting sigmoid_k function.")
            k = float(shaping_function_name.split('_')[-1])
            self.shaping_function = partial(sigmoid_k, k=k)
        elif shaping_function_name.startswith('leaky_'):
            # e.g., leaky_0.05
            print(f"Using leaky-relu like function")
            threshold = float(shaping_function_name.split('_')[1])
            self.shaping_function = partial(leaky_relu_like, threshold=threshold)
        elif shaping_function_name.startswith('comp'): # compound
            # comp_threshold_0.3_sigmoid_6
            threshold = float(shaping_function_name.split('_')[2])
            k = float(shaping_function_name.split('_')[4])
            if 'sigmoidv2fixed' in shaping_function_name:
                print(f"Using sigmoid v2fixed")
                self.shaping_function = partial(threshold_t_sigmoidv2fixed_k, t=threshold, k=k)
            elif 'sigmoidv3' in shaping_function_name:
                print(f"Using sigmoid v3")
                self.shaping_function = partial(threshold_t_sigmoidv3_k, t=threshold, k=k)
            elif 'sigmoidv2' in shaping_function_name:
                print(f"Using sigmoid v2")
                self.shaping_function = partial(threshold_t_sigmoidv2_k, t=threshold, k=k)
            elif 'sigmoid' in shaping_function_name:
                print(f"Using sigmoid v1")
                self.shaping_function = partial(threshold_t_sigmoid_k, t=threshold, k=k)
            elif 'tanh' in shaping_function_name:
                self.shaping_function = partial(threshold_t_tanh_k, t=threshold, k=k)
            else:
                raise ValueError
        else:
            print(f"{shaping_function_name=}")
            raise NotImplementedError(f"{shaping_function_name=}")
        for i in [-1, 0, 0.05, 0.15, 0.25, 0.35, 0.45, 0.75, 0.85, 0.95, 1, 2]:
            print(f"{i=} {self.shaping_function(i)=}", flush=True)

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        # print(f"Inside ce reward mangaer: {data=}")
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses'] # len(response_ids): 1024
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum() # 329
            valid_response_ids = response_ids[:valid_response_length] 

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences) # <|im_start|>system\n..the answer is: \boxed{10, 30, 40}<|im_end|>.

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            data_source = data_item.non_tensor_batch['data_source']
            extra_info = data_item.non_tensor_batch.get('extra_info', None)


            if data_source in [
                    'numina_aops_forum', 'numina_synthetic_math', 'numina_amc_aime', 'numina_synthetic_amc', 'numina_cn_k12',
                    'numina_olympiads', 'allenai-tulu-3-sft-mixture-train-math',
            ]:
                score = self.compute_score(
                    data_source=data_source,
                    # solution_str=sequences_str,
                    solution_str=self.tokenizer.decode(valid_response_ids, skip_special_tokens=True),
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                    prompt_str=self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True),
                )
                # print(f"{__file__} We call {self.compute_score=} for {data_source=} with {score=}")
            else:
                old_log_probs = data_item.batch['old_log_probs_ce'] # shape: [1024]
                ground_truth_mask = data_item.batch['ground_truth_mask_ce']
                if ground_truth_mask.sum() == 0:
                    score = 0
                else:
                    # print(f"{ground_truth_mask.shape=} {ground_truth_mask=}")
                    old_log_probs_in_gt = old_log_probs[ground_truth_mask.bool()]
                    # loss = -torch.mean(log_probs_in_gt)
                    # score = np.exp(-loss)
                    if self.compute_score_name == 'mean_exp_log_softmax':
                        # print(f"We use mean_exp_log_softmax")
                        score = torch.mean(torch.exp(old_log_probs_in_gt)).item()
                    elif self.compute_score_name == 'mean_log_softmax':
                        # print(f"We use mean_log_softmax")
                        score = torch.mean(old_log_probs_in_gt).item()
                    elif self.compute_score_name == 'exp_sum_log_softmax':
                        # score = torch.mean(torch.exp(old_log_probs_in_gt)).item()
                        # print(f"We use exp_sum_log_softmax")
                        score = torch.exp(torch.sum(old_log_probs_in_gt)).item()
                    elif self.compute_score_name == 'exp_mean_log_softmax':
                        # score = torch.mean(torch.exp(old_log_probs_in_gt)).item()
                        # print(f"We use exp_sum_log_softmax")
                        score = torch.exp(torch.mean(old_log_probs_in_gt)).item()
                    else:
                        raise ValueError
                    # print(f"{score=}")
                    score = self.shaping_function(score)
                # print(f"{__file__} We call ce reward for {data_source=} with {score=}")
            reward_tensor[i, valid_response_length - 1] = score

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(f"{data_source=} {sequences_str=}", flush=True)

            debug = False
            if debug:
                if score == 1:
                    print(f"="*50)
                    print(__file__)
                    print(f"{data_item.batch=}", flush=True)
                    print(f"{old_log_probs=}", flush=True)
                    print(f"{old_log_probs_in_gt=}", flush=True)
                    print(f"{score=}", flush=True)
                    print(f"{data_source=} {sequences_str=}", flush=True)
                    model_output = self.tokenizer.decode(data_item.batch['input_ids'][data_item.batch['attention_mask'] == 1])
                    print(f"{model_output=}", flush=True)
                    concat_gt_output = self.tokenizer.decode(data_item.batch['input_ids_ce'][data_item.batch['attention_mask_ce'] == 1])
                    print(f"{concat_gt_output=}", flush=True)
                    ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
                    print(f"{ground_truth=}")
                    print(f"="*50)


        return reward_tensor


            # data_item.batch:
            # 'input_ids': torch.Size([1536]); tensor([151643, 151643, 151643,  ..., 151643, 151643, 151643])
            # 'position_ids': torch.Size([1536]); tensor([   0,    0,    0,  ..., 1177, 1178, 1179])
            # 'attention_mask': torch.Size([1536]); tensor([0, 0, 0,  ..., 0, 0, 0])
            # 'responses': torch.Size([1024]); tensor([ 22519,   3580,  19122,  ..., 151643, 151643, 151643])
            # 'prompts': torch.Size([512]); tensor([ 151643,   ...,  151643,  8948, 271, ..., 77091, 198])
            # 'old_log_probs': torch.Size([1024]); tensor([-0.8154, -0.0071, -0.0011,  ...,  0.0000,  0.0000,  0.0000])
            # 'ref_log_prob': torch.Size([1024]); tensor([-0.7590, -0.0066, -0.0009,  ...,  0.0000,  0.0000,  0.0000])
            # 'all_old_logits': torch.Size([1024, 152064]); tensor([[ 3.0938, 5.0000, 10.5625, ..., -0.2949, -0.2949, -0.2949], ..., 0.0000, 0.0000, 0.0000]],dtype=torch.bfloat16)



            # ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth'] # 30, 10, 40

            # data_item.non_tensor_batch:
            # 'data_source': numina_synthetic_math
            # 'ability': math
            # 'reward_model': {'ground_truth': '30, 10, 40', 'style': 'rule'}
            # 'extra_info': {'index': 0, 'split': 'dummy'}
            # 'index': 0 # TODO: no shuffle?
            # 'uid': 0bbc17c5-93ef-447c-b5c1-6c9e966a0af3

            # # for k, v in data_item.non_tensor_batch.items():
            # #     print('='*50)
            # #     print(k)
            # #     if hasattr(v, 'shape'):
            # #         print(v.shape)
            # #     print(v)
            # #     print('='*50)
            # # print(f"ground-truth: {ground_truth}")
            # old_log_probs = data_item.batch['old_log_probs'] 
            # valid_old_log_probs = old_log_probs[:valid_response_length]
            # # print(f"valid_old_log_probs: {valid_old_log_probs}") # tensor([-8.1543e-01, -7.0915e-03, ... -0.0000e+00, -7.0992e-02, -1.4114e-04])
            # # print(f"invalid_old_log_probs: {old_log_probs[valid_response_length:]}") # tensor([-36.8125,   0.0000,   0.0000,   0.0000, ..., 0.0000])
            # # all_old_logits = data_item.batch['all_old_logits'] 
            # # valid_all_old_logits = all_old_logits[:valid_response_length]
            # # print(f"valid_all_old_logits.shape: {valid_all_old_logits.shape}") # torch.Size([329, 152064])
            # # print(f"valid_all_old_logits: {valid_all_old_logits}") # tensor([[ 3.0938, 5.0000, 10.5625, ..., 2.1875, -2.1875, -2.1875]],dtype=torch.bfloat16)
            # # print(f"invalid_all_old_logits: {all_old_logits[valid_response_length:]}") # tensor([[ 6.5000, -1.8203, 6.8438, ..., -1.4141, ...,  0.0000, 0.0000]],dtype=torch.bfloat16)
            # # tensor([-0.8154, -0.0071, -0.0011,  ...,  0.0000,  0.0000,  0.0000])
            # # shape: torch.Size([1024])

            # # print("Should we clip the old_log_probs?")
            # # (main_task pid=1376076) ground-truth: 30, 10, 40
            # # (main_task pid=1376076) old_log_probs: tensor([-0.8154, -0.0071, -0.0011,  ...,  0.0000,  0.0000,  0.0000])



            # extra_info = data_item.non_tensor_batch.get('extra_info', None)
            # extra_info['tokenizer'] = self.tokenizer
            # extra_info['valid_old_log_probs'] = valid_old_log_probs
            # # print('*'*50)
            # # print(f"valid_all_old_logits: {valid_all_old_logits}")
            # # print(f"valid_all_old_logits.device: {valid_all_old_logits.device}") # cpu
            # # print(f"valid_all_old_logits.shape: {valid_all_old_logits.shape}") #  torch.Size([329, 152064])
            # # print('*'*50)
            # # extra_info['valid_all_old_logits'] = valid_all_old_logits
            # extra_info['valid_response_ids'] = valid_response_ids
            # # extra_info['valid_response_str'] = self.tokenizer.decode(valid_response_ids)

            # score = self.compute_score(
            #     data_source=data_source,
            #     solution_str=sequences_str,
            #     ground_truth=ground_truth,
            #     extra_info=extra_info,
            # )
            # print(f"Deleting all old logits...")
            # del valid_all_old_logits
            # match_tokens = self.tokenizer.tokenize(r'\boxed{}')
            # match_tokens: ['\\', 'boxed', '{}']
            # match_token_ids: [59, 79075, 6257]
            # print(f"match_tokens: {match_tokens}") # ['\\', 'boxed', '{}']
            # match_token_ids = self.tokenizer.convert_tokens_to_ids(match_tokens)
            # print(f"match_token_ids: {match_token_ids}") # [59, 79075, 6257]
            # match_tokens_start = self.tokenizer.tokenize(r'\boxed{')
            # match_tokens_end = self.tokenizer.tokenize(r'}')
            # match_token_ids_start = self.tokenizer.convert_tokens_to_ids(match_tokens_start)
            # match_token_ids_end = self.tokenizer.convert_tokens_to_ids(match_tokens_end)
            # match_token_ids_start: [59, 79075, 90]
            # match_token_ids_end: [92]
            # print(f"match_token_ids_start: {match_token_ids_start}") # [59, 79075, 90]
            # print(f"match_token_ids_end: {match_token_ids_end}") # [92]

            # start_pos, end_pos = None, None
            # for i in range(len(response_ids) - len(match_token_ids), -1, -1):
            #     if response_ids[i:i + len(match_token_ids)] == match_token_ids:
            #         start_pos = i
            #         end_pos = i + len(match_token_ids)
            #         break
            # print('-'*50)
            # print(f"match_token_ids: {match_token_ids}")
            # print('-'*50)

