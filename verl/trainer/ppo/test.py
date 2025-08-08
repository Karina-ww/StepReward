# import torch

# class PeakPicker:
#     def _pick_topk_peaks_with_gap(self, entropy_vec: torch.Tensor, valid_len: int, k: int, min_gap: int) -> list[int]:
#         """
#         entropy_vec: (L_resp,) 的熵（float）向量
#         valid_len:   有效 response token 数
#         返回升序的切点(以 response 内部 token 下标计，范围 [0, valid_len-1])，长度<=k
#         """
#         L = int(valid_len)
#         scores = entropy_vec[:L].detach().cpu().tolist()
#         idxs = list(range(L))

#         # 贪心 + 抑制窗口
#         selected = []
#         used = [False] * L
#         for _ in range(k):
#             best = -1
#             best_s = float('-inf')
#             for i, s in enumerate(scores):
#                 if not used[i] and s > best_s:
#                     best_s, best = s, i
#             if best < 0:
#                 break
#             selected.append(best)
#             # 抑制窗口
#             left = max(0, best - min_gap)
#             right = min(L-1, best + min_gap)
#             for t in range(left, right+1):
#                 used[t] = True

#         selected = sorted(selected)
#         # 兜底：至少要有最后一个 token 作为段末
#         if len(selected) == 0 or selected[-1] != L-1:
#             selected.append(L-1)
#         # 截断到 k 个
#         return selected[:k]

# # 演示代码
# if __name__ == "__main__":
#     # 创建实例
#     picker = PeakPicker()
    
#     # 1. 简单测试案例
#     print("=== 简单测试案例 ===")
#     entropy = torch.tensor([1.2, 3.5, 0.8, 4.2, 2.1, 5.0, 1.8, 3.0])
#     valid_len = 8
#     k = 2
#     min_gap = 2
    
#     print(f"熵值向量: {entropy.numpy()}")
#     print(f"有效长度: {valid_len}, 选取数量: {k}, 最小间隔: {min_gap}")
    
#     result = picker._pick_topk_peaks_with_gap(entropy, valid_len, k, min_gap)
#     print(f"选择的切点: {result}\n")
    
#     # 2. 包含重复高值的案例
#     print("=== 包含重复高值的案例 ===")
#     entropy2 = torch.tensor([5.0, 2.0, 5.0, 2.0, 5.0, 2.0, 5.0])
#     valid_len2 = 7
#     k = 3
#     min_gap2 = 1
    
#     print(f"熵值向量: {entropy2.numpy()}")
#     print(f"有效长度: {valid_len2}, 选取数量: {k}, 最小间隔: {min_gap2}")
    
#     result2 = picker._pick_topk_peaks_with_gap(entropy2, valid_len2, k, min_gap2)
#     print(f"选择的切点: {result2}\n")
    
#     # 3. 测试兜底机制（未选中最后一个元素时）
#     print("=== 兜底机制测试 ===")
#     entropy3 = torch.tensor([3.0, 2.0, 1.0, 0.5, 0.3])
#     valid_len3 = 5
#     k = 1
#     min_gap3 = 10  # 较大的间隔确保只能选第一个元素
    
#     print(f"熵值向量: {entropy3.numpy()}")
#     print(f"有效长度: {valid_len3}, 选取数量: {k}, 最小间隔: {min_gap3}")
    
#     result3 = picker._pick_topk_peaks_with_gap(entropy3, valid_len3, k, min_gap3)
#     print(f"选择的切点: {result3}")


import torch
from transformers import AutoTokenizer

# 模拟环境（你可以替换成自己的 tokenizer）
tokenizer = AutoTokenizer.from_pretrained("/data/work_backup/jingyiwang/models/Qwen2.5-Math-1.5B-Instruct")

# 示例：GT 和包含 boxed 的 full_text
gt = r"\frac{2}{3}"
full_text = r"Let us simplify the fraction. The result is \boxed{\frac{2}{3}}."

# Tokenize 整段文本
input_ids = tokenizer.encode(full_text, add_special_tokens=False, return_tensors="pt").squeeze(0)

# Tokenize GT 本身（不加 special token）
gt_ids = tokenizer.encode(gt, add_special_tokens=False)

# Tokenize boxed，查找 \boxed{...} 的 token 区间
boxed_start_str = r"\boxed{"
boxed_start = full_text.find(boxed_start_str)
boxed_end = full_text.find("}", boxed_start)

assert boxed_start != -1 and boxed_end != -1, "Can't find boxed region in text."

# 确定 boxed 范围内的字符 token 映射
offsets = tokenizer(full_text, return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"]

# 定位 token index 的 boxed_start_token 和 boxed_end_token（闭区间）
boxed_token_indices = [
    idx for idx, (start, end) in enumerate(offsets)
    if start >= boxed_start and end <= boxed_end + 1  # +1 to include final "}"
]
boxed_start_token = boxed_token_indices[0]
boxed_end_token = boxed_token_indices[-1]

print(f"Token index range of \\boxed{{...}}: {boxed_start_token} - {boxed_end_token}")

# 提取 boxed 对应 token ids
boxed_span = input_ids[boxed_start_token:boxed_end_token + 1]

# ========== 滑窗匹配 ==========
def find_subseq(haystack: torch.Tensor, needle: list[int]) -> tuple[int, int] | None:
    n = len(needle)
    for s in range(len(haystack) - n + 1):
        if torch.equal(haystack[s:s+n], torch.tensor(needle, device=haystack.device)):
            return s, s + n - 1
    return None

rel = find_subseq(boxed_span, gt_ids)
if rel is None:
    raise ValueError("GT token sequence not found inside \\boxed{}")

# 转换为在整段文本里的 token 索引
gt_start = boxed_start_token + rel[0]
gt_end = boxed_start_token + rel[1]

# 验证解码是否准确
gt_tokens = input_ids[gt_start:gt_end + 1]
gt_decoded = tokenizer.decode(gt_tokens, skip_special_tokens=False)

print(f"\n==== 匹配结果 ====")
print(f"GT token range: {gt_start}-{gt_end}")
print(f"Decoded GT: {gt_decoded}")
print("=================")