#%%
import os
import pandas as pd

VAL_FILES=['datasets/test/AIME2024_Avg16.parquet','datasets/test/Minerva_Avg4.parquet','datasets/test/gpqa_diamond_Avg4.parquet','datasets/test/MMLUPro-1000_Avg2.parquet  datasets/test/TheoremQA_Avg2.parquet','datasets/test/Math-500_Avg2.parquet','datasets/test/WebInstruct-verified-val_Avg2.parquet']

save_dir = 'datasets/test_llama_and_gemma'
os.makedirs(save_dir, exist_ok=True)


save_path_list = []

for fn in VAL_FILES:
    df = pd.read_parquet(fn)

    def modify_prompt(row):
        row = row[1:] # remove system prompt
        row[0]['content'] = row[0]['content'] + "\nPlease reason step by step, and put your final answer within <answer> </answer>."
        if any(text in fn for text in ['MMLUPro', 'gpqa_diamond', 'WebInstruct-verified', 'SuperGPQA']):
            row[0]['content'] = row[0]['content'] + '\nPlease only provide the letter of the answer in the tags.'
        return row

    df['prompt'] = df['prompt'].apply(modify_prompt)
    save_path = os.path.join(save_dir, os.path.basename(fn))
    assert save_path != fn
    print(f"{save_path=}")
    save_path_list.append(save_path)
    df.to_parquet(save_path)

print(str(save_path_list).replace(' ',''))