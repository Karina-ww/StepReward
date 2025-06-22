set -x

# export FILE_PATH=assets/orz-webinst/ORZ_clip027_withstdfilter_nohuman.xlsx
# export CUDA_VISIBLE_DEVICES=1,2,3,4

# python tools/compute_token_probability.py --file_path ${FILE_PATH} --checkpoint /user/jibo/models/Qwen2.5-7B --task pr_analysis # 7B

# python tools/compute_token_probability.py --file_path ${FILE_PATH} --checkpoint /user/jibo/models/Qwen2.5-7B-Instruct --task pr_analysis # 7B-Instruct

# python tools/compute_token_probability.py --task pr_analysis_gr_verifier --file_path ${FILE_PATH} # General-Verifier


# python tools/compute_token_probability.py --file_path ${FILE_PATH} --checkpoint /user/jibo/models/Qwen2.5-1.5B --task pr_analysis # 1.5B

# python tools/compute_token_probability.py --file_path ${FILE_PATH} --checkpoint /user/jibo/models/Qwen2.5-3B --task pr_analysis # 3B

# python tools/compute_token_probability.py --file_path ${FILE_PATH} --checkpoint /user/jibo/models/Qwen2.5-0.5B --task pr_analysis # 0.5B

# python tools/compute_token_probability.py --file_path ${FILE_PATH} --checkpoint /user/jibo/models/Qwen2.5-72B --task pr_analysis # 72B

# python tools/compute_token_probability.py --file_path ${FILE_PATH} --checkpoint /user/jibo/models/Qwen2.5-32B --task pr_analysis # 32B

# python tools/compute_token_probability.py --file_path ${FILE_PATH} --checkpoint /user/jibo/models/Qwen2.5-14B --task pr_analysis # 14B

python tools/compute_token_probability.py --file_path ${FILE_PATH} --checkpoint /user/jibo/checkpoints/0531_Qwen2.5Base_PR_clip0.27_StdFilterEMA6-11-0.99-0.5_WebInst034N2_b256_mbs64_lr1e-6_r3_n8_e100-2node/global_step_40 --task pr_analysis --save_dir ${SAVE_DIR}
python tools/compute_token_probability.py --file_path ${FILE_PATH} --checkpoint /user/jibo/checkpoints/0531_Qwen2.5Base_PR_clip0.27_StdFilterEMA6-11-0.99-0.5_WebInst034N2_b256_mbs64_lr1e-6_r3_n8_e100-2node/global_step_80 --task pr_analysis --save_dir ${SAVE_DIR}
python tools/compute_token_probability.py --file_path ${FILE_PATH} --checkpoint /user/jibo/checkpoints/0531_Qwen2.5Base_PR_clip0.27_StdFilterEMA6-11-0.99-0.5_WebInst034N2_b256_mbs64_lr1e-6_r3_n8_e100-2node/global_step_120 --task pr_analysis --save_dir ${SAVE_DIR}
python tools/compute_token_probability.py --file_path ${FILE_PATH} --checkpoint /user/jibo/checkpoints/0531_Qwen2.5Base_PR_clip0.27_StdFilterEMA6-11-0.99-0.5_WebInst034N2_b256_mbs64_lr1e-6_r3_n8_e100-2node/global_step_160 --task pr_analysis --save_dir ${SAVE_DIR}
python tools/compute_token_probability.py --file_path ${FILE_PATH} --checkpoint /user/jibo/checkpoints/0531_Qwen2.5Base_PR_clip0.27_StdFilterEMA6-11-0.99-0.5_WebInst034N2_b256_mbs64_lr1e-6_r3_n8_e100-2node/global_step_200 --task pr_analysis --save_dir ${SAVE_DIR}
python tools/compute_token_probability.py --file_path ${FILE_PATH} --checkpoint /user/jibo/checkpoints/0531_Qwen2.5Base_PR_clip0.27_StdFilterEMA6-11-0.99-0.5_WebInst034N2_b256_mbs64_lr1e-6_r3_n8_e100-2node/global_step_240 --task pr_analysis --save_dir ${SAVE_DIR}
python tools/compute_token_probability.py --file_path ${FILE_PATH} --checkpoint /user/jibo/checkpoints/0531_Qwen2.5Base_PR_clip0.27_StdFilterEMA6-11-0.99-0.5_WebInst034N2_b256_mbs64_lr1e-6_r3_n8_e100-2node/global_step_280 --task pr_analysis --save_dir ${SAVE_DIR}
python tools/compute_token_probability.py --file_path ${FILE_PATH} --checkpoint /user/jibo/checkpoints/0531_Qwen2.5Base_PR_clip0.27_StdFilterEMA6-11-0.99-0.5_WebInst034N2_b256_mbs64_lr1e-6_r3_n8_e100-2node/global_step_320 --task pr_analysis --save_dir ${SAVE_DIR}
python tools/compute_token_probability.py --file_path ${FILE_PATH} --checkpoint /user/jibo/checkpoints/0531_Qwen2.5Base_PR_clip0.27_StdFilterEMA6-11-0.99-0.5_WebInst034N2_b256_mbs64_lr1e-6_r3_n8_e100-2node/global_step_360 --task pr_analysis --save_dir ${SAVE_DIR}
python tools/compute_token_probability.py --file_path ${FILE_PATH} --checkpoint /user/jibo/checkpoints/0531_Qwen2.5Base_PR_clip0.27_StdFilterEMA6-11-0.99-0.5_WebInst034N2_b256_mbs64_lr1e-6_r3_n8_e100-2node/global_step_400 --task pr_analysis --save_dir ${SAVE_DIR}