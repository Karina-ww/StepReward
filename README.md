<div align="center">

# RLPR: Extrapolating RLVR To General Domain

</div>

<div align="center" style="font-size: 15pt">

<a href='https://arxiv.org'><img src='https://img.shields.io/badge/Paper-PDF-purple'></a>
<a href='https://huggingface.co/datasets'><img src='https://img.shields.io/badge/Dataset-HF-Green'></a>
<a href='https://huggingface.co'><img src='https://img.shields.io/badge/Model-7B-orange'></a>

<h4 align="center">
    <p>
        <a href="README_zh.md">中文</a> | <b>English</b>
    </p>
</h4>

</div>


## 🎊 News <!-- omit in toc -->

- [2025.06.23] We open-source the code, [weights](https://huggingface.co) and [data](https://huggingface.co/datasets) of RLPR!


## 📜 Brief Introduction <!-- omit in toc -->

We introduce the RLPR (Reinforcement Learning with Reference Probability Reward) framework that enhances the reasoning capabilities of Large Language Models (LLMs). RLPR uses LLM's generation probabilities as a reward signal and eliminates reliance of external verifiers. This approach enables robust, general-domain reasoning improvements with greater efficiency and broader applicability. Notable features of RLPR include:

* 💡 **Stronger Reasoning Enhancement**. 
	RLPR achieves better reasoning capability enchancement on both mathematical and general-domain reasoning benchmarks, even surpassing strong methods using verifier models.

<div align="center"> <img src="assets/performance_fig1.png" width = 80% /> </div>

* 🛠️ **Simple and Scalable Reward**.
    RLPR features an efficient Probability-based Reward (PR) using average decoding probabilities of reference answers. Without the need for laborious rule-based verifier construction, we simply calculate rewards with a single forward pass. 

<div align="center"> <img src="assets/framework.png" width = 80% /> </div>

* 🚀 **Better Reward Quality and Robust Training**.
    PR exhibits better reward quality compared with rule-based, model-based reward, and naive likelihood as a reward. We apply RLPR with different training prompt templates and find it achieves robustness reasoning capability enhancement.

<div align="center"> <img src="assets/PR_quality.png" width = 50% /> </div>

<div align="center"> <img src="assets/robustness.png" width = 80% /> </div>

## 📌Contents <!-- omit in toc -->

- [RLPR: Extrapolating RLVR To General Domain](#rlpr-extrapolating-rlvr-to-general-domain)
  - [Dataset](#dataset)
  - [Install](#install)
  - [Train](#train)
  - [Evaluation](#evaluation)
  - [Citation](#citation)

## Dataset

We present the [RLPR Train Dataset](https://huggingface.co/) and [evaluation benchmarks](https://huggingface.co/) for easier usage.

## Install

1. Clone this repository and navigate to RLPR folder
```bash
git clone 
cd RLPR
```

2. Install package
```bash
bash scripts/setup_env.sh
```

<!-- ## Model Weights 

| Model   | Description |           Download           |
| ------- | ----------- | :--------------------------: |
| RLPR 7B |      placeholder       | [🤗](https://huggingface.co) | -->


## Train

1. Prepare data

Download the [train](https://huggingface.co/datasets/openbmb/RLPR-Train-Dataset) and [test](https://huggingface.co/datasets/openbmb/RLPR-Evaluation) dataset. Move `rlpr_train.parquet` to `./datasets/train`, and move all the test datasets to `./datasets/test`.
```bash
huggingface-cli download --repo-type dataset --resume-download openbmb/RLPR-Train-Dataset --local-dir ./datasets/train
huggingface-cli download --repo-type dataset --resume-download openbmb/RLPR-Evaluation --local-dir ./datasets/test
```

2. Specify the base model path `examples/RLPR/reproduce.sh`.
```bash
MODEL=path_to_base_model
```

3. (Optional) Login wandb and set USE_WANDB to True in the `examples/RLPR/reproduce.sh` if you want to use wandb for logging.

```bash
USE_WANDB=${USE_WANDB:-"false"}
```

4. (Optional) Follow the following steps to use the `llm as a judge` eval method. Skip this step if you want to use a rule-based verifier to judge the answer.
	- Open-Source Model as judge
	    1. Create a new environment for the server and deploy the model. (Specify judge model, host and port in the `setup_server.sh`)
	        
	        ```shell
	        bash scripts/setup_server.sh
	        ```
	        
	    2. Specify the judge model in the `examples/RLPR/reproduce.sh`.
	        
	        ```shell
	        export CLIENT_IP=http://127.0.0.1:8001
            export USED_MODEL=Qwen/Qwen2.5-72B-Instruct
	        ```
	- API-Based Model (gpt-4o / 4pt-4.1) as judge 
		
        Specify token and the judge model in the `examples/RLPR/reproduce.sh` to use OpenAI API.
        
        ```shell
        export OPENAI_API_KEY=your_api_token
        export OPENAI_API_BASE=your_api_base  # default is https://api.openai.com/v1
        export USED_MODEL=gpt-4.1
        ```

5. Run the training script

```shell
bash examples/RLPR/reproduce.sh
```

## Evaluation

1. Follow the steps 1~4 in the [Train](#train) section to prepare the data, model and judge model (optional).

2. Run the evaluation script

```shell
bash examples/RLPR/reproduce.sh +trainer.val_only=True
```

## Licenses <!-- omit in toc -->


[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE)
[![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/DATA_LICENSE)

**Usage and License Notices**: The data, code, and checkpoint are intended and licensed for research use only. The dataset is CC BY NC 4.0 (allowing only non-commercial use) and models trained using the dataset should not be used outside of research purposes.



## Acknowledgement <!-- omit in toc -->

- [veRL](https://github.com/volcengine/verl): The codebase we built upon.


## Citation

If you find our model/code/data/paper helpful, please consider cite our papers 📝 and star us ⭐️！

```bibtex
@article{yu2025rlpr,
  title={RLPR: Extrapolating RLVR to General Domain without Verifiers},
  author={Yu, Tianyu and Ji, Bo and Wang, Shouli and Yao, Shu and Wang, Zefan and Cui, Ganqu and Yuan, Lifan and Ding, Ning and Yao, Yuan and Liu, Zhiyuan and Sun, Maosong and Chua, Tat-Seng},
  journal={arXiv preprint arXiv:2506.xxxxx},
  year={2025}
}
```