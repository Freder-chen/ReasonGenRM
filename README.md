# ReasonGenRM with Qwen2.5

## Overview

This repository introduces ReasonGenRM, a reward model built on the Qwen2.5 architecture, designed to enhance the accuracy of reward evaluations by integrating structured reasoning. While promising, this model is still under development and not yet fully optimized.

## Requirements

Install dependencies:

```bash
pip install datasets==2.16.1 transformers==4.45.2 trl==0.11.4 vllm==0.6.2
```

**Note**: Depending on hardware and library versions, additional configuration may be necessary. Compatibility checks are recommended.

## Data Preparation

1. Download the dataset: [Skywork/Skywork-Reward-Preference-80K-v0.2](https://huggingface.co/datasets/Skywork/Skywork-Reward-Preference-80K-v0.2).

2. Prepare the data by running:

```bash
bash 1_prepare.sh
```

Data format (example):

```json
{
  "prompt": "prompt",
  "reason": "reason",
  "response": "response"
}
```

## Model Training

Run the following command to train the model:

```bash
bash 2_train.sh
```

## Model Evaluation

To evaluate the trained model, use:

```bash
bash 3_eval_reward_bench.sh
```

## Results on RewardBench

Results of ReasonGenRM models on RewardBench are as follows:

|              Model               |   Score   |   Chat    | Chat Hard |  Safety   | Reasoning |
| :------------------------------: | :-------: | :-------: | :-------: | :-------: | :-------: |
|       Qwen2.5-7B-Instruct        |   80.83   |   97.07   |   60.42   |   85.03   |   80.80   |
|      Qwen2.5-7B-ReasonGenRM      |   83.04   |   94.27   |   66.67   |   83.01   |   88.22   |


**NOTE:**

- **Instruct** refers to the baseline Qwen2.5 model without additional training.
- **GenRM** uses standard SFT with user-assistant dialogues, while **ReasonGenRM** incorporates user-reason-assistant data for enhanced reasoning.
- **wGen** denotes models trained with additional general-purpose datasets.

## TODO

- [ ] Add RewardBench results for Qwen2.5-7B-GenRM fine-tuned on **Skywork-Reward-Preference-80K-v0.2**.
