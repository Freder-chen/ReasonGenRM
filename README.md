## Reasoning GenRM with Qwen2.5

### Overview

This repository hosts the Reasoning GenRM, an advanced reward model developed on the Qwen2.5 architecture. This model enhances the precision of reward-based assessments by integrating structured reasoning, with notable benefits in complex reasoning and decision-making tasks.

### Requirements

Ensure the following environment setup:

```bash
pip install datasets==2.16.1 transformers==4.45.2 trl==0.11.4 vllm==0.6.2
```

**Note**: Certain configurations may need adjustment to align with local hardware or library specifics. Compatibility checks are recommended.

### Data Preparation

Prepare the dataset for training by downloading [Skywork/Skywork-Reward-Preference-80K-v0.2](https://huggingface.co/datasets/Skywork/Skywork-Reward-Preference-80K-v0.2) and executing:

```bash
bash 1_prepare.sh
```

This will format the data as follows:

```json
{
  "prompt": "prompt",
  "reason": "reason",
  "response": "response"
}
```

### Model Training

Initiate model training by running:

```bash
bash 2_train.sh
```

### Model Evaluation

To evaluate the trained model, use:

```bash
bash 3_eval_reward_bench.sh
```

### Results on RewardBench

Below are the benchmark results for the Reasoning GenRM model on RewardBench:

| Model                        | Score | Chat  | Chat Hard | Safety | Reasoning |
|------------------------------|-------|-------|-----------|--------|-----------|
| Qwen2.5-7B-Instruct          | 80.83 | 97.07 | 60.42     | 85.03  | 80.80     |
| Qwen2.5-7B-ReasonGenRM       | 79.98 | 90.64 | 65.68     | 80.12  | 83.50     |
| Qwen2.5-7B-ReasonGenRM-wGen  | 80.00 | 91.76 | 66.94     | 81.28  | 80.01     |
| Qwen2.5-14B-ReasonGenRM      | 86.40 | 94.76 | 75.11     | 87.76  | 87.98     |
| Qwen2.5-14B-ReasonGenRM-wGen | 86.17 | 93.58 | 76.54     | 88.20  | 86.36     |

### TODO

[ ] Added RewardBench results for Qwen2.5-7B-GenRM fine-tuned with **Skywork-Reward-Preference-80K-v0.2**.
