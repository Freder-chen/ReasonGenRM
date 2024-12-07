# GenRM

A simple implementation of GenRM.

## Results on RewardBench

This section presents the performance of different GenRM models on the RewardBench benchmark.

### LLaMA Results

|            Model             |     Method     | Score  |  Chat  | Chat Hard | Safety | Reasoning |
| :---------------------------:| :------------: | :----: | :----: | :-------: | :----: | :-------: |
|   LLaMA3.1-8B-Instruct       |   Text Match   | 71.30  | 91.83  |   51.97   | 78.92  |   62.48   |
|   **LLaMA3.1-8B-GenRM**      |   Text Match   | 88.76  | 93.30  |   81.58   | 91.55  |   88.60   |
| Skywork-Critic-Llama-3.1-8B  |   Text Match   | 89.00  | 93.02  |   81.14   | 91.89  |   89.97   |
| Skywork-Reward-Llama-3.1-8B  |  Bradley-Terry | 92.50  | 95.80  |   87.30   | 90.80  |   96.20   |

### Qwen Results

|            Model             |     Method     | Score  |  Chat  | Chat Hard | Safety | Reasoning |
| :---------------------------:| :------------: | :----: | :----: | :-------: | :----: | :-------: |
|   Qwen2.5-7B-Instruct        |   Text Match   | 79.78  | 96.65  |   60.09   | 81.69  |   80.67   |
|   **Qwen2.5-7B-GenRM**       |   Text Match   | 88.87  | 89.94  |   91.69   | 91.69  |   89.29   |
|   Qwen2.5-14B-Instruct       |   Text Match   | 82.28  | 95.39  |   67.32   | 84.73  |   81.66   |
|   **Qwen2.5-14B-GenRM**      |   Text Match   | 89.60  | 91.90  |   86.95   | 90.68  |   88.86   |

## Usage

The `examples` directory contains scripts for fine-tuning (SFT) and evaluation on RewardBench.

### SFT (Fine-Tuning)

The SFT scripts are located in the `examples/sft` directory. These scripts include data preparation and training scripts. Please modify the parameters in the scripts as needed before running them.

```bash
bash examples/sft/1_prepare.sh
bash examples/sft/2_openrlhf_sft.sh
```

### Reward Bench Evaluation

The evaluation scripts for RewardBench are in the `examples/reward_bench` directory. Again, please adjust the parameters in the script as required.

```bash
bash examples/eval/rewardbench.sh
```

## Notes

Historically, this project used TRL duplex results, but the accuracy was consistently inferior to OpenRLHF. Therefore, TRL has been discontinued in favor of the current implementation.
