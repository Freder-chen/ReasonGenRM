# GenRM

A simple implementation of GenRM. 

## Results on RewardBench

This section summarizes the performance of different GenRM models on RewardBench. 

### LLaMA Results

|            Model             |     Method     | Score  |  Chat  | Chat Hard | Safety | Reasoning |
| :---------------------------:| :------------: | :----: | :----: | :-------: | :----: | :-------: |
|   LLaMA3.1-8B-Instruct       |   Text Match   | 71.52  | 91.62  |   51.59   | 78.21  |   64.67   |
|   **LLaMA3.1-8B-GenRM**      |   Text Match   | 87.28  | 93.85  |   79.82   | 88.18  |   87.28   |
| Skywork-Critic-Llama-3.1-8B  |   Text Match   | 88.93  | 92.46  |   81.25   | 91.82  |   90.18   |
| Skywork-Reward-Llama-3.1-8B  |  Bradley-Terry | 92.50  | 95.80  |   87.30   | 90.80  |   96.20   |

### Qwen Results

|            Model             |     Method     | Score  |  Chat  | Chat Hard | Safety | Reasoning |
| :---------------------------:| :------------: | :----: | :----: | :-------: | :----: | :-------: |
|   Qwen2.5-7B-Instruct        |   Text Match   | 79.88  | 96.79  |   60.20   | 81.82  |   80.71   |
|   **Qwen2.5-7B-GenRM**       |   Text Match   | 84.79  | 92.04  |   76.54   | 84.93  |   85.66   |
|   **Qwen2.5-14B-GenRM**      |   Text Match   | 84.79  | 92.04  |   76.54   | 84.93  |   85.66   |
