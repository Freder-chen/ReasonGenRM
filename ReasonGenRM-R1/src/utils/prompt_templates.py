# # Copied from https://huggingface.co/Skywork/Skywork-Critic-Llama-3.1-70B
# CRITIC_PROMPT_TEMPLATE = """
# Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better. 
# Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. 
# Please directly output your final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better.

# [User Question]
# {question}    

# [The Start of Assistant A's Answer]
# {response_a}
# [The End of Assistant A's Answer]

# [The Start of Assistant B's Answer]
# {response_b}
# [The End of Assistant B's Answer]
# """.strip()

# Modified form https://huggingface.co/Skywork/Skywork-Critic-Llama-3.1-70B
CRITIC_PROMPT_TEMPLATE = """
Act as an impartial judge to evaluate the quality of responses from two AI assistants to the user question below. Choose the assistant that better follows the user's instructions and answers the question effectively.
Consider the following factors: helpfulness, relevance, accuracy, depth, creativity, harmlessness, and overall quality. Analyze these dimensions based on the specific problem, as different tasks may emphasize different criteria.
Avoid biases related to the position of responses, response length, or assistant names. Be objective in your assessment.
Output your final verdict strictly in this format: “[[A]]” if assistant A is better, “[[B]]” if assistant B is better.

[User Question]
{question}    

[The Start of Assistant A's Answer]
{response_a}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{response_b}
[The End of Assistant B's Answer]
""".strip()

REWARD_PROMPT_TEMPLATE = """
Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user's question below. Your task is to assess whether the assistant's response is accurate, complete, safe, and meets the requirements of the user's question.
Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. Be as objective as possible.
Use a rating scale of 0 to 9, where 0 is the lowest and 9 is the highest. Please directly output your final verdict as a single integer between 0 and 9, without any additional text, symbols, or formatting.

[User Question]
{question}

[The Start of Assistant's Answer]
{response}
[The End of Assistant's Answer]
""".strip()


# Copy from https://huggingface.co/datasets/KingNish/reasoning-base-20k/discussions/2
REASON_PROMPT_TEMPLATE = '''
You are a Large Reasoning model with advanced reasoning capabilities. Your primary goal is to provide accurate and well-reasoned answers to complex prompts by generating an internal "human-like thinking process" – a detailed step-by-step breakdown of your reasoning.

**Guidelines for Generating Human-Like Thinking/Reasoning:**

1. **Problem Decomposition:** 
    * Analyze the prompt carefully and decompose it into smaller, manageable sub-problems.
    * Explicitly state each sub-problem you identify.

2. **Knowledge Retrieval and Integration:**
    * Access and integrate relevant information from your vast knowledge base.
    * Clearly identify the knowledge sources you are using (e.g., facts, concepts, procedures).
    * Demonstrate how you are adapting the retrieved knowledge to the specific context of the prompt.

3. **Chain of Thought Elaboration:**
    * Generate a detailed sequence of logical deductions, intermediate conclusions, and potential solution paths.
    * Articulate your reasoning for each step in a clear and concise manner.
    * Explore different approaches, evaluate their feasibility, and justify your choices.
    * Use a variety of reasoning strategies, including:
        * **Deductive Reasoning:** Drawing logical conclusions from established facts and principles.
        * **Inductive Reasoning:** Forming generalizations based on specific observations or examples.
        * **Abductive Reasoning:** Inferring the most likely explanation for a set of observations.
        * **Reasoning by Analogy:** Applying previously learned problem-solving strategies to similar situations.

4. **Error Detection and Correction:**
    * Actively look for potential errors or inconsistencies in your reasoning.
    * Explicitly state any assumptions you are making.
    * Demonstrate how you identify and correct errors in your chain of thought.

5. **Solution Synthesis and Justification:**
    * Based on your chain of thought, select the most promising solution path.
    * Clearly articulate your final answer or solution.
    * Provide a comprehensive justification for your chosen solution, referencing specific steps in your chain of thought.

6. **Metacognitive Awareness:**
    * Demonstrate awareness of your own reasoning process. 
    * Reflect on the effectiveness of your chosen strategies.
    * Identify potential limitations or biases in your reasoning.

7. **Clarity and Conciseness:**
    * Present your "thinking process" in a clear, concise, and well-organized manner.
    * Use appropriate formatting and visual aids (if applicable) to enhance readability.

I'll give you both Users query and Correct best answer and your task is to generate very big and detailed reasoning for this. Length of Reasoning varies according to difficulty of question. But reason everything from basic to fully advanced. Reason in the manner that you don't know answer. Just like example
Reason in best way and reply with very detailed example like reasoning only.
Reason like human is doing it. Make reasoning like what human is thinking while solving that problem use words like humans and think very  long.
Don't use words like lets Think about problem Step by Step. Just starts reasoning. And Do reasoning very hard.

[User Query]
{question}

[The Start of Correct Best Answer]
{response}
[The End of Correct Best Answer]
'''.strip()
