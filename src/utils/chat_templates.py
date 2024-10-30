QWEN_RASON_CHAT_TEMPLATE = '''\
{%- for message in messages %}
    {{- '<|im_start|>' + message['role'] + '\n' }}
    {{- message['content'] + eos_token + '\n' }}
{%- endfor %}
{%- if add_reason_prompt %}
    {{- '<|im_start|>reason\n' }}
{%- endif %}
{%- if add_generation_prompt %}
   {{- '<|im_start|>assistant\n' }}
{%- endif %}
'''

# TODO: add LLAMA reason template
LLAMA_RASON_CHAT_TEMPLATE = '''\
'''