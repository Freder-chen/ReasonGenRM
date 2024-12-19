QWEN_REASON_CHAT_TEMPLATE = '''\
{%- for message in messages %}
    {{- '<|im_start|>' + message['role'] + '\n' }}
    {{- message['content'] + eos_token + '\n' }}
{%- endfor %}
{%- if add_reason_prompt %}
    {{- '<|im_start|>reason\n' }}
{%- endif %}
{%- if add_generation_prompt %}
   {{- '<|im_start|>assistant\n' }}
{%- endif %}\
'''


LLAMA_REASON_CHAT_TEMPLATE = '''\
{{- bos_token }}
{%- for message in messages %}
    {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' }}
    {{- message['content'] + eos_token }}
{%- endfor %}
{%- if add_reason_prompt %}
    {{- '<|start_header_id|>reason<|end_header_id|>\n\n' }}
{%- endif %}
{%- if add_generation_prompt %}
   {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
{%- endif %}\
'''


HISTORY_DIALOG_CHAT_TEMPLATE = '''\
{%- if messages|length == 1 %}
    {{- messages[0]['content']  + '\n' }}
{%- else %}
{%- for message in messages -%}
    {{- '<turn>' + message['role'] + '\n' }}
    {{- message['content'] + '\n' }}
{%- endfor %}
{%- endif %}\
'''