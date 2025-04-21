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
