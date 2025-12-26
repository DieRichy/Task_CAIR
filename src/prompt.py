import textwrap
from jinja2 import Template

def apply_chat_template(model_name, system_msg, user_msg):
    """
    Based on the model name, apply the corresponding chat template.
    """
    model_name = model_name.lower()
    
    if 'qwen' in model_name:
        template_str = textwrap.dedent("""\
            <|im_start|>system
            {{ system_msg }}<|im_end|>
            <|im_start|>user
            {{ user_msg }}<|im_end|>
            <|im_start|>assistant\n 
            """) # note that "\n" is necessary for make sure Qwen generate answer directly.
    elif 'llama-3' in model_name:
        template_str = textwrap.dedent("""\
            <|start_header_id|>system<|end_header_id|>
            {{ system_msg }}<|eot_id|>
            <|start_header_id|>user<|end_header_id|>
            {{ user_msg }}<|eot_id|>
            <|start_header_id|>assistant<|end_header_id|>
            """)
    else:
        # default template
        return f"System: {system_msg}\nUser: {user_msg}\nAssistant:"

    return Template(template_str).render(system_msg=system_msg, user_msg=user_msg)