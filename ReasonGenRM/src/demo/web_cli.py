
import os
import torch
import argparse
import gradio as gr

from threading import Thread
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer


def load_model_and_tokenizer(model_name_or_path: str):
    """
    Load the tokenizer and model from a given path or HuggingFace repository.
    """
    print(f"Loading model from: {model_name_or_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        use_fast=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
        device_map="auto",
    ).eval()

    return tokenizer, model


def stream_generate_response(
    messages,
    tokenizer,
    model,
    add_reason_prompt=False,
    add_generation_prompt=False,
    max_new_tokens=1024,
    temperature=0.5,
):
    """
    Stream text generation in a token-by-token manner.
    """
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_reason_prompt=add_reason_prompt,
        add_generation_prompt=add_generation_prompt
    )
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(
        input_ids=inputs["input_ids"],
        max_new_tokens=max_new_tokens,
        early_stopping=True,
        temperature=temperature,
        streamer=streamer,
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    partial_text = ""
    for new_token in streamer:
        partial_text += new_token
        yield partial_text


def stream_reason_and_answer(user_input, tokenizer, model, max_new_tokens=1024, temperature=0.5):
    """
    Two-step reasoning: 
    1) Generate 'reason' content.
    2) Generate final 'assistant' content.
    Each step is streamed to the frontend.
    """
    messages = [{"role": "user", "content": user_input}]

    # 1) Reason phase
    reason_text = ""
    for partial_reason_text in stream_generate_response(
        messages,
        tokenizer,
        model,
        add_reason_prompt=True,
        max_new_tokens=max_new_tokens,
        temperature=temperature
    ):
        reason_text = partial_reason_text
        yield reason_text, ""

    # 2) Assistant phase
    messages.append({"role": "reason", "content": reason_text})

    assistant_text = ""
    for partial_assistant_text in stream_generate_response(
        messages,
        tokenizer,
        model,
        add_generation_prompt=True,
        max_new_tokens=max_new_tokens,
        temperature=temperature
    ):
        assistant_text = partial_assistant_text
        yield reason_text, assistant_text


def build_demo(tokenizer, model):
    """
    Build a minimal Gradio interface for two-step reasoning (Reason + Assistant).
    """
    custom_css = """
    #banner {
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .gr-textbox textarea {
        font-family: monospace;
    }
    """

    def inference_fn(user_input):
        for reason_text, assistant_text in stream_reason_and_answer(user_input, tokenizer, model):
            yield reason_text, assistant_text

    with gr.Blocks(css=custom_css, title="Two-Step Reasoning", theme=gr.themes.Default()) as demo:
        gr.Markdown("<div id='banner'>Two-Step Reasoning (Streaming)</div>")

        with gr.Row():
            with gr.Column():
                user_input = gr.Textbox(
                    label="User Input",
                    lines=6,
                    placeholder="Type your query here..."
                )
                generate_button = gr.Button("Run")

            with gr.Column():
                reason_output = gr.Textbox(
                    label="Reason Output",
                    lines=6
                )
                assistant_output = gr.Textbox(
                    label="Assistant Output",
                    lines=6
                )

        generate_button.click(
            fn=inference_fn,
            inputs=user_input,
            outputs=[reason_output, assistant_output]
        )

    return demo


def main():
    """
    Parse command-line arguments and launch the Gradio application.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="jiulaikankan/Qwen2.5-14B-ReasonGenRM",
        help="Path or HuggingFace repo name of the model"
    )
    parser.add_argument(
        "--cuda_visible_devices",
        type=str,
        default="0,1,2,3",
        help="Set visible CUDA devices (e.g. '0' or '0,1,2,3')"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port for Gradio server"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Whether to enable Gradio share link"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024 * 8,
        help="Default max generation tokens for the Reason & Assistant phase"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="Temperature for generation"
    )
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    tokenizer, model = load_model_and_tokenizer(
        model_name_or_path=args.model_path
    )

    demo = build_demo(tokenizer, model)

    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share
    )


if __name__ == "__main__":
    main()
