from argparse import ArgumentParser
from enum import Enum
from typing import Optional

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TextStreamer
import torch

# Text format
RICH_TEXT = False
CMD_LINE_LENGTH = 90
PLAIN_TEXT_PROMPT = f"\nThe output provided must be in plain-text format, easy to read in a terminal with monospace character with line size {CMD_LINE_LENGTH}."
DEVICE = "mps" if torch.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

# Default system prompts
PRIOR_SYSTEM_PROMPT = {
    "no-system-prompt": "",
    "reasoner": (
        "You are an helpful assistant. Think step-by-step following the following four steps:\n"
        "First step: describe the concepts given in the question.\n"
        "Second step: provide further insight on the previous information.\n"
        "Third step: Critic your answer by adding 'Wait, I'm wrong.' to your answer.\n"
        "Fourth step: Provide a complete answer based on your previous steps.\n"
    ),
    "coder": (
        "You are an helpful assistant. Think step-by-step. Follow the following steps: \n"
        "First step: describe the problem. Provide more context to the problem you have to solve.\n"
        "Second step: split the problem in sub-problems. Simplify each problem separately. \n"
        "Third step: criticise each sub-problem. Find possible issues or bugs.\n"
        "Fourth step: given all the reasonings, provide a complete answer."
    ),
}


class SYSTEM_PROMPTS(Enum):
    NO_SYSTEM_PROMPT = ("no-system-prompt",)
    REASONER = ("reasoner",)
    CODER = "coder"


def format_text(text):
    """Auto markdown italic/bold renderer for text-input with ANSI codes
    TODO: This is dead code, this functionality is not added to the main code
    """

    formatted_text = ""
    i = 0
    while i < len(text):
        if text[i] == "**":
            # Check for bold
            start_bold = i
            while i < len(text) and text[i] != "**":
                i += 1
            end_bold = i
            formatted_text += f"\033[1m{text[start_bold+2:end_bold-2]}\033[0m"
            i += 2
        elif text[i] == "*":
            # Check for italic
            start_italic = i
            while i < len(text) and text[i] != "*":
                i += 1
            end_italic = i
            formatted_text += f"\033[3m{text[start_italic+1:end_italic-1]}\033[0m"
            i += 1
        else:
            formatted_text += text[i]
            i += 1
    return formatted_text


def load_model(size: str):
    if size not in {"0.5", "1.5", "3", "7"}:
        raise ValueError(f"A model of size={size} is not available")

    tokenizer = AutoTokenizer.from_pretrained(f"Qwen/Qwen2.5-Coder-{size}B-Instruct")
    kwargs = dict(device_map=DEVICE)
    model = AutoModelForCausalLM.from_pretrained(f"Qwen/Qwen2.5-Coder-{size}B-Instruct", **kwargs)
    model.eval()

    return model, tokenizer


def stream_answer(question, model, tokenizer):
    streamer = TextStreamer(tokenizer, skip_prompt=True)
    _ = model.generate(
        **question, streamer=streamer, pad_token_id=tokenizer.pad_token_id, max_length=4096, temperature=0.1, top_k=45
    )
    return streamer


def generate_question(msg: str, system_prompt: str, tokenizer):
    chat = [
        {"role": "user", "content": msg},
    ]
    if system_prompt:
        chat.append({"role": "system", "content": system_prompt})
    question = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    question = tokenizer(question, return_tensors="pt").to(DEVICE)
    return question


def answer_question(question: str, system_prompt: Optional[str] = str):
    if system_prompt is None:
        system_prompt = input("Provide a system prompt (No answer does not use a system prompt): ")
        if system_prompt.lower().strip() in PRIOR_SYSTEM_PROMPT:
            system_prompt = PRIOR_SYSTEM_PROMPT[system_prompt]

        if not RICH_TEXT:
            system_prompt = system_prompt + PLAIN_TEXT_PROMPT
        print("The system prompt is: ")
        print(system_prompt)

    question = generate_question(question, system_prompt, tokenizer)
    _ = stream_answer(question, model, tokenizer)


def interactive(system_prompt: Optional[str] = None):
    while 1:
        question = input("> ")
        if question.startswith("!end") or question.startswith("!stop") or question.startswith("!exit"):
            break
        answer_question(question, system_prompt)


if __name__ == "__main__":
    parser = ArgumentParser(prog="Ask stuff", description="Ask a local Language Model with additional system prompts")
    parser.add_argument("-q", "--question", default="", type=str)
    parser.add_argument("-i", "--interactive", action="store_true")
    parser.add_argument("-p", "--prior", default="no-system-prompt", type=str)
    parser.add_argument("-s", "--size", default="1.5", type=str)

    args = parser.parse_args()
    args.prior = PRIOR_SYSTEM_PROMPT.get(args.prior.lower().strip(), args.prior)

    print("#" * CMD_LINE_LENGTH)
    print("Loading the model".lower().center(CMD_LINE_LENGTH))
    print("#" * CMD_LINE_LENGTH)
    model, tokenizer = load_model(args.size)
    system_prompt = args.prior
    if args.interactive:
        print("#" * CMD_LINE_LENGTH)
        print("INTERACTIVE LLM QWEN2.5-CODER-INSTRUCT".lower().center(CMD_LINE_LENGTH))
        print("#" * CMD_LINE_LENGTH)
        interactive(system_prompt)
    else:
        answer_question(question=args.question, system_prompt=system_prompt)
