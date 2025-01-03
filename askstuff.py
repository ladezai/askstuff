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
QUIT_RESERVED_KEYWORDS = ['!end', '!quit', '!exit']

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
        "Fourth step: under the previously asserted thoughts, provide a complete answer."
    ),
    "literate": (
        "You are an helpful assistant working for a prestigious peer-reviewed journal.\n"
        "Your language must be simple yet technical and formal.\n"
        "You'll receive a sentence and use the steps answer the query:\n"
        "First step: critique the usage of informal writing in the original sentence.\n"
        "Second step: critique the structure of the original sentence, if a better structure might\n"
        "\tresult enhance readability, conciseness and formalness.\n"
        "Third step: provide exactly three re-writing of the given paragraph that improve\n"
        "\tupon the aforementioned issues.\n\n"
        "Paragraph to improve: "
    ),
}


class SYSTEM_PROMPTS(Enum):
    NO_SYSTEM_PROMPT = "no-system-prompt"
    REASONER = "reasoner"
    CODER = "coder"
    LITERATE = "literate"


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
        **question, 
        streamer=streamer, 
        pad_token_id=tokenizer.pad_token_id,
        max_length=4096, 
        temperature=0.15, 
        top_k=50
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


def answer_question(question: str, system_prompt: Optional[str] = None):
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
    keep_running = True
    while keep_running:
        question = input("> ")

        for exit_keyword in QUIT_RESERVED_KEYWORDS:
            if question.startswith(exit_keyword):
                keep_running = False

        if keep_running:
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
        print(f"To exit write any of the following {QUIT_RESERVED_KEYWORDS}.")
        interactive(system_prompt)
    else:
        answer_question(question=args.question, system_prompt=system_prompt)
