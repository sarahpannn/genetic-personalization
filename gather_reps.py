import argparse
import json

import tqdm
import wandb
import torch
import datasets

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--output_file", type=str, default="responses_and_reps.json")
    parser.add_argument("--generation_temp", type=float, default=0.7)
    parser.add_argument("--num_generations", type=int, default=10)
    parser.add_argument("--generation_length", type=int, default=150)
    parser.add_argument("--system_prompt",
                        type=str,
                        default="You are answering a political value questionnaire. Answer as if you hold the political beliefs as specified. Always seek to be as representative and accurate as possible.")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    print("loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    print("loading dataset...")
    dataset = datasets.load_dataset("sarahpann/political-spectrum-questionnaire")

    dataset = dataset.map(lambda x: tokenizer(f"[INST] <<SYS>>\n {args.system_prompt} \n<</SYS>>\n\n" + x['original_questions'] + " [/INST]", return_tensors="pt"), batched=False)

    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "original_questions"])

    print("loading model...")
    model = AutoModelForCausalLM.from_pretrained(args.model_name, load_in_8bit=True)

    model = model.eval()

    wandb.init(project="political_answers")

    gen_config = GenerationConfig.from_pretrained(args.model_name, generation_config = gen_config)

    for i in tqdm.tqdm(range(len(dataset['auth_dataset']))):
        for j in range(args.num_generations):
            output = model.generate(torch.tensor(dataset['auth_dataset'][i]['input_ids'].to("cuda")), )
            response = tokenizer.decode(output[0])
            wandb.log({"type": "auth",
                    "input": dataset['auth_dataset'][i]['original_questions'],
                    "response": response,})
    

    for i in tqdm.tqdm(range(len(dataset['lib_dataset']))):
        for j in range(args.num_generations):
            output = model.generate(torch.tensor(dataset['lib_dataset'][i]['input_ids'].to("cuda")), generation_config = gen_config)
            response = tokenizer.decode(output[0])
            wandb.log({"type": "lib",
                        "input": dataset['lib_dataset'][i]['original_questions'],
                        "response": response,})


    for i in tqdm.tqdm(range(len(dataset['left_dataset']))):
        for j in range(args.num_generations):
            output = model.generate(torch.tensor(dataset['left_dataset'][i]['input_ids'].to("cuda")), generation_config = gen_config)
            response = tokenizer.decode(output[0])
            # write these to a file
            wandb.log({"type": "left",
                        "input": dataset['left_dataset'][i]['original_questions'],
                        "response": response,})


    for i in tqdm.tqdm(range(len(dataset['right_dataset']))):
        for j in range(args.num_generations):
            output = model.generate(torch.tensor(dataset['right_dataset'][i]['input_ids'].to("cuda")), generation_config = gen_config)
            response = tokenizer.decode(output[0])
            # write these to a file
            wandb.log({"type": "right",
                        "input": dataset['right_dataset'][i]['original_questions'],
                        "response": response,})


if __name__ == "__main__":
    main()