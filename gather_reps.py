import transformers
import datasets
import torch
import argparse
import tqdm
import json
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--output_file", type=str, default="responses_and_reps.json")
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

    dataset = dataset.map(lambda x: tokenizer(f"[INST] <<SYS>>\n + {args.system_prompt} + \n<</SYS>>\n\n" + x['original_questions'] + " [/INST]", return_tensors="pt"), batched=False)

    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "original_questions"])

    print("loading model...")
    model = AutoModelForCausalLM.from_pretrained(args.model_name, load_in_8bit=True)

    model = model.eval()

    auth_responses_and_reps = {}
    lib_responses_and_reps = {}
    left_responses_and_reps = {}
    right_responses_and_reps = {}

    for i in tqdm.tqdm(range(len(dataset['auth_dataset']))):
        output = model.generate(torch.tensor(dataset['auth_dataset'][i]['input_ids'].to("cuda")), max_new_tokens=100, output_hidden_states=True, return_dict_in_generate=True)
        response = tokenizer.decode(output.sequences[0])
        hidden_states = output.hidden_states
        # write these to a file
        auth_responses_and_reps[dataset['auth_dataset'][i]['original_questions']] = [response, hidden_states]
    
    with open(args.output_file + "_auth", "w") as f:
        json.dump(auth_responses_and_reps, f)

    for i in tqdm.tqdm(range(len(dataset['lib_dataset']))):
        output = model.generate(torch.tensor(dataset['lib_dataset'][i]['input_ids'].to("cuda")), max_new_tokens=100, output_hidden_states=True, return_dict_in_generate=True)
        response = tokenizer.decode(output.sequences[0])
        hidden_states = output.hidden_states
        # write these to a file
        lib_responses_and_reps[dataset['lib_dataset'][i]['original_questions']] = [response, hidden_states]

    with open(args.output_file + "_lib", "w") as f:
        json.dump(lib_responses_and_reps, f)

    for i in tqdm.tqdm(range(len(dataset['left_dataset']))):
        output = model.generate(torch.tensor(dataset['left_dataset'][i]['input_ids'].to("cuda")), max_new_tokens=100, output_hidden_states=True, return_dict_in_generate=True)
        response = tokenizer.decode(output.sequences[0])
        hidden_states = output.hidden_states
        # write these to a file
        left_responses_and_reps[dataset['left_dataset'][i]['original_questions']] = [response, hidden_states]

    with open(args.output_file + "_left", "w") as f:
        json.dump(left_responses_and_reps, f)

    for i in tqdm.tqdm(range(len(dataset['right_dataset']))):
        output = model.generate(torch.tensor(dataset['right_dataset'][i]['input_ids'].to("cuda")), max_new_tokens=100, output_hidden_states=True, return_dict_in_generate=True)
        response = tokenizer.decode(output.sequences[0])
        hidden_states = output.hidden_states
        # write these to a file
        right_responses_and_reps[dataset['right_dataset'][i]['original_questions']] = [response, hidden_states]

    with open(args.output_file + "_right", "w") as f:
        json.dump(right_responses_and_reps, f)


if __name__ == "__main__":
    main()