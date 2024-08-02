from argparse import ArgumentParser
import os
def init_args():
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="openai-community/gpt2")
    parser.add_argument("--txt", type=str, required=True)
    parser.add_argument("--jsonl", type=str, default="./subgraph-gen/webnlg/train.jsonl")
    parser.add_argument("--gpu", type=str, help="Example: 0,1,2,3 (similar to CUDA_VISIBLE_DEVICES)")
    parser.add_argument("-b", "--bsize", type=int, default=16)
    parser.add_argument("--score", action="store_true")
    parser.add_argument("-o", "--out", type=str, default=".")
    parser.add_argument("--index", type=int)
    

    args = parser.parse_args()
    return args

args = init_args()
if args.gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import sys
sys.path.append("../src")
from model import AutoModelForLMKBC
from transformers import AutoTokenizer
import torch
from tqdm import tqdm
import pandas as pd
from itertools import combinations
import json

# sentence -> quoted string for universality
def baseline_template(text):
    return text

def eol_template(text):
    template = f'This quoted string : "{text}" means in one word:'
    return template

def pretended_cot_template(text):
    template = f'After thinking step by step , this quoted string : "{text}" means in one word:'
    return template

def knowledge_enhancement_template(text):
    template = f'The essence of a sentence is often captured by its main subjects and actions, while descriptive terms provide additional but less central details. With this in mind , this quoted string : "{text}" means in one word:'
    return template

if __name__ == "__main__":
    # DATA
    with open(args.txt, 'r', encoding="utf-8") as fp:
        text = fp.read().strip().splitlines()
    
    ## BASELINE
    baseline_text = [baseline_template(el) for el in text]
    ## EOL
    eol_text = [eol_template(el) for el in text]
    ## Pretended COT
    pretended_cot_text = [pretended_cot_template(el) for el in text]
    ## KE
    ke_text = [knowledge_enhancement_template(el) for el in text]

    # MODEL
    model = AutoModelForLMKBC.from_pretrained(args.model, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model.prepare_tokenizer(tokenizer)

    baseline_tensor = []
    eol_tensor = []
    pcot_tensor = []
    ke_tensor = []

    for i in tqdm(range(0, len(text), args.bsize)):
        ## BASELINE
        tokenized_baseline = tokenizer(baseline_text[i:i+args.bsize], padding=True, return_tensors="pt").to(model.device)
        ## EOL
        tokenized_eol = tokenizer(eol_text[i:i+args.bsize], padding=True, return_tensors="pt").to(model.device)
        ## PCOT
        tokenized_pcot = tokenizer(pretended_cot_text[i:i+args.bsize], padding=True, return_tensors="pt").to(model.device)
        ## KE
        tokenized_ke = tokenizer(ke_text[i:i+args.bsize], padding=True, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out_baseline = model.text_embedding(index=args.index, **tokenized_baseline) # N_LAYER * N_TEXT * N_DIM
            out_eol = model.text_embedding(index=args.index, **tokenized_eol)
            out_pcot = model.text_embedding(index=args.index, **tokenized_pcot)
            out_ke = model.text_embedding(index=args.index, **tokenized_ke)

            if args.index:
                if args.index < 0:
                    args.index = len(out_baseline) + args.index

        if not args.index:
            baseline_tensor.append(torch.stack(out_baseline).cpu())
            eol_tensor.append(torch.stack(out_eol).cpu())
            pcot_tensor.append(torch.stack(out_pcot).cpu())
            ke_tensor.append(torch.stack(out_ke).cpu())
        else:
            baseline_tensor.append(out_baseline.unsqueeze(0).cpu())
            eol_tensor.append(out_eol.unsqueeze(0).cpu())
            pcot_tensor.append(out_pcot.unsqueeze(0).cpu())
            ke_tensor.append(out_ke.unsqueeze(0).cpu())
    
    baseline_tensor = torch.cat(baseline_tensor, dim=1)
    eol_tensor = torch.cat(eol_tensor, dim=1)
    pcot_tensor = torch.cat(pcot_tensor, dim=1)
    ke_tensor = torch.cat(ke_tensor, dim=1)

    base_filename = f"{os.path.split(args.txt)[-1].replace('.txt','')}.{args.model.replace('/','_')}"

    filename = f"{base_filename}.tensor" if not args.index else f"{base_filename}.{args.index}.tensor"

    all_tensor = {
        "baseline" : baseline_tensor.squeeze(),
        "eol" : eol_tensor.squeeze(),
        "pcot" : pcot_tensor.squeeze(),
        "ke" : ke_tensor.squeeze()
    }
    torch.save(all_tensor, os.path.join(args.out, filename))

    # SCORE
    if args.score:
        sts_data = pd.read_json(args.jsonl, lines=True)
        cosine_sim = torch.nn.CosineSimilarity(dim=1)

        sts_result = {}
        for k, tensor in all_tensor.items():
            for i in range(len(tensor)):
                sts_result[f"{k}-{i}"] = []
                for _, row in tqdm(sts_data.iterrows(), desc=f"scoring {k}-{i}"):
                    pair = torch.tensor(list(combinations(row["text"],2)))
                    if len(pair) > 0:
                        input1 = tensor[i][pair[:,0]]
                        input2 = tensor[i][pair[:,1]]
                        sts_result[f"{k}-{i}"].append(
                            cosine_sim(input1, input2)
                        )
                
                sts_result[f"{k}-{i}"] = torch.cat(sts_result[f"{k}-{i}"]).mean().item()
        filename = f"{base_filename}.json" if not args.index else f"{base_filename}.{args.index}.json"
        with open(filename, 'w') as fp:
            json.dump(sts_result, fp)