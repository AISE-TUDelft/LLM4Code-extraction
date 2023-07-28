import torch

import tqdm

import pandas as pd
import ast
import sys

from transformers import AutoTokenizer, AutoModelForCausalLM
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
smoothie = SmoothingFunction().method4
tqdm.tqdm.pandas()

pretrained_model_name = sys.argv[1]

print(f'training for model: {pretrained_model_name}')

import feather
df = feather.read_dataframe('text_benchmark.feather')


def find_batch_size():
    if pretrained_model_name == 'codeparrot/codeparrot': # The parrot is really hungry 🦜
        return 10 
    if n_params < 0.35:
        return 160
    elif n_params < 1.0:
        return 128
    elif n_params < 3.0:
        return 18
    else:
        return 10

base_tokenizer = AutoTokenizer.from_pretrained(
        'EleutherAI/gpt-neo-1.3B',
        padding_side='left',
        add_special_tokens=True
    )

tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name,
        padding_side='left',
        add_special_tokens=True,
    )


model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name,
        vocab_size = tokenizer.vocab_size,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
	trust_remote_code=True
    ).to(device='cuda:0', non_blocking=True)
n_params = sum([p.numel() for p in model.parameters()]) / 10**9  ## billion
print(f"Weights loaded: {pretrained_model_name} (# params: {n_params:.2f}B)") 

def generate(tokenizer, model, prompt):
    with torch.no_grad():
        ## prompt to tensor
        tokens = torch.tensor(prompt).to(device='cuda:0')
        ## Generate texts from tokens.
        gen_tokens = model.generate(tokens, max_length=100, pad_token_id=tokenizer.eos_token_id)
        generated = [g.tolist() for g in gen_tokens]     
    return generated

print(hasattr(tokenizer, "vocab"))

if hasattr(tokenizer, "vocab") and base_tokenizer.vocab != tokenizer.vocab:
    print('Different tokenizers: Re-encoding samples...')
    df['sample'] = df['sample'].progress_apply(lambda x: tokenizer.encode(base_tokenizer.decode(x)))
    df = df[df['sample'].apply(len) >= 100].reset_index(drop=True) # drop samples that are too short 
    df['prefix'] = df['sample'].progress_apply(lambda x: x[:50])
    df['suffix'] = df['sample'].progress_apply(lambda x: x[50:100])
else: # same tokenizer
    print('Same tokenizers: No need to re-encode samples...')
    df['prefix'] = df['sample'].progress_apply(lambda x: x[:50])
    df['suffix'] = df['sample'].progress_apply(lambda x: x[50:100])

texts = []

# iterate with batch size
batch_size = find_batch_size() # 18 for 2B. 128 for 350M. 160 for 125M.
with tqdm.tqdm(total=len(df)) as pbar:
    for i in range(0, len(df), batch_size):
        batch = list(df.iloc[i:i+batch_size].prefix)
        generated = generate(tokenizer, model, prompt=batch)
        texts.extend(generated)
        pbar.update(batch_size)
 
# calculate BLEU-4 score
def calc_bleu4(tokenizer, sample, generated):
    ref = tokenizer.decode(sample)
    hyp = tokenizer.decode(generated)
    return sentence_bleu([ref], hyp, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)

df['gen_tokens'] = texts 
df['gen_suffix'] = df['gen_tokens'].progress_apply(lambda x: x[50:100])
df['bleu4'] = df.progress_apply(lambda x: calc_bleu4(tokenizer, x['suffix'], x['gen_suffix']), axis=1)
df['em'] = df['bleu4'] == 1

print(pretrained_model_name)
# save results
name = pretrained_model_name.replace('/', '_')
df.to_pickle(f'results/{name}text_benchmark.pkl')


print(f'num_params: {round(n_params, 3) * 1000}')
# count number of exact match rounded to 4 decimals
print(f"em: {df['em'].mean()}")
# average bleu4 score
print(f"bleu4: {df['bleu4'].mean()}")
