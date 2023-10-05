import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
from peft import PeftModel

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.gleu_score import sentence_gleu
from bert_score import BERTScorer
from utils import generate

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="output/stories_sft")
parser.add_argument("--temp", type=float, default=1.0)
parser.add_argument("--max_eval_len", type=int, default=12)
parser.add_argument("--batch_size", type=int, default=100)
parser.add_argument("--load_checkpoint_path", type=str, default=None)
parser.add_argument("--query_type", type=str, default="infill")
args = parser.parse_args()


scorer = BERTScorer(model_type="microsoft/deberta-xlarge-mnli", lang="en", rescale_with_baseline=True)


load_checkpoint_path = args.load_checkpoint_path

tokenizer = AutoTokenizer.from_pretrained(args.model_name)
model = AutoModelForCausalLM.from_pretrained(args.model_name,
                                            torch_dtype=torch.bfloat16,
                                            trust_remote_code=True,
                                            use_auth_token=True)
model.to('cuda')
if load_checkpoint_path is not None:
    model = PeftModel.from_pretrained(model, load_checkpoint_path)
model.eval()



df = pd.read_csv("data/stories.csv")
data = []

for index in range(0, 100):
    row = df.iloc[index]
    story = {}
    if args.query_type == "infill":
        story["str_query"] = "Beginning: " + row["sentence1"] + " " + row["sentence2"] + " " + row["sentence3"]
        story["str_query"] += "\nEnding: " + row["sentence5"] + "\nMiddle:"
    elif args.query_type == "beginning":
        story["str_query"] = row["sentence1"] + " " + row["sentence2"] + " " + row["sentence3"]
    story["str_sol"] = ". " + row["sentence5"]
    story["str_rationale"] = " " + row["sentence4"][:-1]
    story["beginning"] = row["sentence1"] + " " + row["sentence2"] + " " + row["sentence3"]

    data.append(story)

train_queries = []
train_rationales = []
train_sols = []
train_weight = []
train_beginning = []

test_queries = []
test_num_sols = []
test_sols = []
test_beginning = []

for i, sample in enumerate(data):
    train_queries.append(sample['str_query'])
    train_rationales.append(sample['str_rationale'])
    train_sols.append(sample['str_sol'])
    train_weight.append(1)
    train_beginning.append(sample['beginning'])

train_weight = np.array(train_weight)
train_weight = train_weight / train_weight.sum()

for sample in data:
    test_queries.append(sample['str_query'])
    if "num_sol" in sample:
        test_num_sols.append(sample['num_sol'])
    else:
        test_num_sols = None
    test_sols.append(sample['str_sol'])
    test_beginning.append(sample['beginning'])

encoded_train_queries = [tokenizer(query, return_tensors='pt')['input_ids'].cuda() for query in train_queries]
encoded_train_sols = [tokenizer(answer, return_tensors='pt')['input_ids'].cuda() for answer in train_sols]
encoded_test_queries = [tokenizer(query, return_tensors='pt')['input_ids'].cuda() for query in test_queries]
encoded_train_beginning = [tokenizer(query, return_tensors='pt')['input_ids'].cuda() for query in train_beginning]
encoded_test_beginning = [tokenizer(query, return_tensors='pt')['input_ids'].cuda() for query in test_beginning]

pad_token_id = tokenizer.eos_token_id
eos_token_id = tokenizer.eos_token_id

def compute_metrics(ref, cands):
    # BLEU
    ref_tokens = ref.split()
    cands_tokens = [cand.split() for cand in cands]
    bleu_scores = [sentence_bleu([ref_tokens], cand, smoothing_function=SmoothingFunction().method1) for cand in cands_tokens]

    # GLEU
    gleu_scores = [sentence_gleu([ref_tokens], cand) for cand in cands_tokens]

    # BERTScore
    P, R, F1 = scorer.score(cands, [ref]*len(cands), verbose=False)
    bertscore_f1 = F1.tolist()

    return bleu_scores, gleu_scores, bertscore_f1


def eval_model(model, tokenizer, temp=1.0, max_eval_len=12, batch_size=100):
    metrics = {
        "bleu": [],
        "gleu": [],
        "bertscore_f1": []
    }
    for idx in tqdm(range(100)):
        encoded_input = encoded_train_queries[idx]
        with torch.no_grad():
            generated_tokens, _, full_state = generate(model,
                                    encoded_input.repeat(batch_size, 1),
                                    eos_token_id=eos_token_id,
                                    vocab_nice_mask=None,
                                    max_len=max_eval_len,
                                    temperature=temp,
                                    tokenizer=tokenizer,
                                    use_tools=False,
                                    limit_capability=None,
                                    operators=None,
                                    use_cache=True)
        torch.cuda.empty_cache()
        
        # generated_tokens = append_sol_and_remove_eos(model, encoded_input, generated_tokens, encoded_train_sols[idx], tokenizer)
        generated_text = tokenizer.batch_decode(generated_tokens[:, encoded_input.shape[-1]:], skip_special_tokens=True)
        # print(generated_text)
        generated_text = [gt[1:] for gt in generated_text]
        bleu, gleu, bertscore_f1 = compute_metrics(train_rationales[idx], generated_text)
        metrics["bleu"].append(bleu)
        metrics["gleu"].append(gleu)
        metrics["bertscore_f1"].append(bertscore_f1)

    return metrics


eval_metrics = eval_model(model, tokenizer, temp=args.temp, max_eval_len=args.max_eval_len, batch_size=args.max_eval_len)

print("BLEU: ", np.mean(eval_metrics["bleu"]))
print("GLEU: ", np.mean(eval_metrics["gleu"]))
print("BERTScore F1: ", np.mean(eval_metrics["bertscore_f1"]))