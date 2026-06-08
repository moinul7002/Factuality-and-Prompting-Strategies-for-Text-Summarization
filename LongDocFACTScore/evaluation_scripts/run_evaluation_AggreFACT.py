import pandas as pd
import numpy as np
from utils.correlations import (
    fleiss_kappa,
    krippendorff_alpha,
    interval_metric,
    kendal_tau_matrix,
)
from utils.metrics import test_huggingface_rouge, test_questeval
import argparse
from longdocfactscore.ldfacts import LongDocFACTScore
from utils.preprocess import clean_abstract
from BARTScore.bart_score import BARTScorer
from bert_score.scorer import BERTScorer
import json
from nltk import sent_tokenize
from nltk.tokenize import word_tokenize
import subprocess
import time
import os
import seaborn as sb
import torch

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from sacrebleu import sentence_bleu
from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

ldfacts_scorer = LongDocFACTScore(device=device)
bart_scorer = BARTScorer(checkpoint="facebook/bart-large", device=device)
bert_scorer = BERTScorer("bert-base-uncased", device=device)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="AggreFact", help="Dataset to run evaluation on")
args = parser.parse_args()

# Load the dataset
def load_dataframes(dataset):
    df = pd.read_csv('./data/AggreFACT/aggreFACT.csv')
    return df

def get_rouge_row(df, col_input, col_output, tgt_col):
    for idx, row in df.iterrows():
        rouge = test_huggingface_rouge([row[col_input]], [row[tgt_col]])
        for k, v in rouge.items():
            df.loc[idx, f"{col_output}_{k}"] = v
    return df

# BLEU Score
def get_bleu_row(df, col_input, col_output, tgt_col):
    for idx, row in df.iterrows():
        reference = [row[tgt_col]]
        candidate = row[col_input]
        bleu_score = sentence_bleu(candidate, reference).score
        df.loc[idx, col_output] = bleu_score
    return df

# METEOR Score
def get_meteor_row(df, col_input, col_output, tgt_col):
    for idx, row in df.iterrows():
        hypothesis = word_tokenize(row[col_input])
        reference = word_tokenize(row[tgt_col])        
        df.loc[idx, col_output] = meteor_score([reference], hypothesis)
    return df

# FuzzyWuzzy
def get_fuzzywuzzy_row(df, col_input, col_output, tgt_col):
    for idx, row in df.iterrows():
        df.loc[idx, col_output] = fuzz.ratio(row[col_input], row[tgt_col])
    return df

# Cosine Similarity
def get_cosine_similarity_row(df, col_input, col_output, tgt_col):
    vectorizer = TfidfVectorizer().fit(df[[col_input, tgt_col]].values.flatten())
    for idx, row in df.iterrows():
        vecs = vectorizer.transform([row[col_input], row[tgt_col]])
        cosine_sim = cosine_similarity(vecs[0:1], vecs[1:2])
        df.loc[idx, col_output] = cosine_sim[0, 0]
    return df

def get_bert_score_row(df, col_input, col_output, tgt_col):
    inputs = list(df[col_input])
    outputs = list(df[tgt_col])
    P_sci, R_sci, F1_sci = bert_scorer.score(inputs, outputs)
    df[col_output] = F1_sci
    return df

def get_bartscore_row(df, col_input, col_output, tgt_col):
    inputs = list(df[col_input])
    outputs = list(df[tgt_col])
    df[col_output] = np.array(bart_scorer.score(inputs, outputs))
    return df

def get_ldfacts_row(df, col_input, col_output, tgt_col):
    inputs = list(df[col_input])
    outputs = list(df[tgt_col])
    df[col_output] = np.array(ldfacts_scorer.score_src_hyp_long(inputs, outputs))
    return df

def get_questeval_row(df, col_input, col_output, tgt_col):
    inputs = list(df[col_input])
    outputs = list(df[tgt_col])
    df[col_output] = test_questeval(hypothesis=inputs, sources=outputs)
    return df

def create_data_file_factcc(df, model, col_output):
    path = f"./data/formatted_for_factcc/duc_2004/data-dev.json"
    subprocess.run(
        [
            "python",
            "evaluation_scripts/factCC/modeling/run.py",
            "--task_name",
            "factcc_annotated",
            "--do_eval",
            "--eval_all_checkpoints",
            "--do_lower_case",
            "--max_seq_length",
            "512",
            "--per_gpu_train_batch_size",
            "12",
            "--model_type",
            "bert",
            "--model_name_or_path",
            "factcc-checkpoint",
            "--data_dir",
            path,
            "--output_dir",
            "./factcc-checkpoint",
            "--tokenizer_name",
            "bert-base-uncased",
            "--config_name",
            "bert-base-uncased",
            "--no_cuda",
        ]
    )
    with open(os.path.join(path, "data-dev.jsonl"), "r") as f:
        data = f.readlines()
    data = [json.loads(d) for d in data]
    with open(os.path.join(path, "results.json"), "r") as f:
        results = json.load(f)
    for ii, row in enumerate(data):
        data[ii][col_output] = 1 - results[ii]
    df_results = pd.DataFrame(data)
    df_results = df_results[[col_output, "id"]].groupby("id").mean()
    df = df.join(df_results, on="id")
    return df

def get_factcc_row(df, col_input, col_output, tgt_col):
    return create_data_file_factcc(df, col_input, col_output)

def update_df_with_metric_scores(
        df, 
        metric_function, 
        src_cols=["summary"], 
        metric_name="rouge",
        tgt_col="doc",
        switch_tgt=False
):
    print(f"Calculating metrics for {metric_name}")
    output_cols = [f"{col}_{metric_name}" for col in src_cols]
    for src, output_col in zip(src_cols, output_cols):
        print(f"     Calculating output for {output_col}")
        if switch_tgt:
            df = metric_function(df, tgt_col, output_col, src)
        else:
            df = metric_function(df, src, output_col, tgt_col)
    return df

def get_scores_from_metrics(df, src_cols, ref_col, document):
    df = update_df_with_metric_scores(df, get_rouge_row, src_cols, "rouge", document)
    df = update_df_with_metric_scores(df, get_bleu_row, src_cols, "bleu", document)
    df = update_df_with_metric_scores(df, get_meteor_row, src_cols, "meteor", document)
    # df = update_df_with_metric_scores(df, get_nist_row, src_cols, "nist", ref_col)
    df = update_df_with_metric_scores(df, get_fuzzywuzzy_row, src_cols, "fuzzywuzzy", document)
    df = update_df_with_metric_scores(df, get_cosine_similarity_row, src_cols, "cosine_similarity", document)
    df = update_df_with_metric_scores(df, get_bert_score_row, src_cols, "bertscore", document)
    df = update_df_with_metric_scores(df, get_bartscore_row, src_cols, "bartscore_src_hyp", document, switch_tgt=True)
    df = update_df_with_metric_scores(df, get_ldfacts_row, src_cols, "ldfacts_src_hyp", document, switch_tgt=True)
    df = update_df_with_metric_scores(df, get_ldfacts_row, ref_col, "ldfacts_src_hyp", document, switch_tgt=True)
    # df = update_df_with_metric_scores(df, get_questeval_row, src_cols, "questeval", ref_col, switch_tgt=True)
    # df = update_df_with_metric_scores(df, get_factcc_row, src_cols, "factcc", text_col)
    return df


if __name__ == "__main__":

    df_all = load_dataframes(args.dataset)

    summary_cols = ["llama2chat7b_summary", "gpt35_summary", "gpt4oMini_summary", "gptInstruct_summary", "BART_summary", "T5_summary", "Pegasus_summary", "Orca7b2_summary", "Vic7b13_summary"]

    #change here
    ref_col = "summary"
    txt = 'doc'

    # Get scores from metrics
    df_all = get_scores_from_metrics(df_all, summary_cols, ref_col, txt)

    # Save updated DataFrame to CSV
    output_path = f"./data/AggreFACT/{args.dataset}_longdocfactscore.csv"
    df_all.to_csv(output_path, index=None)
    
    # Verify the DataFrame is not empty and has the expected columns
    if df_all.empty:
        raise ValueError("The DataFrame is empty after processing.")
    print("DataFrame columns:", df_all.columns)
