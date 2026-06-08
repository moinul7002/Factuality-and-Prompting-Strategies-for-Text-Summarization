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
import subprocess
import time
import os
import seaborn as sb
import torch

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

ldfacts_scorer = LongDocFACTScore(device=device)
bart_scorer = BARTScorer(checkpoint="facebook/bart-large", device=device)
bert_scorer = BERTScorer("bert-base-uncased", device=device)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="duc_2004", help="Dataset to run evaluation on")
args = parser.parse_args()

# Load DUC 2004 dataset
def load_dataframes(dataset):
    df = pd.read_csv('./data/DUC_2004/duc_2004.csv')
    return df

def get_rouge_row(df, col_input, col_output, tgt_col):
    for idx, row in df.iterrows():
        rouge = test_huggingface_rouge([row[col_input]], [row[tgt_col]])
        for k, v in rouge.items():
            df.loc[idx, f"{col_output}_{k}"] = v
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
        src_cols=["Summary1","Summary2", "Summary3", "Summary4"], 
        metric_name="rouge", 
        tgt_col="Texts", 
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

def get_scores_from_metrics(df, src_cols, text_col):
    df = update_df_with_metric_scores(df, get_rouge_row, src_cols, "rouge", text_col)
    df = update_df_with_metric_scores(df, get_bert_score_row, src_cols, "bertscore", text_col)
    df = update_df_with_metric_scores(df, get_bartscore_row, src_cols, "bartscore_src_hyp", text_col, switch_tgt=True)
    df = update_df_with_metric_scores(df, get_ldfacts_row, src_cols, "ldfacts_src_hyp", text_col, switch_tgt=True)
    # df = update_df_with_metric_scores(df, get_questeval_row, src_cols, "questeval", ref_col, switch_tgt=True)
    # df = update_df_with_metric_scores(df, get_factcc_row, src_cols, "factcc", text_col)
    return df


if __name__ == "__main__":

    df_all = load_dataframes(args.dataset)

    summary_cols = ["Summary1","Summary2", "Summary3", "Summary4"]
    text_col = "Texts"

    # Get scores from metrics
    df_all = get_scores_from_metrics(df_all, summary_cols, text_col)

    # Save updated DataFrame to CSV
    output_path = f"{args.dataset}_longdocfactscore.csv"
    df_all.to_csv(output_path, index=None)
    
    # Verify the DataFrame is not empty and has the expected columns
    if df_all.empty:
        raise ValueError("The DataFrame is empty after processing.")
    print("DataFrame columns:", df_all.columns)
