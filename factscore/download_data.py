import argparse
import os
import shutil
import zipfile
from pathlib import Path

import requests
import torch
import tqdm
import transformers


def _download_google_drive(file_id, destination):
    session = requests.Session()
    url = "https://docs.google.com/uc?export=download"
    response = session.get(url, params={"id": file_id}, stream=True, timeout=60)
    response.raise_for_status()
    token = next((value for key, value in response.cookies.items() if key.startswith("download_warning")), None)
    if token:
        response = session.get(url, params={"id": file_id, "confirm": token}, stream=True, timeout=60)
        response.raise_for_status()
    with open(destination, "wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)


def download_file(file_id, dest, cache_dir):
    destination = Path(dest)
    cached_destination = Path(cache_dir) / destination
    if destination.exists() or cached_destination.exists():
        print ("[Already exists] Skipping", dest)
        print ("If you want to download the file in another location, please specify a different path")
        return

    extracted = destination.with_suffix("") if destination.suffix == ".zip" else destination
    if extracted.exists() or (Path(cache_dir) / extracted).exists():
        print ("[Already exists] Skipping", dest)
        print ("If you want to download the file in another location, please specify a different path")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    if file_id.startswith("https://"):
        response = requests.get(file_id, stream=True, timeout=60)
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    else:
        _download_google_drive(file_id, destination)
    print("Download {} ... [Success]".format(dest))

    if destination.suffix == ".zip":
        with zipfile.ZipFile(destination) as archive:
            archive.extractall(destination.parent)
        destination.unlink()
        print("Unzip {} ... [Success]".format(dest))



def smart_tokenizer_and_embedding_resize(special_tokens_dict, tokenizer, model):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def recover_instruct_llama(path_raw, output_path, device="cpu", test_recovered_model=False):
    """Heavily adapted from https://github.com/tatsu-lab/stanford_alpaca/blob/main/weight_diff.py."""

    model_raw = transformers.AutoModelForCausalLM.from_pretrained(
        path_raw,
        device_map={"": torch.device(device)},
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model_recovered = transformers.AutoModelForCausalLM.from_pretrained(
        "kalpeshk2011/instruct-llama-7b-wdiff",
        device_map={"": torch.device(device)},
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )

    tokenizer_raw = transformers.AutoTokenizer.from_pretrained(path_raw)
    if tokenizer_raw.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token="[PAD]"),
            model=model_raw,
            tokenizer=tokenizer_raw,
        )
    tokenizer_recovered = transformers.AutoTokenizer.from_pretrained("kalpeshk2011/instruct-llama-7b-wdiff")

    state_dict_recovered = model_recovered.state_dict()
    state_dict_raw = model_raw.state_dict()
    for key in tqdm.tqdm(state_dict_recovered):
        state_dict_recovered[key].add_(state_dict_raw[key])

    if output_path is not None:
        model_recovered.save_pretrained(output_path)
        tokenizer_recovered.save_pretrained(output_path)

    if test_recovered_model:
        input_text = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\r\n\r\n"
            "### Instruction:\r\nList three technologies that make life easier.\r\n\r\n### Response:"
        )
        inputs = tokenizer_recovered(input_text, return_tensors="pt")
        out = model_recovered.generate(inputs=inputs.input_ids, max_new_tokens=100)
        output_text = tokenizer_recovered.batch_decode(out, skip_special_tokens=True)[0]
        output_text = output_text[len(input_text) :]
        print(f"Input: {input_text}\nCompletion: {output_text}")

    return model_recovered, tokenizer_recovered

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',
                        type=str,
                        default=".cache/factscore")
    parser.add_argument('--model_dir',
                        type=str,
                        default=".cache/factscore")
    parser.add_argument('--llama_7B_HF_path',
                        type=str,
                        default=None)

    args = parser.parse_args()

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)

    download_file("1IseEAflk1qqV0z64eM60Fs3dTgnbgiyt", "demos.zip", args.data_dir)
    download_file("1enz1PxwxeMr4FRF9dtpCPXaZQCBejuVF", "data.zip", args.data_dir)
    download_file("1mekls6OGOKLmt7gYtHs0WGf5oTamTNat", "enwiki-20230401.db", args.data_dir)

    if args.llama_7B_HF_path:
        recover_instruct_llama(args.llama_7B_HF_path, os.path.join(args.model_dir, "inst-llama-7B"))

    download_file(
        "https://raw.githubusercontent.com/shmsw25/FActScore/main/roberta_stopwords.txt",
        "roberta_stopwords.txt",
        args.data_dir,
    )

    for source in [Path("demos"), Path("enwiki-20230401.db"), Path("roberta_stopwords.txt")]:
        destination = Path(args.data_dir) / source.name
        if source.exists() and source.resolve() != destination.resolve():
            shutil.move(str(source), str(destination))

