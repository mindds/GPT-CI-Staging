import os
import pandas as pd
import numpy as np
import random
import re
from tqdm import tqdm
tqdm.pandas()
import torch
from transformers import AutoTokenizer, AutoModel
from transformers import BertForSequenceClassification, BertTokenizer
from torch import Tensor

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r"[^0-9A-Za-z.' ]", ' ', text)
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def read_summary_files(folder_path):
    txt_ls, patientid_ls = [], []
    pattern = os.path.join(folder_path, '*_summaries.txt')
    for file_path in glob.glob(pattern):
        file_name = os.path.basename(file_path)
        id_number = file_name.split('_')[0]
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                txt_ls.append(clean_text(content))
                patientid_ls.append(id_number)
        except Exception as e:
            print(f"Failed to read {file_path}: {e}")
    return pd.DataFrame({"PatientID": patientid_ls, "SummariesTXT": txt_ls})

# Embedding Generation
def tokenize_chunk_embed(df, model, tokenizer, max_length=512):
    all_rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Encoding Summaries"):
        patient_id, text = row['PatientID'], row['SummariesTXT']
        tokenized = tokenizer(text, return_tensors='pt', truncation=False)
        input_ids = tokenized['input_ids'][0]
        attention_mask = tokenized['attention_mask'][0]

        for i in range(0, len(input_ids), max_length):
            chunk_input_ids = input_ids[i:i+max_length]
            chunk_attention_mask = attention_mask[i:i+max_length]

            chunk = {
                'input_ids': chunk_input_ids.unsqueeze(0).to(device),
                'attention_mask': chunk_attention_mask.unsqueeze(0).to(device)
            }

            with torch.no_grad():
                outputs = model(**chunk)
                hidden_states = outputs.hidden_states
                last_four = torch.stack(hidden_states[-4:]).sum(0)
                embedding = last_four.sum(1).squeeze().cpu().tolist()

            all_rows.append([patient_id] + embedding)
    return pd.DataFrame(all_rows)

if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    finetune_dir = 'ClinicalBERT/ClinicalBERT-Training/0719/trial_4/'
    tokenizer = BertTokenizer.from_pretrained(finetune_dir)
    model = BertForSequenceClassification.from_pretrained(finetune_dir, output_hidden_states=True).to(device)

    folder_path = 'summaries/gpt'  
    df = read_summary_files(folder_path)
    emb_df = tokenize_chunk_embed(df, model, tokenizer)
    emb_df.to_pickle('dataframes/data_3year_summaries_encoded.pkl')
