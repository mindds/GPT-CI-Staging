import os
import pandas as pd
import numpy as np
import random
import re
from tqdm import tqdm
tqdm.pandas()

import seaborn as sns
import matplotlib.pyplot as plt 

# functions to get word embedding for a specific word in a sentence 
import torch
from transformers import AutoTokenizer, AutoModel
from transformers import BertForSequenceClassification, BertTokenizer
# from torch import Tensor

import torch
from transformers import AutoModel, AutoTokenizer
import random


from sklearn.decomposition import PCA
import umap.umap_ as umap
def clean_text(text):
    '''
    Cleaning the text files- replacing the number with N- removing the punctuation and URLs
    '''
    text = text.lower()
    text = re.sub(r'http\S+', '', text) # substitute webpage
    text = re.sub(r'www\S+', '', text) # webpage
    text = re.sub(r'((1-\d{3}-\d{3}-\d{4})|(\(\d{3}\) \d{3}-\d{4})|(\d{3}-\d{3}-\d{4}))$', '', text); # phone number
    text = re.sub("[^0-9A-Za-z.']", " " ,text) # keep the number or letter. remove weird symbols. 
    text = re.sub("\.{2}",".", text) # making two dots one dot
    text = re.sub("\s+"," ", text) # white space
    text = re.sub('\d+', 'N', text) # replacing numbers with "N"
    text = re.sub('N,N', 'N', text) 
    text = re.sub('N.N', 'N', text)
    return text

def get_bert_embedding(text, model):
    """
    Function to encode a sentence using BERT, extracting embeddings from the last four layers.
    """
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    inputs = {key: tensor.to(device) for key, tensor in inputs.items()}

    # Forward pass to get hidden states from BERT
    with torch.no_grad():
        outputs = model(**inputs)

    # Get hidden states from all layers
    hidden_states = outputs.hidden_states

    # Sum the last four layers to create the sentence embedding
    last_four_layers = hidden_states[-4:]
    sum_embeddings = torch.stack(last_four_layers, dim=0).sum(dim=0)  # Sum of the last four layers
    sentence_embedding = sum_embeddings.sum(dim=1).squeeze()  # Sum along the sequence length

    return sentence_embedding.cpu().numpy()  # Convert to numpy array

def sent_to_num(df_text, df_features, column_names, model):
    '''
    Encoding the sentences with BERT.
    Parameters:
        df_text: Data containing the sentences.
        df_features: Dataframe to store the encoded sentences.
        column_names: The feature columns (like IDs, diagnoses) to keep in the final encoded dataframe.
        model: The BERT model for encoding.
    Return:
        Encoded data for the entire dataset.
    '''
    id_list = df_text['PatientID'].unique().tolist()
    
    for PatientID in tqdm(id_list):
        # Get the column values to retain
        info = df_text[df_text["PatientID"] == PatientID].iloc[0][column_names].values.tolist()
        
        # Collect all sentences
        txt_arr1 = df_text[df_text["PatientID"] == PatientID].sents_table1.tolist()
        txt_arr2 = df_text[df_text["PatientID"] == PatientID].sents_table2.tolist()
        flat_arr = [item for sublist in txt_arr1 + txt_arr2 for item in sublist]
        
        sampling_len = 15
        
        # If fewer than 15 sentences, try additional sentence sources
        if len(flat_arr) < 15:
            txt_arr3 = df_text[df_text["PatientID"] == PatientID].sents.tolist()
            flat_arr2 = [item for sublist in txt_arr3 for item in sublist]
            flat_arr = flat_arr2
            if len(flat_arr2) < 15:
                sampling_len = len(flat_arr2)

        # Sample and encode sentences
        for i in range(30):  # 30 samples per patient
            arr_sampled = random.sample(flat_arr, sampling_len)
            txt = ' '.join(arr_sampled)
            txt = clean_text(txt)  # Clean text before embedding

            # Use BERT to embed the sampled text
            features = get_bert_embedding(txt, model).tolist()  # Get embedding as a list
            
            # Combine the extracted features with the patient info
            arr = info + features
            
            # Ensure correct number of features and append to the dataframe
            if len(arr) == len(df_features.columns):
                new_row = pd.DataFrame([arr], columns=df_features.columns)
                df_features = pd.concat([df_features, new_row], ignore_index=True)
            else:
                print(f"Length mismatch between arr and df_features.columns at iteration {i}!")
                print(f"Length of arr: {len(arr)}, Length of df_features.columns: {len(df_features.columns)}")
    
    return df_features

def create_df(col):  
    '''
    Create a dataframe for embedding the text data
        the Embedding will generate an output of 512 dimention
    Parameters:
        col_names: these features are the ids and diagnoses that we want to have in the final encoded dataframe.
        a dataframe with columns named for 512 features for each category
    '''
    col_n = col.copy()
    for counter in range(768):
        col_n.append('feature'+str(counter))
    DF = pd.DataFrame(columns=col_n) 
    return DF

if __name__ == "__main__":
    # load model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    finetune_dir = 'ClinicalBERT/ClinicalBERT-Training/0719/trial_4/'
    model = BertForSequenceClassification.from_pretrained(finetune_dir, output_hidden_states=True, output_attentions=True).to(device)
    tokenizer = BertTokenizer.from_pretrained(finetune_dir)

    # load data
    df = pd.read_pickle('dataframes/df_john_cleaned.pkl')
    keyword_class = ['Dem1','ADL2']
    col_names = ["PatientID","syndromic_dx","syndromic_dx_certainty","dementia_severity","dementia_severity_certainty"]

    df_features = create_df(col_names)
    df_encoded = sent_to_num(df,df_features,col_names,model)
    df_encoded.to_pickle('dataframes/data_3year_30encoded.pkl')
