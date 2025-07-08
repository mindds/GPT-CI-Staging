import os
os.environ["OMP_NUM_THREADS"] = "4" 
os.environ["OPENBLAS_NUM_THREADS"] = "4" 
os.environ["MKL_NUM_THREADS"] = "6" 
os.environ["VECLIB_MAXIMUM_THREADS"] = "4" 
os.environ["NUMEXPR_NUM_THREADS"] = "6" 
import pandas as pd
import numpy as np
import re
import pickle
import random
from copy import deepcopy
import time
from xgboost import XGBClassifier

from sklearn import metrics
from sklearn.metrics import cohen_kappa_score, accuracy_score
from sklearn.feature_selection import RFE
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

from tqdm import tqdm

def mapped_data(data_dict):
    """
    Filters and maps the data to the desired 3-class structure (0, 2, 4 -> 0, 1, 2)
    for the classifier. It also supports mapping back to the original labels.
    """
    label_mapping = {0: 0, 2: 1, 4: 2}
    # inverse_label_mapping = {v: k for k, v in label_mapping.items()}  # For reverse mapping
    mapped_data = deepcopy(data_dict)
    diagnosis = [0, 2, 4]
    mapped_data = mapped_data[mapped_data.syndromic_dx.isin(diagnosis)]
    mapped_data['syndromic_dx'] = mapped_data['syndromic_dx'].map(label_mapping)
    return mapped_data

def summaries_score(id_list_, data_, train_id_, test_id_, num_features_, save_path, n=10):
    """
    3-class classification with XGBoost.
    Save test set predictions for each fold.
    """
    id_list = np.array(id_list_)
    col_to_drop = ["syndromic_dx", "PatientID"]
    
    mat_avg = [{'Random_sampling':[]} for i in range(2)]  # Two metrics: Quadratic Kappa and Accuracy

    for m in tqdm(range(n), desc='Summaries-chunk-method loop', position=0):
        print(f'fold{m}')
        datacp = deepcopy(data_)
        test = datacp[datacp['PatientID'].isin(id_list[test_id_[m]])]
        train = datacp[datacp['PatientID'].isin(id_list[train_id_[m]])]
        y_train = train.syndromic_dx
        x_train = train.drop(columns=col_to_drop)
        # y_test = test.syndromic_dx
        x_test = test.drop(columns=col_to_drop)

        # Feature Selection using XGBoost and RFE
        model = XGBClassifier(eval_metric='mlogloss', n_jobs=-1)
        print(f"Starting selection for {num_features_} features")
        start_time = time.time()
        rfe = RFE(model, n_features_to_select=num_features_, step=10)
        fit = rfe.fit(x_train.values, y_train.values)
        print(f"Feature selection done in {time.time() - start_time} seconds")

        arr = x_train.keys().array[0:768]
        remove_col = np.ma.masked_array(arr, fit.support_)
        x_train = x_train.drop(columns=remove_col[~remove_col.mask].data)
        x_test = x_test.drop(columns=remove_col[~remove_col.mask].data)
        test = test.drop(columns=remove_col[~remove_col.mask].data)

        # Train XGBoost on the reduced feature set
        model_xgb = XGBClassifier(eval_metric='mlogloss', random_state=0)
        y_train = y_train.astype('int')
        model_xgb.fit(x_train, y_train)

        y_pred = []
        y_test_labels = []
        test_patient_ids = []

        for i in test['PatientID'].unique():
            pred_arr = model_xgb.predict_proba(test[test['PatientID'] == i].drop(columns=col_to_drop))
            pred_class = np.argmax(np.mean(pred_arr, axis=0))  # Get class with highest mean probability
            label = test[test['PatientID'] == i]['syndromic_dx'].tolist()[0]
            y_pred.append(pred_class)
            y_test_labels.append(label)
            test_patient_ids.append(i)

        # Save predictions for this fold
        predictions_df = pd.DataFrame({
            'PatientID': test_patient_ids,
            'syndromic_dx': y_test_labels,
            'predicted_dx': y_pred
        })
        fold_save_path = f'{save_path}/fold_{m}_predictions.csv'
        predictions_df.to_csv(fold_save_path, index=False)

        # Calculate evaluation metrics (Quadratic Kappa and Accuracy)
        kappa = cohen_kappa_score(y_test_labels, y_pred, weights='quadratic')
        acc = accuracy_score(y_test_labels, y_pred)

        mat_avg[0]['Random_sampling'].append(kappa)  # Quadratic Kappa
        mat_avg[1]['Random_sampling'].append(acc)    # Accuracy

    print(f"Quadratic Cohen's Kappa is {np.mean(mat_avg[0]['Random_sampling'])}")
    print(f"Accuracy is {np.mean(mat_avg[1]['Random_sampling'])}")
    
    df_res = pd.DataFrame(mat_avg, index=['Quadratic Kappa', 'Accuracy'])
    
    return df_res

    
if __name__ == "__main__":
    embeddings_df = pd.read_pickle('dataframes/data_3year_summaries_encoded.pkl')
    mapped_embeddings_df = mapped_data(embeddings_df)

    filename_ids = "dataframes/data_cv_id_list.pkl"
    with open(filename_ids, 'rb') as f:
        id_list1 = pickle.load(f)
    filename_train = "dataframes/data_cv_train1.pkl"
    with open(filename_train, 'rb') as f:
        train_id1 = pickle.load(f)
    filename_test = "dataframes/data_cv_test1.pkl"
    with open(filename_test, 'rb') as f:
        test_id1 = pickle.load(f)

    folder_name_main = 'results-predictions-dementiabert-summaries'
    if not os.path.exists(folder_name_main):
        os.makedirs(folder_name_main)

    num_features = 512
    res = summaries_score(id_list1, mapped_embeddings_df, train_id1,test_id1, num_features, save_path=folder_name_main, n=10)
    res.to_csv(f'{folder_name_main}/df_results.csv')   
    print('Done!')
