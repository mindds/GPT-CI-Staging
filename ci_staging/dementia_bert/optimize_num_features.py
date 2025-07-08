import os
os.environ["OMP_NUM_THREADS"] = "4" 
os.environ["OPENBLAS_NUM_THREADS"] = "4" 
os.environ["MKL_NUM_THREADS"] = "6" 
os.environ["VECLIB_MAXIMUM_THREADS"] = "4" 
os.environ["NUMEXPR_NUM_THREADS"] = "6" 
import pandas as pd
import numpy as np
import re
import random
from copy import deepcopy
import time
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from sklearn import metrics
from sklearn.metrics import cohen_kappa_score, accuracy_score
from sklearn.feature_selection import RFE
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

from tqdm import tqdm

class data_generator():
    def __init__(self,sections, keywords, data):
        self.sections = sections
        self.keywords = keywords
        self.data = data
        features = {}
        for key in keywords:
            features[key] = {}
            for section in sections:
                features[key][section] = [col for col in data.columns if (section in col) & (key in col)]
                features[key][section][0:0] = data.columns.tolist()[0:5]
        self.features = features
    def get_data(self):
        df = {}
        for key in self.keywords:
            df[key] = {}
            for section in self.sections:
                df[key][section] = self.data[self.features[key][section]].copy()
        return df
    
class data_splitter():
    def __init__ (self, data, balanced = True):
        '''
        data is for get the IDs and the diagnosis
        '''
        self.data = data
        self.balanced = balanced
        
    def get_train_test_id(self, kfold = 10, test_ratio = 0.1):
        diagnosis = [0, 2, 4]
        df = deepcopy(self.data)

        # Filter the dataset for relevant classes
        df = df[df.syndromic_dx.isin(diagnosis)]
        
        # Get indices for each class
        class_0_idx = df[df.syndromic_dx == 0].index
        class_2_idx = df[df.syndromic_dx == 2].index
        class_4_idx = df[df.syndromic_dx == 4].index

        # Get unique patient IDs for each class
        class_0_ids = df.loc[class_0_idx].PatientID.unique()
        class_2_ids = df.loc[class_2_idx].PatientID.unique()
        class_4_ids = df.loc[class_4_idx].PatientID.unique()

        print(f"Class 0: {len(class_0_ids)}, Class 2: {len(class_2_ids)}, Class 4: {len(class_4_ids)}")

        if self.balanced:
            min_class_size = min(len(class_0_ids), len(class_2_ids), len(class_4_ids))
            id_list = list(class_0_ids[:min_class_size]) + list(class_2_ids[:min_class_size]) + list(class_4_ids[:min_class_size])
        else:
            id_list = list(class_0_ids) + list(class_2_ids) + list(class_4_ids)

        random.seed(42)
        random.shuffle(id_list)
        print(f"Total IDs for classification: {len(id_list)}")
        # Generate labels for patient IDs
        id_label = [df[df["PatientID"] == i].syndromic_dx.values[0] for i in id_list]

        # Split the data using StratifiedShuffleSplit for balanced class representation
        sss = StratifiedShuffleSplit(n_splits=kfold, test_size=test_ratio, random_state=42)
        train_id, test_id = [], []
        for train_index, test_index in sss.split(id_list, id_label):
            train_id.append(train_index)
            test_id.append(test_index)

        return id_list, train_id, test_id       
    
    def mapped_data(self, data_dict):
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

def metrics_cal(y_pred, y_test):
    # Quadratic Cohen's kappa score
    kappa = cohen_kappa_score(y_test, y_pred, weights='quadratic')
    
    # Other evaluation metrics (like accuracy) if needed
    acc = metrics.accuracy_score(y_test, y_pred)
    
    print(f"Quadratic Cohen's Kappa: {kappa}")
    print(f"Accuracy: {acc}")
    
    return [kappa, acc]

def random_sampling_score(id_list_, data_, train_id_, test_id_, num_features_, n=10):
    """
    3-class classification with XGBoost.
    """
    id_list = np.array(id_list_)
    col_to_drop = ["syndromic_dx", "PatientID", "syndromic_dx_certainty", "dementia_severity", "dementia_severity_certainty"]
    
    mat_avg = [{'Random_sampling':[]} for i in range(2)]  # Two metrics: Quadratic Kappa and Accuracy

    for m in tqdm(range(n), desc='Random-sampling-method loop', position=0):
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
        for i in test['PatientID'].unique():
            pred_arr = model_xgb.predict_proba(test[test['PatientID'] == i].drop(columns=col_to_drop))
            pred_class = np.argmax(np.mean(pred_arr, axis=0))  # Get class with highest mean probability
            label = test[test['PatientID'] == i]['syndromic_dx'].tolist()[0]
            y_pred.append(pred_class)
            y_test_labels.append(label)

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
    df_john_cleaned = pd.read_pickle('dataframes/df_john_cleaned.pkl')
    data = pd.read_pickle('dataframes/data_3year_30encoded.pkl')
    sections = df_john_cleaned.EncounterTypeDSC.unique().tolist()
    folder_name_main = 'results-optimizer-dementiabert-RS'
    bal = False
    df_embedding = data_generator(sections, ['Dem1','ADL2'], data).get_data()
    embedding_obj = data_splitter(df_embedding['Dem1']['Office Visit'], balanced = bal)
    # get 10-fold CV; save patient ids
    id_list1, train_id1, test_id1 = embedding_obj.get_train_test_id()
    data2_mapped = embedding_obj.mapped_data(data)

    # get inner train-validation splits within the training of each fold; save patient ids
    train_id2 = []
    valid_id2 = []
    for f in range(len(train_id1)):
        random.seed(41)
        train_id2.append(np.array(random.sample(list(train_id1[f]), int(len(train_id1[f])*0.8))))
        valid_id2.append(np.array([ids for ids in train_id1[f] if ids not in train_id2[f]]))
        
    num_f = [512, 256, 128, 64, 32, 16]    
    for num in tqdm(num_f, desc='features-loop',position=0):
        print('num-features'+str(num))
        folder_name = f"{folder_name_main}/num_features{num}"
        if not os.path.exists(f'{folder_name}'):
            os.makedirs(f'{folder_name}')
        
        res = random_sampling_score(id_list1,data2_mapped,train_id2,valid_id2,num,n=10)
        res_path = f'{folder_name}/df_results.csv'
        res.to_csv(res_path)   
    print('Done!')
