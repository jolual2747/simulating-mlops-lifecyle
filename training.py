import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

def num_and_cat_cols(df):    
    # num cols should be float or int, if int should have less than 5 unique values
    num_cols = []
    for column in df.columns:
        if len(np.unique(df[column])) >= 5 and (df[column].dtype == np.dtype('float') or df[column].dtype == np.dtype('int')):
            num_cols.append(column)

    cat_cols = []
    for column in df.columns:
        if df[column].dtype == np.dtype('object') or (df[column].dtype == np.dtype('int') and len(np.unique(df[column])) < 5):
            cat_cols.append(column)
    return num_cols, cat_cols

def create_clusters(df):
    print('Creating clusters...')
    num_cols, cat_cols = num_and_cat_cols(df)
    nums_df = df[num_cols].copy()
    cats_df = df[cat_cols].copy()
    scaler = MinMaxScaler()
    num_df = pd.DataFrame(scaler.fit_transform(nums_df), columns=num_cols)
    cat_df = pd.get_dummies(cats_df, dtype= 'float')
    aux_df = pd.concat([num_df, cat_df], axis = 1)
    spectral = SpectralClustering(n_clusters=2, n_jobs=-1, affinity= 'rbf', gamma = 0.05, random_state = 1234, assign_labels = 'kmeans')
    aux_df['label'] = spectral.fit_predict(aux_df)   
    aux_df.to_csv('datalake/customers_clustered.csv', index=False)
    print('Dataset with clusters succesfully saved to Datalake!')

def feature_engineering(df):
    print('Initiliazing training...')
    num_cols, cat_cols = num_and_cat_cols(df)



if __name__ == '__main__':
    df = pd.read_csv('datalake/customers_info.csv')
    create_clusters(df)
