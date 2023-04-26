import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import warnings


warnings.filterwarnings('ignore')


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
    spectral = SpectralClustering(n_clusters=2, n_jobs=-1, affinity = 'sigmoid', n_neighbors=5, random_state = 1234, assign_labels = 'kmeans')
    df['label'] = spectral.fit_predict(aux_df)   
    df.to_csv('datalake/customers_clustered.csv', index=False)
    print('Dataset with clusters succesfully saved to Datalake!')
    print(f"silhouette_score: {silhouette_score(aux_df, df['label'].values)}")

def feature_engineering(df):
    print('Initiliazing feature engineering...')
    num_cols, cat_cols = num_and_cat_cols(df)

    # numerical features
    f_test = SelectKBest(f_classif, k = 3).set_output(transform='pandas')
    best_num_features = f_test.fit_transform(df[num_cols], df['label'])

    # categorical features
    encoder = OrdinalEncoder().set_output(transform='pandas')
    encoded = encoder.fit_transform(df[cat_cols])
    chi2_test = SelectKBest(chi2, k = 3).set_output(transform='pandas')
    best_cat_features = chi2_test.fit_transform(encoded, df['label'])
    aux = pd.concat([best_num_features, df[best_cat_features.columns.tolist()]], axis = 1)
    return pd.get_dummies(aux, dtype='float64')

def train_model(df):
    features = feature_engineering(df)
    print('Initializing training...')
    scaler = MinMaxScaler().set_output(transform='pandas')
    scaled = scaler.fit_transform(features.drop(columns=['label']))
    X, y = scaled, features['label'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    tree = DecisionTreeClassifier()
    param_grid = {'max_depth': [20, 25]}
    grid = GridSearchCV(tree, param_grid=param_grid, n_jobs=-1, cv = 5, verbose=0, scoring='accuracy')
    grid.fit(X_train, y_train)
    model = grid.best_estimator_
    best_params = grid.best_params_
    print(f"Params of best model: {best_params} ")
    print(f"Accuracy of model: {accuracy_score(y_test, model.predict(X_test)):.2%}")
    pipeline = Pipeline([('scaler', MinMaxScaler()), ('model', DecisionTreeClassifier(max_depth=best_params['max_depth']))])
    pipeline.fit(features.drop(columns=['label']), features['label'].values)

def main():
    df = pd.read_csv('datalake/customers_info.csv')
    create_clusters(df)
    clustered_data = pd.read_csv('datalake/customers_clustered.csv')
    train_model(clustered_data)

if __name__ == '__main__':
    main()
