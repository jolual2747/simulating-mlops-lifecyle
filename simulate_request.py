import requests
import joblib
import pandas as pd
import numpy as np

# features to request post on API
features_training = joblib.load('datalake/models/features_in_training.joblib')

database = pd.read_csv('datalake/tables/customers_scored.csv')
# features to ask at input
all_features = database.drop(columns=['label']).columns.tolist()

data = {}
for column in all_features:
    if len(np.unique(database[column])) <= 3 and (database[column].dtype == np.dtype('float') or database[column].dtype == np.dtype('int') or database[column].dtype == np.dtype('object')):
        _input = input("Enter "+column+f'(choose between = {np.unique(database[column])}): ')
    else:
        _input = input("Enter "+column+f'(dtype = {database[column].dtype}): ')
    try:
        data[column] = float(_input)
    except:
        data[column] = _input

# let's take only the features that model at API receive
json = {}
for k in features_training:
    json[k] = data[k]

# request post method to API
request = requests.post(url='http://127.0.0.1:5000/predict', json=json)
response = request.json()
print(f"The predicted class is {response['class']}!")

# convert all collected data from user in the way that pd.DataFrame receives records from dict in order to store it (persist prediction)
new_record_dict = {}
for k in data.keys():
    new_record_dict[k] = [data[k]]
new_record = pd.DataFrame.from_dict(new_record_dict)
new_record['label'] = response['class']

# overwrite database with predicted classes
database = pd.concat([database, new_record], axis = 0, ignore_index=True)
database.to_csv('datalake/tables/customers_scored.csv', index=False)
print("The record and its class have been stored at datalake succesfully!")