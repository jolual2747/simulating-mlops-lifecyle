import joblib
import pandas as pd
import numpy as np
from training import feature_engineering

def load_pipeline():
    pipeline = joblib.load('datalake/models/pipeline.joblib')
    return pipeline

def preprocessing(features_dict):
    new_dict = {}
    for k in features_dict.keys():
        new_dict[k] = [features_dict[k]]
    return pd.DataFrame.from_dict(new_dict)

def make_prediction(json):
    pipeline= load_pipeline()
    df = preprocessing(json)
    pred = pipeline.predict(df)[0]
    prob = pipeline.predict_proba(df)[0][0]
    #return {"class": pred, "prob": round(prob,2)}
    return pred, round(prob,2)


    #my_json = {"age": 45, "tenure": 4, "balance": 10452.4, "gender": "Female", "numofproducts": 3}

    

