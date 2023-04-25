import pandas as pd
import numpy as np

def etl():
    try:
        df = pd.read_csv('datalake/customers_info.csv')
        print('dataset read from datalake')
    except:
        df = pd.read_csv('https://raw.githubusercontent.com/jolual2747/bank-customer-churn-prediction/master/datasets/Churn_Modelling.csv', index_col= 0)
        print('dataset read from GitHub')
        df.reset_index(drop=True, inplace=True)
        df.drop(columns=['CustomerId', 'Surname', 'Exited'], inplace=True)
        for column in df.columns:
            df.rename(columns = {column:column.lower()}, inplace=True)
        df.to_csv('datalake/customers_info.csv', index=False)

if __name__ == '__main__':
    etl()