import scipy.stats as stats
import pandas as pd
import numpy as np
from training import main as retrain
from sklearn.metrics import accuracy_score

# first let's do t-test hypothesis
df1 = pd.read_csv('datalake/tables/customers_scored.csv')['age'].values
df2 = pd.read_csv('datalake/tables/customers_scored.csv')['age'].values

# t test for mean differences
results_t = stats.ttest_ind(df1, df2)

# score results
print('t score: ', results_t.statistic)
print('p: value', results_t.pvalue)

# accuracy score 

comparisons =pd.read_csv('datalake/tables/customers_comparison.csv')
acc = accuracy_score(comparisons['real_label'], comparisons['label'])

# evaluar los resultados de la prueba
if results_t.pvalue < 0.05 or acc < 0.8:
    if results_t.pvalue < 0.05:
        print(f'Age means are differents, p-value: {results_t.pvalue}, so now we have to re-train/re-fit!')
    elif acc < 0.8:
        print(f'Accuracy score is: {acc}, so now we have to re-train/re-fit!')
    retrain()
else:
    print('Accuracy scores are ok and mean values of records still the same!')