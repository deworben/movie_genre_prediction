import numpy as np
import pandas as pd

#Try to guess the label of this data where the data is segregated in 5 year splits
#1. Create the datasets
####### For each 5-year datagroup AND the whole dataset (AND try decade?)
#2. Predict with perceptron - With just image and then all data
#3. Predict with something else
#4. Predict with shitty baseline for comparison 
#5. Use a svm/another grouping algo to see which years/movie are closest - find years by grouping

df = pd.read_csv('train_features.tsv', sep='\t')

# filtered = df[df.year.isin(['2015'])]
# print(filtered)


def data_from_time_range(data, start_date, end_date):
    years_allowed = []
    for i in range(start_date, end_date):
        years_allowed.append(str(i))
    
    filtered_data = data[data.year.isin(years_allowed)]
    return filtered_data

a = data_from_time_range(df, 1900, 1920)[['title', 'year']]

#def predict_for_ever_x_years(x):


