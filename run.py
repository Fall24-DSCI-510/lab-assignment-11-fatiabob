# Solution code for the Iris Dataset Homework (run.py)

import pandas as pd
from scipy.stats import zscore

# Question 1: Pre-process the data
def preprocess_data(input_filename):
    df = pd.read_csv(input_filename, sep=',')
    cols =['SepalLengthCm','SepalWidthCm','PetalLengthCm',
                          'PetalWidthCm','Species']
    df.columns = cols
    
    df['SepalLengthCm_z'] = abs((df['SepalLengthCm'] - df['SepalLengthCm'].mean())/ df['SepalLengthCm'].std())
    df['SepalWidthCm_z'] = abs((df['SepalWidthCm'] - df['SepalWidthCm'].mean())/ df['SepalWidthCm'].std())

    df_new = df[
    (df['SepalLengthCm_z'] > -2) & 
    (df['SepalLengthCm_z'] < 2) & 
    (df['SepalWidthCm_z'] > -2) & 
    (df['SepalWidthCm_z'] < 2)].copy()

    df_new = df_new.drop(['SepalLengthCm_z','SepalWidthCm_z'], axis=1)
    df_new['ID'] = range(1, len(df_new) + 1)

    return df_new


# Question 2: Descriptive Statistics Functions
def species_count(data):
    data=preprocess_data(data)
    return data['Species'].value_counts().to_dict()

def average_sepal_length(data):
    data=preprocess_data(data)
    return round(data['SepalLengthCm'].mean(),1)

def max_petal_width(data):
    data=preprocess_data(data)
    return round(data['PetalWidthCm'].max(),1)

def min_petal_length(data):
    data=preprocess_data(data)
    return round(data['PetalLengthCm'].min(),1)

def count_sepal_length_above_5(data):
    data=preprocess_data(data)
    return (data['SepalLengthCm']>5).sum()


# Question 3: Analysis Functions
def count_petal_length_below_2(data):
    data=preprocess_data(data)
    return int((data['PetalLengthCm'] < 2.0).sum())

def get_sepal_width_above_3_5(data):
    data=preprocess_data(data)
    return sorted(data[data['SepalWidthCm']>3.5]['ID'].to_list())

def species_count_petal_width_above_1_5(data):
    data=preprocess_data(data)
    return data[data['PetalWidthCm']>1.5]['Species'].value_counts().to_dict()

def get_virginica_petal_length_above_6(data):
    data=preprocess_data(data)
    return sorted(data[
        (data['Species'] == 'Iris-virginica') & 
        (data['PetalLengthCm'] > 6.0)]['ID'].tolist())
    
def get_largest_sepal_width(data):
    data=preprocess_data(data)
    return data[data['SepalWidthCm'] == data['SepalWidthCm'].max()]['ID'].iloc[0]