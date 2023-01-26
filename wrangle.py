### module for functions prepare and acquire
import os
import pandas as pd
from scipy import stats
from pydataset import data
import numpy as np
import wrangle
import env
from sklearn.model_selection import train_test_split



def get_connection(db, username=env.username, host=env.host, password=env.password):
    return f'mysql+pymysql://{username}:{password}@{host}/{db}'
    '''
    this function acts as a part of the function below to establish a connection
    with the sql server
    '''
    
def get_zillow_data():
    
    '''
    this function retrieves the zillow info from the sql server
    or calls up the csv if it's saved in place
    
    '''
    
    filename = "zillow_sfr.csv"

    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        # read the SQL query into a dataframe
        df = pd.read_sql("SELECT * FROM properties_2017 JOIN propertylandusetype USING (propertylandusetypeid) JOIN predictions_2017 USING (id) HAVING transactiondate BETWEEN '2017-01-01' AND '2017-12-31' ", get_connection ('zillow'))

        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename)

        # Return the dataframe to the calling code
        return df
    
    
def train_val_test(df):
    '''
    this function splits up the data into sections for training,
    validating, and testing
    models
    '''
    seed = 99
    train, val_test = train_test_split(df, train_size = 0.7,
                                       random_state = seed)
    
    validate, test = train_test_split(val_test, train_size = 0.5, random_state = seed)
    
    return train, validate, test

def clean_prep_zillow(df):
    df = df.dropna()
    df.drop_duplicates(inplace = True)
    df.drop([df.columns[0]], axis = 1, inplace = True)
    df.rename(columns = {'bedroomcnt' : 'bedroom',
                       'bathroomcnt': 'bathroom',
                       'calculatedfinishedsquarefeet':'sqrft',
                       'taxvaluedollarcnt':'tax_value',
                       'yearbuilt':'year_built'}, inplace = True)
    
    train, validate, test = train_val_test(df)
    
    return train, validate, test

def na_stats(df):
    odf = pd.DataFrame({'features': list(df.columns)})
    num_na = []
    percent_na = []
    for col in df.columns:
        num_na.append(df[col].isna().sum())
        percent_na.append((df[col].isna().sum()/len(df[col])))
    odf['num_na'] = num_na         
    odf['percent_na'] = percent_na
        
    return odf

def gap_killer(df):
    to_drop = []
    col_count = na_stats(df)
    for index, row in col_count.iterrows():
        if row['percent_na'] >= .60:
            to_drop.append(index)
        out_df = df.drop(df.columns[to_drop], axis = 1)
    null_row = out_df.isnull().sum(axis=1)
    out_df['nulls'] = null_row
    out_df = out_df[out_df['nulls'] < 9]
    out_df.drop(columns = ['nulls', 'isdupe'], inplace = True)
    return out_df
        