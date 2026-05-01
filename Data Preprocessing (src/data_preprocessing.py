import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(path):
    return pd.read_csv(path)

def preprocess_data(df):
    scaler = StandardScaler()
    df['Amount'] = scaler.fit_transform(df[['Amount']])
    
    df = df.drop(['Time'], axis=1)
    
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    return X, y
