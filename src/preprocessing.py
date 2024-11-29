import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

def load_data(data_path):
    df = pd.read_csv(data_path)
    return df

def preprocess_data(df):
    encoder = OneHotEncoder(sparse_output=False)
    district_encoded = encoder.fit_transform(df[['district_label']])
    
    X_numeric = df[['students_passed', 'num_teachers', 'total_students', 'Total_budget', 
                    'dropout_rate', 'avg_class_size', 'school_distance']].values
    X = np.hstack((X_numeric, district_encoded))
    
    district_mapping = {'Nyarugenge': 0, 'Kicukiro': 1, 'Gasabo': 2}
    df['target'] = df['district_label'].map(district_mapping)
    y = df['target'].values
    return X, y

def split_data(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    return X_train, X_val, X_test, y_train, y_val, y_test

def scale_data(X_train, X_val, X_test):
    scaler = StandardScaler()
    X_train[:, :4] = scaler.fit_transform(X_train[:, :4])
    X_val[:, :4] = scaler.transform(X_val[:, :4])
    X_test[:, :4] = scaler.transform(X_test[:, :4])
    return X_train, X_val, X_test

# preprocessing student dataset

def process_student_data(df):
    df = df.drop(columns=['StudentID','GradeClass'])
    df.head()
    X = df.drop(columns=['GPA'])
    y = df['GPA']
    
    return X, y

def scale_student_data(X):
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

def split_student_data(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test

