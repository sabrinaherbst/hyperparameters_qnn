import pandas as pd
from sklearn.datasets import fetch_covtype, fetch_kddcup99
from sklearn.model_selection import train_test_split
import numpy as np
import time


def load_covtype(path: str = '../../data/external/covtype/'):
    """
    Load covtype dataset and split into train, validation and test sets.
    :param path: path to save data
    """
    print(f'Loading covtype dataset from sklearn...')
    start = time.time()

    data = fetch_covtype(shuffle=False)
    # split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.20, random_state=5, stratify=data.target)
    # split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=5, stratify=y_train)

    # combine X and y in a DataFrame
    df_train = pd.DataFrame(np.c_[X_train, y_train], columns=data.feature_names + ['target'])
    df_val = pd.DataFrame(np.c_[X_val, y_val], columns=data.feature_names + ['target'])
    df_test = pd.DataFrame(np.c_[X_test, y_test], columns=data.feature_names + ['target'])

    # save to csv
    df_train.to_csv(path + 'covtype_train.csv', index=False)
    df_val.to_csv(path + 'covtype_val.csv', index=False)
    df_test.to_csv(path + 'covtype_test.csv', index=False)

    end = time.time()
    print(f'Loading covtype took {end - start} seconds.')


def load_kddcup99(path: str = '../../data/external/kddcup99/'):
    """
    Load kddcup99 dataset and split into train, validation and test sets.
    :param path: path to save data
    """
    print(f'Loading kddcup99 dataset from sklearn...')
    start = time.time()

    data = fetch_kddcup99(shuffle=False)

    # combine X and y in a DataFrame
    df = pd.DataFrame(np.c_[data.data, data.target], columns=data.feature_names + ['target'])
    df['target'] = df['target'].str.decode('ascii')
    df['protocol_type'] = df['protocol_type'].str.decode('ascii')
    df['service'] = df['service'].str.decode('ascii')
    df['flag'] = df['flag'].str.decode('ascii')
    # change column types
    df['duration'] = df['duration'].astype('int')
    df['src_bytes'] = df['src_bytes'].astype('int')
    df['dst_bytes'] = df['dst_bytes'].astype('int')
    df['land'] = df['land'].astype('int')
    df['wrong_fragment'] = df['wrong_fragment'].astype('int')
    df['urgent'] = df['urgent'].astype('int')
    df['hot'] = df['hot'].astype('int')
    df['num_failed_logins'] = df['num_failed_logins'].astype('int')
    df['logged_in'] = df['logged_in'].astype('int')
    df['num_compromised'] = df['num_compromised'].astype('int')
    df['root_shell'] = df['root_shell'].astype('int')
    df['su_attempted'] = df['su_attempted'].astype('int')
    df['num_root'] = df['num_root'].astype('int')
    df['num_file_creations'] = df['num_file_creations'].astype('int')
    df['num_shells'] = df['num_shells'].astype('int')
    df['num_access_files'] = df['num_access_files'].astype('int')
    df['num_outbound_cmds'] = df['num_outbound_cmds'].astype('int')
    df['is_host_login'] = df['is_host_login'].astype('int')
    df['is_guest_login'] = df['is_guest_login'].astype('int')
    df['count'] = df['count'].astype('int')
    df['srv_count'] = df['srv_count'].astype('int')
    df['serror_rate'] = df['serror_rate'].astype('float')
    df['srv_serror_rate'] = df['srv_serror_rate'].astype('float')
    df['rerror_rate'] = df['rerror_rate'].astype('float')
    df['srv_rerror_rate'] = df['srv_rerror_rate'].astype('float')
    df['same_srv_rate'] = df['same_srv_rate'].astype('float')
    df['diff_srv_rate'] = df['diff_srv_rate'].astype('float')
    df['srv_diff_host_rate'] = df['srv_diff_host_rate'].astype('float')
    df['dst_host_count'] = df['dst_host_count'].astype('int')
    df['dst_host_srv_count'] = df['dst_host_srv_count'].astype('int')
    df['dst_host_same_srv_rate'] = df['dst_host_same_srv_rate'].astype('float')
    df['dst_host_diff_srv_rate'] = df['dst_host_diff_srv_rate'].astype('float')
    df['dst_host_same_src_port_rate'] = df['dst_host_same_src_port_rate'].astype('float')
    df['dst_host_srv_diff_host_rate'] = df['dst_host_srv_diff_host_rate'].astype('float')
    df['dst_host_serror_rate'] = df['dst_host_serror_rate'].astype('float')
    df['dst_host_srv_serror_rate'] = df['dst_host_srv_serror_rate'].astype('float')
    df['dst_host_rerror_rate'] = df['dst_host_rerror_rate'].astype('float')
    df['dst_host_srv_rerror_rate'] = df['dst_host_srv_rerror_rate'].astype('float')

    # split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(df.loc[:,df.columns != 'target'], df['target'], test_size=0.20, random_state=5, stratify=df['target'])
    # split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=5, stratify=y_train)

    # combine X and y in a DataFrame
    df_train = pd.DataFrame(np.c_[X_train, y_train], columns=data.feature_names + ['target'])
    df_val = pd.DataFrame(np.c_[X_val, y_val], columns=data.feature_names + ['target'])
    df_test = pd.DataFrame(np.c_[X_test, y_test], columns=data.feature_names + ['target'])

    # save to csv
    df_train.to_csv(path + 'kddcup99_train.csv', index=False)
    df_val.to_csv(path + 'kddcup99_val.csv', index=False)
    df_test.to_csv(path + 'kddcup99_test.csv', index=False)

    end = time.time()
    print(f'Loading kddcup99 took {end - start} seconds.')

def load_glass_identification(path: str = '../../data/external/glass_identification/'):
    """
    Load glass identification dataset from UCI repository.
    :param path: path to save data
    """

    from ucimlrepo import fetch_ucirepo

    start = time.time()

    data = fetch_ucirepo(id=42)

    X = data.data.features
    y = data.data.targets

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=5, stratify=y)
    # split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=5, stratify=y_train)

    # combine X and y in a DataFrame´
    df_train = pd.DataFrame(np.c_[X_train, y_train], columns=data.variables['name'][1:-1].tolist() + ['target'])
    df_val = pd.DataFrame(np.c_[X_val, y_val], columns=data.variables['name'][1:-1].tolist() + ['target'])
    df_test = pd.DataFrame(np.c_[X_test, y_test], columns=data.variables['name'][1:-1].tolist() + ['target'])

    # save to csv
    df_train.to_csv(path + 'glass_identification_train.csv', index=False)
    df_val.to_csv(path + 'glass_identification_val.csv', index=False)
    df_test.to_csv(path + 'glass_identification_test.csv', index=False)

    end = time.time()
    print(f'Loading glass identification took {end - start} seconds.') 

def load_rice(path: str = '../../data/external/rice/'):  
    from ucimlrepo import fetch_ucirepo 
    from sklearn.preprocessing import LabelEncoder

    start = time.time()
  
    # fetch dataset 
    rice_cammeo_and_osmancik = fetch_ucirepo(id=545) 
    
    # data (as pandas dataframes) 
    X = rice_cammeo_and_osmancik.data.features 
    y = rice_cammeo_and_osmancik.data.targets['Class']

    # encode target variable
    le = LabelEncoder()
    y = le.fit_transform(y.ravel())
    
    # split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=5, stratify=y)
    # split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=5, stratify=y_train)
    
    # combine X and y in a DataFrame´
    df_train = pd.DataFrame(np.c_[X_train, y_train], columns=rice_cammeo_and_osmancik.variables['name'][:-1].tolist() + ['target'])
    df_val = pd.DataFrame(np.c_[X_val, y_val], columns=rice_cammeo_and_osmancik.variables['name'][:-1].tolist() + ['target'])
    df_test = pd.DataFrame(np.c_[X_test, y_test], columns=rice_cammeo_and_osmancik.variables['name'][:-1].tolist() + ['target'])

    # save to csv
    df_train.to_csv(path + 'rice_train.csv', index=False)
    df_val.to_csv(path + 'rice_val.csv', index=False)
    df_test.to_csv(path + 'rice_test.csv', index=False)

    end = time.time()
    print(f'Loading rice took {end - start} seconds.')

if __name__ == '__main__':
    load_covtype()
    load_kddcup99()
    load_glass_identification()
    load_rice()