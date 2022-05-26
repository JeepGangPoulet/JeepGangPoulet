from typing import Optional, Tuple, List, Dict
import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from tqdm import tqdm


def dataset_processing(features, targets, testing=False):
    """
        This function merges the features and targets dataset, does the scaling and creates the final two datasets (or one if testing equals True)
        param features: features dataset as a pd.DataFrame
        param targets: targets dataset as a pd.DataFrame
        param testing: True if the dataset is for testing, False by default
        return: DataFrame with all the features (and the target Dataframe)
    """
    data = features.merge(targets, on='Ground Motion ID').drop(columns=['Ground Motion ID', 'Unnamed: 0_x'])
    
    # Multiply dependent columns (Sa(T)s, sa_avg, fiv3) with scaling factor
    cols_to_be_changed = np.append(data.columns[:106].values, data.columns[108])
    data[cols_to_be_changed] = data[cols_to_be_changed].mul(data['scale factor'], axis=0)
    data = data.sort_values(by='Unnamed: 0_y').drop(columns=['scale factor', 'Unnamed: 0_y'])
    
    if testing:
        df_X = data
        return df_X
    else :
        df_X = data.drop(columns=['is_collapse'])
        df_y = data['is_collapse']
        return df_X, df_y
    
    
def check_duplicates(features, targets, dataset_name):
    """
        This function checks if there are duplicated rows in the dataset and if yes, removes them 
        param features: features dataset as a pd.DataFrame
        param dataset_name: name of the dataset
        return: DataFrame without the duplicated rows
    """
    
    print(dataset_name + ':')
    
    idx_duplicates = features[features.duplicated(keep=False)].index
    
    if not idx_duplicates.empty:
        print(f'{len(idx_duplicates)} duplicate rows exist and are removed \n')
        features = features.drop(index=idx_duplicates)
        targets = targets.drop(index=idx_duplicates)
        return features, targets
    
    else:
        print('There are no duplicate rows to remove \n')
        return features, targets
    
    
def check_corrupted(features, targets, dataset_name):
    """
        This function checks if there are corrupted values in the dataset
        param features: features dataset as a pd.DataFrame
        param dataset_name: name of the dataset
        return: DataFrame without the corrupted values
    """
        
    print(dataset_name + ':')
    
    idx_missing = features.loc[(features == -999).any(axis=1)].index
    
    if not idx_missing.empty:
        print(f'{len(idx_missing)} ({len(idx_missing)/features.shape[0]*100:.2f} %) rows have missing values and are removed')
        features_reduced = features.drop(index=idx_missing)
        targets_reduced = targets.drop(index=idx_missing)
        return features_reduced, targets_reduced
    
    else:
        print('There are no -999 rows to remove \n')
        return features, targets
    
    
def check_Nan(features, targets, dataset_name):
    """
        This function checks if there are Nan values in the dataset
        param features: features dataset as a pd.DataFrame
        param dataset_name: name of the dataset
        return: DataFrame without the corrupted values
    """
        
    print(dataset_name + ':')
    
    idx_missing = features.loc[features.isnull().any(axis=1)].index
    
    if not idx_missing.empty:
        print(f'{len(idx_missing)} ({len(idx_missing)/features.shape[0]*100:.2f} %) rows have Nan values and are removed')
        features_reduced = features.drop(index=idx_missing)
        targets_reduced = targets.drop(index=idx_missing)
        return features_reduced, targets_reduced
    
    else:
        print('There are no Nan rows to remove \n')
        return features, targets
    
    
def bool_to_num(df):
    """
        This function converts the boolean columns to numerical
        param features: features dataset as a pd.DataFrame
        return: DataFrame with numerical columns only
    """
    df = df.replace(['NO','YES'],[0,1])
    return df
    
    
def numpy_converter(df):
    """
        This function converts a DataFrame or Series into a numpy array 
        param df: dataframe
        return: numpy array with values of the dataframe
    """
    array = df.to_numpy()
    return array


def create_train_test_dataset_for_regression(
    df: pd.DataFrame,
    train_size: float = 0.8,
    test_size: float = 0.2,
    seed: Optional[int] = None):
    
    """ Transforms the dataset (df) given into dataframes splitted in
        a train and a test set. 
        
        Args:
            df: Data to split
            train_size: fraction of the dataset to use as training set
            test_size: fraction of the dataset to use as test set
            seed: random seed
        Returns:
            Object: Tuple containing a training set as numpy array
            and a test set as numpy array.
    """
    df=df.sample(frac=1, random_state = seed) #returns a fraction (here 100%) of the dataset randomly mixed
    train, test = (
        df[:int(len(df)*train_size)],
        df[int(len(df)*train_size):]
    )
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    return train, test

def hide_values_SaT(
    df: pd.DataFrame
):
    """ Function that replaces some values with np.nan in the dataframe.
        Args: 
            df: dataframe from which to hide 30% of the values
        Returns:
            Object: Tuple containing the original set and the set with randomly hidden
            values.
    """
    df_hide_values = df.copy(deep=True).iloc[:,:105]
    df_original = df.copy(deep=True).iloc[:,:]
    df_missing_values = df.copy(deep=True).iloc[:,:]
    for i in range(0, np.shape(df)[0]):
        id_miss = np.random.randint(0, high=105, size=int(105*.35))
        df_hide_values.iloc[i,id_miss] = np.nan

    df_missing_values.iloc[:,:105] = df_hide_values    

    return df_original, df_missing_values



def hide_values_999(
    df: pd.DataFrame,
    percentage: float 
):
    """ Function that replaces some values with np.nan in the dataframe.
        Args: 
            df: dataframe from which to hide 30% of the values
        Returns:
            Object: Tuple containing the original set and the set with randomly hidden
            values.
    """
    df_hide_values = df.copy(deep=True).iloc[:,109:112]
    df_original = df.copy(deep=True).iloc[:,:]
    df_missing_values = df.copy(deep=True).iloc[:,:]

    id_miss = np.random.randint(0, high=df.shape[0]-1, size=int((df.shape[0]-1)*percentage))
    df_hide_values.iloc[id_miss,:] = np.nan

    df_missing_values.iloc[:,109:112] = df_hide_values    

    return df_original, df_missing_values

def score_estimation(
    original: np.ndarray,
    estimated: np.ndarray
):
    """ Function that computes the score between the original and the estimated
    values dataframes. Estimation done using RMSE.
        Args: 
            orignal: orignal values as np.ndarray
            orignal: estimated values as np.ndarray
        Returns:
            float: score of the estimation
    """
    difference = original - estimated
    #score = np.linalg.norm(difference,ord=1)
    score = np.sqrt(np.mean(difference**2))
    return score


def KNNReg_K_finder(
    training: pd.DataFrame,
    missing: pd.DataFrame,
    original: pd.DataFrame,
    K: int
):
    """ Function that imputes the missing data using a KNN Regressor for
    different values of K. Returns the estimated values as ndarray and
    the score compared to the original datas computed using the 1-norm.
        Args: 
            training: training set
            missing: the set with NaNs
            original: the set without NaNs (used for the score)
        Returns:
            Object: Tuple containing the dataset as numpy ndarray 
            and it's score as 1d-array for 
            different Ks. 
    """
    #K=4
    estimations = np.ndarray((0,np.shape(missing)[0], np.shape(missing)[1]))
    #estimated = np.ndarray((np.shape(missing)[0], np.shape(missing)[1]))
    score = np.ndarray((0))
    for k in tqdm(range(1,K+1)):
        imp=IterativeImputer(estimator = KNeighborsRegressor(n_neighbors=k), max_iter=10, random_state=0) 
        imp.fit(training)
        estimated = imp.transform(missing)
        estimations = np.append(estimations,[estimated],axis=0)
        
        new_score = score_estimation(original.iloc[:,:105].to_numpy(),estimated[:,:105])
        score = np.append(score,new_score)
        
    return estimations, score