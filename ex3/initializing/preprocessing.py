import pandas as pd
import numpy as np

#Imputer
from sklearn.impute import SimpleImputer
#Encoders
from category_encoders.target_encoder import TargetEncoder
from category_encoders.one_hot import OneHotEncoder


def category_numeric_or_string(X, cat_ind):
    ''' Returns the categorical, numerical and string columns of the dataset.

    Parameters:
    -----------
    X: pd.DataFrame
        Contains the dataframe of a given dataset excluding its target column.
    cat_ind: list
        Contains boolean values to determine whether a column is categorical or
        not based on OpenML implementation.

    Returns:
    --------
    list, list, list
        Contains column names of categorical features, contains column names of numerical features,
            contains column names of string features.
    '''
    categorical = []
    numeric = []
    string = []

    for i in range(len(cat_ind)):
        if cat_ind[i] == True:
            categorical.append(X.columns[i])
        elif isinstance(X[X.columns[i]][0], int) | isinstance(X[X.columns[i]][0],
         float):
            numeric.append(X.columns[i])
        else:
            string.append(X.columns[i])

    return categorical, numeric, string


def fill_na_categorical(s):
    ''' Returns a series with a new added category for missing values called
            "missing".

    Parameters:
    -----------
    s: pd.Series
        Contains a series of a specific column of a dataset.

    Returns:
    --------
    pd.Series
        Contains an updated series with the category missing added for imputation.
    '''
    s = s.astype('category')
    s = s.cat.add_categories("missing").fillna("missing")

    return s

def impute(X, y, categorical, numeric, string, strategy):
    ''' Fills missing values of the dataset based on their datatype and a fill
    strategy for numeric features.

    Parameters:
    -----------
    X: pd.DataFrame
        Contains the dataframe of a given dataset excluding its target column.
    y: pd.Series
        Contains the series of the target of a given dataset.
    categorical: list
        Contains the names of the categorical columns.
    numeric: list
        Contains the names of the numerical columns.
    string: list
        Contains the names of the string columns.
    strategy: string
        Contains the imputing strategy for imputing numerical features.

    Returns:
    --------
    pd.DataFrame, pd.Series
        Contains an updated pd.DataFrame after imputation, contains an updated pd.Series after imputation. 
    '''
    for column in categorical:
        X[column] = fill_na_categorical(X[column])
    for column in numeric:
        imp = SimpleImputer(missing_values=np.nan, strategy=strategy)
        X[column] = imp.fit_transform(X[column].values.reshape(-1,1))
    for column in string:
        X[column] = X[column].fillna('missing')

    try:
        if y.dtype == 'category':
            y = fill_na_categorical(y)
    except:
        if isinstance(y[0], np.integer) | isinstance(y[0], np.float):
            imp = SimpleImputer(missing_values=np.nan, strategy=strategy)
            y = imp.fit_transform(y.values.reshape(-1, 1))

    return X, y

def drop_string_column(X, string):
    ''' Returns an adjusted dataframe of a dataset with its string features
    removed.

    Parameters:
    -----------
    X: pd.DataFrame
        Contains the dataframe of a given dataset excluding its target column.
    string: list
        Contains the names of the string columns.

    Returns:
    --------
    pd.DataFrame
        Contains an updated pd.DataFrame with string columns removed. 
    '''
    X = X.drop(columns=string)
    return X

def onehot_or_targ(X, y, categorical, k):
    ''' Returns the X, y with encoded categorical variables based on a threshold
     value k.

    Parameters:
    -----------
    X: pd.DataFrame
        Contains the dataframe of a given dataset excluding its target column.
    y: pd.Series
        Contains the series of the target of a given dataset.
    categorical: list
        Contains the names of the categorical columns.
    k: int
        Contains threshold value to determine whether to perform target encoding
        or one-hot encoding.

    Returns:
    --------
    pd.DataFrame, pd.Series
        Contains an updated pd.DataFrame with encoding of categorical features,
            contains an updated pd.Series with encoding of a categorical target.
    '''
    for column in categorical:
        if len(X[column].unique()) > k:
            if X[column].dtype.name == 'category':
                X[column] = X[column].cat.codes
            if y.dtype.name == 'category':
                y = y.cat.codes
            X = TargetEncoder(cols=[column]).fit_transform(X,y)
        else:
            X = OneHotEncoder(cols=[column]).fit_transform(X)
    return X, y
