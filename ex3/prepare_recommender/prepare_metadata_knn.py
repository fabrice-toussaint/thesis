import pandas as pd
import numpy as np
from copy import deepcopy
from category_encoders import TargetEncoder

def nan_to_none(x):
    ''' If a value is equal to NaN, it is converted to "None" else the value is returned.

    Parameters:
    -----------
    x: scikit-learn preprocessor
        Contains a scikit-learn preprocessor.

    Returns:
    --------
    str or scikit-learn preprocessor
        Contains either a string with "None" or the output is equal to the input.
    '''
    try:
        if np.isnan(x):
            return "None"
    except:
        return x
    
def prepare_df(df, columns, target):
    ''' Prepares a pd.DataFrame by turning missing scikit-learn preprocessors into "None" strings and
            performs target encoding at the input columns.

    Parameters:
    -----------
    df: pd.DataFrame
        Contains a pd.DataFrame with the generated meta-data.
    columns: list
        Contains a list with the columns that contain scikit-learn estimators and scikit-learn preprocessors.
    target: str
        Contains a string that represents the name of the column that is the target of the dataset.
    Returns:
    --------
    pd.DataFrame
        Contains adjusted pd.DataFrame.
    '''
    df = deepcopy(df)
    df = df.reset_index(drop=True)
    df = df.drop_duplicates()
    y = df[target]
    
    for column in ['component_1', 'component_2', 'component_3']:
        df[column] = df[column].apply(lambda x: nan_to_none(x))
    
    for column in columns:
        df[column] = df[column].astype('category')
        df['{}_codes'.format(column)] = df[column].cat.codes
        enc = TargetEncoder(cols=[column])
        df['{}_encoded'.format(column)] = enc.fit_transform(df[column],y)
        
    return df

def prepare_metadata(df, task):
    ''' Prepares a pd.DataFrame by adjusting the prepare_df function on it and setting the target values,
            dropping duplicates and adjusting the pipeline length to integers.

    Parameters:
    -----------
    df: pd.DataFrame
        Contains a pd.DataFrame with the generated meta-data.
    task: str
        Contains the name of the learning task (i.e. "classification" or "regression").
    Returns:
    --------
    pd.DataFrame
        Contains adjusted pd.DataFrame.
    '''
    if task.lower() == "regression":
        df = prepare_df(df, ['algorithm', 'component_1', 'component_2', 'component_3'], 'r2')
    else:
        df = prepare_df(df, ['algorithm', 'component_1', 'component_2', 'component_3'], 'accuracy')    
    df['length'] = -df['length']
    df = df.drop_duplicates(subset=['id', 'k', 'algorithm', 'component_1', 'component_2', 'component_3'])
    df['length'] = df['length'].astype(int)
    df = df[df['length'].isin([1,2,3,4])]
    return df

def best_scoring_knn(df, n, scoring):
    ''' Calculates a subset of a pd.DataFrame (meta-data) based on the n amount of best scoring pipelines
            available in the dataset.

    Parameters:
    -----------
    df: pd.DataFrame
        Contains a pd.DataFrame with the generated meta-data.
    n: int
        Contains an integer which indicates how many best performing pipelines to include when subsetting.
    scoring: str
        Contains a string that represents the name of the scoring column.
    Returns:
    --------
    pd.DataFrame
        Contains the subset of the input pd.DataFrame based on the best scoring pipelines.
    '''
    best_scoring = df.groupby('id')[scoring].nlargest(n).reset_index()
    best_scoring = df[df.index.isin(best_scoring['level_1'])]
    best_scoring = best_scoring.replace([np.inf, -np.inf], np.nan)
    best_scoring = best_scoring.dropna(how='any')
    
    return best_scoring

if __name__ == "__main__":
    mfc = pd.read_csv('../../data/ex3/all_mfc.csv', sep=';', index_col=[0])
    mfr = pd.read_csv('../../data/ex3/all_mfr.csv', sep=';', index_col=[0])
    
    mdc = pd.read_csv('../../data/ex3/metadata_gama_C.csv', sep=',')
    mdr = pd.read_csv('../../data/ex3/metadata_gama_R.csv', sep=',')
    
    mdc = prepare_metadata(mdc, 'classification')
    mdr = prepare_metadata(mdr, 'regression')
    
    mfc = mfc[mfc.index.isin(mdc['id'])]
    mfr = mfr[mfr.index.isin(mdr['id'])]

    #print(mdc.head(1), mfc.head(1), mdr.head(1), mfr.head(1))

    merged_mdc = pd.merge(mdc, mfc, left_on='id', right_index=True).reset_index(drop=True)
    merged_mdr = pd.merge(mdr, mfr, left_on='id', right_index=True).reset_index(drop=True)

    best_scoring_knn(merged_mdc, 50, 'accuracy').to_csv('../../data/ex3/best_C.csv', sep=';')
    best_scoring_knn(merged_mdr, 50, 'r2').to_csv('../../data/ex3/best_R.csv', sep=';')

    print(merged_mdc.columns)

    
