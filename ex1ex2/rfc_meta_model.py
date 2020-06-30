import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

def better_than_baseline(x):
    ''' Checks whether pipeline is better than a baseline.

    Parameters:
    -----------
    x: float
        Contains a float that denotes the difference in scoring/time of a pipeline and baseline.

    Returns:
    --------
    int
        Contains an integer where 1 indicates that the value is better or equal and 0 worse.
    '''
    if x >= 0:
        return 1
    else:
        return 0

def baseline_comparison(df, base):
    ''' Merges meta-data with baseline data, extracts how each pipeline scores with respect to baseline,
            in terms of scoring and runnig time. Then duplicates are dropped from the merged pd.DataFrame,
            based on the combination of id, k (encoding), algorithm, preprocessors.

    Parameters:
    -----------
    df: pd.DataFrame
        Contains a pd.DataFrame with all pipelines runs for the different datasets.
    base: pd.DataFrame
        Contains a pd.DataFrame with all baseline runs for the different datasets.

    Returns:
    --------
    pd.DataFrame
        Contains an updated pd.DataFrame with scoring difference and time difference and no duplicates.
    '''
    df = pd.merge(df, base, how='left', left_on=['id', 'k'], right_on=['id', 'k'])
    df['score_diff'] = df['score_x'] - df['score_y']
    df['score_diff'] = df['score_diff'].apply(lambda x: better_than_baseline(x))
    df['time_diff'] = df['time_y'] - df['time_x']
    df['time_diff'] = df['time_diff'].apply(lambda x: better_than_baseline(x))
    df = df.drop_duplicates(subset=['algorithm', 'id', 'k', 'processor_1', 'processor_2', 'processor_1'])
    return df

def merge_md_with_mf(md, mf):
    ''' Merges meta-data with meta-features based on dataset id.

    Parameters:
    -----------
    md: pd.DataFrame
        Contains a pd.DataFrame with the meta-data containing all pipelines runs for the different datasets.
    mf: pd.DataFrame
        Contains a pd.DataFrame with all meta-features for the different datasets.

    Returns:
    --------
    pd.DataFrame
        Contains an updated pd.DataFrame where meta-features are merged with the meta-data.
    '''
    return pd.merge(md, mf, left_on='id', right_index=True).reset_index(drop=True)

def max_score(md):
    ''' Extracs the best scoring pipelines for each dataset id from the meta-data.

    Parameters:
    -----------
    md: pd.DataFrame
        Contains a pd.DataFrame with the meta-data containing all pipelines runs for the different datasets.

    Returns:
    --------
    pd.DataFrame
        Contains a pd.DataFrame subset with best scoring pipelines.
    '''
    idx = md.groupby(['id'])['score_x'].transform(max) == md['score_x']
    return md[idx]

def max_time(md):
    ''' Extracs the fastest pipelines for each dataset id from the meta-data.

    Parameters:
    -----------
    md: pd.DataFrame
        Contains a pd.DataFrame with the meta-data containing all pipelines runs for the different datasets.

    Returns:
    --------
    pd.DataFrame
        Contains a pd.DataFrame subset with fastest pipelines.
    '''
    idx = md.groupby(['id'])['time_x'].transform(min) == md['time_x']
    return md[idx]


def nan_to_none(x):
    ''' Checks whether an object is equal to NaN and converts it to a string called "None"

    Parameters:
    -----------
    x: str
        Contains a string representative of a scikit-learn preprocessor or NaN when missing.

    Returns:
    --------
    str
        Contains the same string or "None" when x is equal to NaN.
    '''
    try:
        if np.isnan(x):
            return "None"
    except:
        return x


def evaluate(x):
    ''' Evaluate the string to convert it back to a scikit-learn preprocessor object.

    Parameters:
    -----------
    x: str
        Contains a string representative of a scikit-learn preprocessor or "None" when missing.

    Returns:
    --------
    scikit-learn algorithm or str
        Contains a scikit-learn preprocessor object or a string.
    '''
    try:
        return eval(x)
    except (SyntaxError, TypeError):
        pass
    except NameError:
        return "Remove"


def setattribute(x, n):
    ''' Sets the attribute of the random_state to a fixed number, for generalization purposes, when
            creating categories.

    Parameters:
    -----------
    x: scikit-learn preprocessor
        Contains a scikit-learn preprocessor object or "None" when missing.
    n: int
        Contains a integer to replace the random_state in the object.

    Returns:
    --------
    scikit-learn algorithm
        Contains a scikit-learn preprocessor object.
    '''
    try:
        return setattr(x, 'random_state', n)
    except AttributeError:
        pass


def to_string(x):
    ''' Converts the scikit-learn preprocesor object back to a string, to make generating categories
            available.

    Parameters:
    -----------
    x: scikit-learn preprocessor
        Contains a scikit-learn preprocessor object or "None" when missing.

    Returns:
    --------
    str
        Contains a string of the scikit-learn preprocessor.
    '''
    try:
        return str(x)
    except:
        pass
    
def prepare_metadata(md, target):
    ''' Prepares the meta-data and converts NaN values to None, makes objects of the scikit-learn preprocessors,
            sets a fixed number for random_state of 420, converts this object back to a string and applies categorical
            encoding of the categorical features.

    Parameters:
    -----------
    md: pd.DataFrame
        Contains the pd.DataFrame of the features of the meta-data.
    target: str
        Contains the name of the target in the meta-data (i.e. scoring or running time).

    Returns:
    --------
    md
        Contains an updated pd.DataFrame of the metadata.
    '''    
    columns = ['processor_1', 'processor_2', 'processor_3']
    
    for column in columns:
        md[column] = md[column].apply(lambda x: nan_to_none(x))
        md[column] = md[column].apply(lambda x: evaluate(x))
        md[column].apply(lambda x: setattribute(x, 420))
        md[column] = md[column].apply(lambda x: to_string(x))
        md[column] = md[column].astype('category')
        enc = TargetEncoder(cols=[column])
        md['{}_encoded'.format(column)] = enc.fit_transform(md[column],md[target])
        
    return md

def processor_encoded_pair(df, columns):
    ''' Combines the encoded values with the names of the object, so that when using the meta-model,
            one can easily match the input object and receive the encoded value.

    Parameters:
    -----------
    df: pd.DataFrame
        Contains the pd.DataFrame of the features of the meta-data.
    columns: list
        Contains a list with the names of the preprocessors that are necessary for encoding.

    Returns:
    --------
    dict
        Contains a dictionary with keys for the preprocessor, which contain values with keys for the
            preprocessing algorithms, which contain the encoded values. For example,
                dict[preprocessor_1][preprocessing_algorithm] -> encoded value
    '''  
    pe_pairs = {}
    for column in columns:
        pairs = df[[column, '{}_encoded'.format(column)]].drop_duplicates()
        pairs_dict = pd.Series(pairs['{}_encoded'.format(column)].values,index=pairs[column]).to_dict()
        pe_pairs[column] = pairs_dict
    
    return pe_pairs

def prepare_meta_model(df):
    ''' Removes several columns from the meta-data and drops faulty infinite values from the meta-data.
            Then splits the dataset into features (X) and targets (scoring and running time).

    Parameters:
    -----------
    df: pd.DataFrame
        Contains the pd.DataFrame of the features of the meta-data.

    Returns:
    --------
    pd.DataFrame, pd.DataFrame or pd.Series
        Contains the features of the meta-data excluding the target, contains either a pd.DataFrame with both
            scoring and running time or a pd.Series with one of either.
    ''' 
    delete_columns = ['id', 'algorithm', 'processor_1', 'processor_2',
    'processor_3', 'score_x', 'time_x', 'score_y', 'time_y', 'length']
    
    target_columns = ['score_diff', 'time_diff']
    
    df = df.drop(delete_columns, axis=1)
    df['k'] = df['k'].astype('float')
    df.reset_index(inplace=True, drop=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(how="any")
    
    y = df[target_columns]
    X = df.drop(target_columns, axis=1)
    
    return X, y

def build_and_score_meta_model(md, target):
    ''' Prepares the meta-data, then splits this into a train- and test set, creates a RandomForestClassifier
            and fits and scores this estimator on the data. Both the estimator and the scoring of the estimator
            are returned.

    Parameters:
    -----------
    md: pd.DataFrame
        Contains the pd.DataFrame of the features of the meta-data.
    target: str
        Contains the name of the target in the meta-data (i.e. scoring or running time).

    Returns:
    --------
    RandomForestClassifier, float
        Contains the RandomForestClassifier fitted to the meta-data, contains the scoring of this RandomForestClassifier.
    ''' 
    X, y = prepare_meta_model(md)
    X_train, X_test, y_train, y_test = train_test_split(X, y[target], test_size=0.3, random_state=420)
    est = RandomForestClassifier(oob_score=True)
    est = est.fit(X_train,y_train)
    score = est.score(X_test, y_test)

    return est, score

if __name__ == "__main__":
    mclr = pd.read_csv('../data/ex1ex2/mdll_c.csv', sep=';')
    mrlr = pd.read_csv('../data/ex1ex2/mdll_r.csv', sep=';')
    mcrf = pd.read_csv('../data/ex1ex2/mdrf_c.csv', sep=';')
    mrrf = pd.read_csv('../data/ex1ex2/mdrf_r.csv', sep=';')

    pclr = processor_encoded_pair(mclr, ['processor_1', 'processor_2', 'processor_3'])
    prlr = processor_encoded_pair(mrlr, ['processor_1', 'processor_2', 'processor_3'])
    pcrf = processor_encoded_pair(mcrf, ['processor_1', 'processor_2', 'processor_3'])
    prrf = processor_encoded_pair(mrrf, ['processor_1', 'processor_2', 'processor_3'])

    est_lrc, score_lrc = build_and_score_meta_model(mclr, 'score_diff')
    est_lrr, score_lrr = build_and_score_meta_model(mrlr, 'score_diff')
    est_rfc, score_rfc = build_and_score_meta_model(mcrf, 'score_diff')
    est_rfr, score_rfr = build_and_score_meta_model(mrrf, 'score_diff')

    with open('../data/ex1ex2/rfcs.pkl', 'wb') as f:
        pickle.dump([est_lrc, est_lrr, est_rfc, est_rfr], f)

    with open('../data/ex1ex2/pcs.pkl', 'wb') as f:
        pickle.dump([pclr, prlr, pcrf, prrf], f)

