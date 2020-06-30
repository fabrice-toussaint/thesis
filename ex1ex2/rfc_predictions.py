import pickle
import pandas as pd
from rfc_meta_model import prepare_meta_model
from sklearn.cluster import FeatureAgglomeration
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    PolynomialFeatures,
    RobustScaler,
    StandardScaler,
    Binarizer,
)
from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.decomposition import PCA, FastICA
from sklearn.feature_selection import (
    SelectFwe,
    SelectPercentile,
    VarianceThreshold,
    f_regression,
    f_classif,
)

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from pipeline_metadata import get_oml_dataset
from metafeatures.metafeatures import meta_features
from initializing.preprocessing import impute, category_numeric_or_string

import numpy as np

def mf_values(fixed_mfs, mfs):
    ''' Compares the fixed meta-features with the total meta-features and keeps the values of
            the total meta-features that are available in the fixed meta-features set based
            on their meta-feature name.

    Parameters:
    -----------
    fixed_mfs: list
        Contains the names of the meta-features that are fixed.
    mfs: dict
        Contains the meta-features retrieved from a dataset using Pymfe.

    Returns:
    --------
    list
        Contains the values of the meta-features.
    '''
    values = []
    for mf in fixed_mfs:
        if mf in mfs[0]:
            values.append(mfs[1][mfs[0].index(mf)])
        else:
            values.append(0)
    return pd.Series(values).fillna(0).tolist()

def predict_baseline(X, y, k, cat_ind, task, model, estimators, pairs, fixed_mfs, p1, p2=None, p3=None):
    ''' Predicts for a given dataset whether a given pipeline configuration performs better or equal to
            a baseline which is either LinearRegression or RandomForestRegression for regression and
            LogisticRegression of RandomForestClassification for classification.

    Parameters:
    -----------
    X: pd.DataFrame
        Contains features of the dataset in a pd.DataFrame format.
    y: pd.Series
        Contains target of the dataset in a pd.Series format.
    k: int
        Contains value for target encoding of the categorical features.
    cat_ind: list
        Contains a list containing True or False, indicating whether a column is categorical.
    task: str
        Contains the name of the task (i.e. "regression" or "classification"), to determine the learning type.
    model: scikit-learn estimator
        Contains the baseline model (i.e. LinearRegression, RandomForestRegressor, LogisticRegression or
            RandomForestClassifier)
    estimators: list
        Contains the pretrained RandomForestClassifiers for each specific task and baseline.
    pairs: dict
        Contains the encoded - preprocessor pairs.
    fixed_mfs: list
        Contains the names of the meta-features that are fixed.
    p1: scikit-learn preprocessor
        Contains the scikit-learn preprocessor at the beginning of the pipeline.
    p2: scikit-learn preprocessor (Optional)
        Contains the scikit-learn preprocessor in the middle of the pipeline.
    p3: scikit-learn preprocessor (Optional)
        Contains the scikit-learn preprocessor at the end of the pipeline.

    Returns:
    --------
    int
        Contains a 1 when it is predicted that a configuration improves over the baseline 
            and a 0 when this is not the case.
    '''
    for preprocessor in [p1, p2, p3]:
        if str(preprocessor) not in pairs[0]['processor_1'].keys():
            return "The preprocessor {} is not available in the metadata.".format(preprocessor)

    cat, num, s = category_numeric_or_string(X, cat_ind)
    X, y = impute(X, y, cat, num, s, 'median')
    mfs = meta_features(X, y)
    values = mf_values(fixed_mfs, mfs)
    values.insert(0, k)

    for processor in [p1, p2, p3]:
        try:
            setattr(processor, 'random_state', 420)
        except:
            pass

    p1 = str(p1)
    p2 = str(p2)
    p3 = str(p3)
    
    if task.lower() == "regression":
        if isinstance(model, LinearRegression):
            est = estimators[1]
            values.append(pairs[1]['processor_1'][p1])
            values.append(pairs[1]['processor_2'][p2])
            values.append(pairs[1]['processor_3'][p3])
        elif isinstance(model, RandomForestRegressor):
            est = estimators[3]
            values.append(pairs[3]['processor_1'][p1])
            values.append(pairs[3]['processor_2'][p2])
            values.append(pairs[3]['processor_3'][p3])
        else:
            return "{} is not implemented yet!".format(model)

    elif task.lower() == "classification":
        if isinstance(model, LogisticRegression):
            est = estimators[0]
            values.append(pairs[0]['processor_1'][p1])
            values.append(pairs[0]['processor_2'][p2])
            values.append(pairs[0]['processor_3'][p3])
        elif isinstance(model, RandomForestClassifier):
            est = estimators[2]
            values.append(pairs[2]['processor_1'][p1])
            values.append(pairs[2]['processor_2'][p2])
            values.append(pairs[2]['processor_3'][p3])
        else:
            return "{} is not implemented yet!".format(model)
    else:
        return "{} is not implemented yet!".format(task)

    if est.predict(np.array(values).reshape(1,-1)) == 1:
        return 1 #"The RFC predicts that your current pipeline will improve over using {}.".format(type(model).__name__)
    else:
        return 0 #"The RFC predicts that your current pipeline will not improve over using {}.".format(type(model).__name__)


if __name__ == "__main__":
    with open('../data/ex1ex2/rfcs.pkl', 'rb') as f:
        estimators = pickle.load(f)

    with open('../data/ex1ex2/pcs.pkl', 'rb') as f:
        pairs = pickle.load(f)

    with open('../data/ex1ex2/clf_mf.pickle', 'rb') as f:
        clf_mf = pickle.load(f)

    with open('../data/ex1ex2/rgr_mf.pickle', 'rb') as f:
        rgr_mf = pickle.load(f)

    X, y, cat_ind = get_oml_dataset(8)
    print(predict_baseline(X, y, 1, cat_ind, "regression", LinearRegression(), estimators, pairs, rgr_mf, Binarizer(), Binarizer(), Binarizer()))