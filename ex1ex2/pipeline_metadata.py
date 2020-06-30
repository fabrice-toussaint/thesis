from configuration.config import clf_config, rgr_config
import itertools
import os
from sklearn.model_selection import train_test_split
import time
import timeout_decorator
import os
import arff
import multiprocessing
import pickle
import openml as oml
import pandas as pd
from initializing.preprocessing import category_numeric_or_string, fill_na_categorical, impute, drop_string_column, onehot_or_targ
from copy import deepcopy

from evolutionary_search import maximize, optimize

#Classification algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

#Regression algorithms
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

def dict_combinations(d):
    ''' Returns the different configuration combinations as listed object.

    Parameters:
    -----------
    d: dict
        Contains the configuration file in dictionary form.

    Returns:
    --------
    list
        Contains the configuration file in listed form.
    '''
    algorithms = []

    for preprocessor in d.keys():
        keys = d[preprocessor].keys()
        values = (d[preprocessor][key] for key in keys)
        combinations = [dict(zip(keys, combination)) for combination in itertools.product(*values)]

        for combination in combinations:
            if not preprocessor == None:
                prep = preprocessor()
            for i in range(len(combination.keys())):
                comb = list(combination.keys())[i]
                prep.__dict__[comb] = combination[comb]
            algorithms.append(prep)
    return algorithms

#@timeout_decorator.timeout(180, use_signals=True)
def pipeline_run(X, y, dataset_id, alg, cat_ind, k, categorical, numeric, string, path, metadata, p1=None, p2=None, p3=None):
    ''' Runs the pipeline on dataset X, returns the score and logs the pipeline.

    Parameters:
    -----------
    X: pd.DataFrame
        Contains the dataframe of a given dataset excluding its target column.
    y: pd.Series
        Contains the series of the target of a given dataset.
    dataset_id: integer
        Contains the id of the dataset to be optimized.
    alg: scikit-learn predictors
        Contains the predictive algorithm of the pipeline.
    cat_ind: list
        Contains boolean values to determine whether a column is categorical or
        not based on OpenML implementation.
    k: integer
        Contains the indicator whether to perform one-hot encoding or target
        encoding. For example, k=5 and column X contains 10 categorical values,
        since 10 > 5, column X will be target encoded instead of one-hot encoded.
    categorical: list
        Contains the names of the categorical columns.
    numeric: list
        Contains the names of the numerical columns.
    string: list
        Contains the names of the string columns.
    path: string
        Contains the path to the directory in where the files are logged.
    metadata: pd.DataFrame
        Contains the dataframe of the metadata for dataset with id X.
    p1: scikit-learn preprocessor
        Contains the first preprocessor in the pipeline.
    p2: scikit-learn preprocessor
        Contains the second preprocessor in the pipeline.
    p3: scikit-learn preprocessor
        Contains the last preprocessor in the pipeline.

    Returns:
    --------
    float
        Contains the score of the pipeline that is ran.
    '''
    start = time.process_time()

    X = deepcopy(X)
    y = deepcopy(y)

    p1 = deepcopy(p1)
    p2 = deepcopy(p2)
    p3 = deepcopy(p3)

    alg = deepcopy(alg)

    X, y = onehot_or_targ(X, y, categorical, k)

    X = drop_string_column(X, string)

    X = X.to_numpy()
    y = y.to_numpy()

    for preprocessor in [p1, p2, p3]:
        try:
            if preprocessor is not None:
                if isinstance(preprocessor, SelectPercentile) | isinstance(preprocessor, SelectFwe):
                    X = preprocessor.fit_transform(X, y)
                else:
                    X = preprocessor.fit_transform(X)

                if isinstance(preprocessor, PolynomialFeatures) and len(X[0]) > 5:
                    preprocessor = None
        except:
            preprocessor = None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.33, random_state=dataset_id)
        alg.fit(X_train,y_train)
        score = alg.score(X_test,y_test)
    except ValueError:
        score = 0

    end_time = time.process_time() - start

    params = {'id': dataset_id,
            'algorithm': alg,
            'k': k,
            'processor_1': p1,
            'processor_2': p2,
            'processor_3': p3,
            'score': score,
            'time': end_time}

    metadata = metadata.append(params, ignore_index=True)
    metadata.to_csv(r'{}{}.csv'.format(path, dataset_id), mode='a', index=False, header=False, sep=';')

    return score

#@timeout_decorator.timeout(600, use_signals=True)
def meta_data(X, y, dataset_id, algorithms, cat_ind, preprocessors, path, metadata):
    ''' Runs the pipeline on dataset X, returns the score and logs the pipeline.

    Parameters:
    -----------
    X: pd.DataFrame
        Contains the dataframe of a given dataset excluding its target column.
    y: pd.Series
        Contains the series of the target of a given dataset.
    dataset_id: integer
        Contains the id of the dataset to be optimized.
    algorithms: list
        Contains the search space of predictive algorithms.
    cat_ind: list
        Contains boolean values to determine whether a column is categorical or
        not based on OpenML implementation.
    preprocessors: list
        Contains the searchspace of preprocessing algorithms.
    path: string
        Contains the path to the directory in where the files are logged.
    metadata: pd.DataFrame
        Contains the dataframe of the metadata for dataset with id X.

    Returns:
    --------
    str
        Contains confirmation that the optimization has been ran for dataset with dataset_id.
    '''
    categorical, numeric, string = category_numeric_or_string(X, cat_ind)
    X, y = impute(X, y, categorical, numeric, string, 'mean')

    if len(preprocessors) > 0:

        param_grid = {'p1': preprocessors,
                      'p2': preprocessors,
                      'p3': preprocessors,
                      'alg': algorithms,
                      'k': [1, 2, 5, 10, 25]}

        args = {'X': X,
               'y': y,
               'cat_ind': cat_ind,
               'dataset_id': dataset_id,
               'categorical': categorical,
               'numeric': numeric,
               'string': string,
               'path': path,
               'metadata': metadata}

        best_params, best_score, score_results, _, _ = maximize(pipeline_run, param_grid, args,
                                                            verbose=False, generations_number=10,
                                                           population_size=100,
                                                                tournament_size=6,
                                                               n_jobs=1,
                                                               gene_mutation_prob=0.50,
                                                               gene_crossover_prob=0.90)

    else:
        for algorithm in algorithms:
            for k in [1, 2, 5, 10, 25]:
                pipeline_run(X, y, dataset_id, algorithm, cat_ind, k,
                             categorical, numeric, string, path, metadata, p1=None, p2=None, p3=None)

    return "Dataset {} has finished running.".format(dataset_id)

def executed_datasets(path):
    ''' Returns the dataset ids of the datasets that have been ran during
    optimization.

    Parameters:
    -----------
    path: string
        Contains the path to the directory in where the files are logged.

    Returns:
    --------
    list
        Contains a list with the dataset ids that were ran and stored in the path.
    '''
    csv_listed = os.listdir(path)
    dataset_ids = []

    for csv in csv_listed:
        csv = csv.replace('.csv', '')
        try:
            csv = int(csv)
            dataset_ids.append(csv)
        except:
            pass

    return dataset_ids

def get_oml_dataset(did):
    ''' Retrieves the OpenML dataset with dataset id from the OpenML database.

    Parameters:
    -----------
    did: int
        Contains the dataset id of the OpenML dataset.

    Returns:
    --------
    pd.Dataframe, pd.Series, list
        Contains a pd.Dataframe with the features of the dataset, a pd.Series with the target of the dataset,
            list with boolean values to determine whether a column is categorical or not.

    '''
    ds = oml.datasets.get_dataset(did, download_data=False)
    X, y, cat_ind, attribute_names = ds.get_data(
    dataset_format='DataFrame',
    target=ds.default_target_attribute
    )
    return X, y, cat_ind

def all_runs(datasets, path, pred_config, proc_config):
    ''' Runs the optimization process on all datasets.

    Parameters:
    -----------
    datasets: list
        Contains an overview of all datasets that are put up for optimization.
    path: string
        Contains the path to the directory in where the files are logged.
    predictors: list
        Contains the search space of predictive algorithms.
    config: dict
        Contains the search space of preprocessing algorithms.

    Returns:
    --------
    string
        Contains a confirmation that the meta-data has been collected for every dataset in datasets.

    '''
    executed = executed_datasets(path)

    for dataset_id in datasets:
        if dataset_id not in executed:
            try:
                
                X, y, cat_ind = get_oml_dataset(dataset_id)
                
                predictors = dict_combinations(pred_config)
                preprocessors = dict_combinations(proc_config)

                metadata = pd.DataFrame(columns=['algorithm', 'id', 'k', 'processor_1', 'processor_2', 'processor_3', 'score', 'time'])
                metadata.to_csv(r'{}{}.csv'.format(path, dataset_id), mode='a', index=False, sep=';')

                optimize.compile()
                meta_data(X, y, dataset_id, predictors, cat_ind, preprocessors, path, metadata)
            except (timeout_decorator.timeout_decorator.TimeoutError, OSError, MemoryError,
                    NotImplementedError, AttributeError, arff.BadAttributeType,
                    multiprocessing.pool.MaybeEncodingError, oml.exceptions.OpenMLHashException):
                continue
        else:
            continue

    return "All meta-data collected."

pred_config = {
    RandomForestClassifier: {},
}

#Run a blank preprocessor configuration file for baselines as below
#blank_config = {}

all_runs([5, 7], "\\", pred_config, clf_config)
