import openml as oml
from gama import GamaClassifier, GamaRegressor
import os

from initializing.preprocessing import category_numeric_or_string, impute, onehot_or_targ

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
        Contains dataset ids that have been ran during the optimization process.
    '''
    csv_listed = os.listdir(path)
    dataset_ids = []

    for csv in csv_listed:
        for ch in ['.log', 'a', 'b', 'c', 'd', 'e']:
            if ch in csv:
                csv = csv.replace('.log', '')
        try:
            csv = int(csv)
            dataset_ids.append(csv)
        except:
            pass

    return dataset_ids

def gama_runs(datasets, path, task):
    ''' Executes Gama optimization for different OpenML datasets and stores the
    log files in a specified path.

    Parameters:
    -----------
    datasets: list
        Contains datasets that are going to be optimized using Gama.
    path: string
        Contains the path to the directory in where the files are logged.
    task: string
        Contains learning task to specify the GAMA optimization (either classi-
        fication or regression).

    Returns:
    --------
    string
        Contains a confirmation that the optimization process has finished.
    '''
    executed = executed_datasets(path)
    for dataset_id in datasets:
        if dataset_id not in executed:
            try:
                ds = oml.datasets.get_dataset(dataset_id, download_data=False)
                X, y, categorical_indicator, attribute_names = ds.get_data(
                    dataset_format='DataFrame',
                    target=ds.default_target_attribute
                )

                categorical, numeric, string = category_numeric_or_string(X, categorical_indicator)
                X, y = impute(X, y, categorical, numeric, string, "median")

                for k in [1, 2, 5, 10, 25]:
                    log_k = ''
                    if k == 1:
                        log_k = 'a'
                    elif k == 2:
                        log_k = 'b'
                    elif k == 5:
                        log_k = 'c'
                    elif k == 10:
                        log_k = 'd'
                    else:
                        log_k = 'e'

                    X_adj, y_adj = onehot_or_targ(X, y, categorical, k)
                    if task.lower() == "classification":
                        gama = GamaClassifier(n_jobs=-1, max_total_time=600,
                                                scoring='accuracy',
                                                keep_analysis_log='{}{}{}.log'.format(path, log_k, dataset_id))
                    elif task.lower() == "regression":
                        gama = GamaRegressor(n_jobs=-1, max_total_time=600,
                                                scoring='r2',
                                                keep_analysis_log='{}{}{}.log'.format(path, log_k, dataset_id))
                    else:
                        return "Please select classification or regression as learning task!"
                    gama.fit(X_adj, y_adj)
            except:
                pass

    return "Gama has finished running optimization."

#gama_runs([5], "\\", "classification")
