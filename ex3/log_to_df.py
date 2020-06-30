import pandas as pd
from initializing.oml_datasets import arff_and_first_version, get_dataset_ids
import os
from gama import GamaClassifier, GamaRegressor
from gama.genetic_programming.components import Individual
import glob
from gama.logging.GamaReport import GamaReport
import numpy as np

def mixed_string_to_int(file):
    ''' Extracts integers only from a string (to detect dataset ID).

    Parameters:
    -----------
    file: string
        Contains the file name (example: "a3.log").

    Returns:
    --------
    int
        Contains an integer for the dataset id linked to the file. 
    '''
    combined_string = list(file)
    int_to_string = []
    for char in combined_string:
        try:
            if isinstance(int(char), int):
                int_to_string.append(char)
        except ValueError:
            pass

    int_string = "".join(int_to_string)
    return int(int_string)

def return_k(file):
    ''' Returns the integer code for the value k.

    Parameters:
    -----------
    file: string
        Contains the file name (example: "a3.log").

    Returns:
    --------
    int
        Contains the integer for the value k linked to its respective letter. 
    '''
    if "a" in file:
        return 1
    elif "b" in file:
        return 2
    elif "c" in file:
        return 5
    elif "d" in file:
        return 10
    elif "e" in file:
        return 25
    else:
        return 0

def log_to_df(path, classification, regression):
    ''' Converts multiple log files to dataframes depending on their learning
            task.

    Parameters:
    -----------
    path: string
        Contains the path name to where the log files are stored.
    classification: list
        Contains an overview of classification learning task dataset ids.
    regression: list
        Contains an overview of regression learning task dataset ids.

    Returns:
    --------
    pd.DataFrame, pd.DataFrame
        Contains a pd.DataFrame of classification datasets where multiple pd.DataFrame have been combined into one,
            Contains a pd.DataFrame of regression datasets where multiple pd.DataFrame have been combined into one

    '''
    df_class = pd.DataFrame()
    df_regr = pd.DataFrame()

    for log_file in glob.glob(path):

        #gives file name of current log_file
        file_name = os.path.basename(log_file)
        dataset_id = mixed_string_to_int(file_name)

        #create df from GAMA report
        report = GamaReport(logfile=log_file)
        report_df = report.evaluations

        #Set k in df
        report_df['k'] = return_k(file_name)
        report_df['id'] = dataset_id

        report_df = report_df.replace([np.inf, -np.inf], np.nan).dropna()

        if dataset_id in classification:
            drop_columns = ['start', 'accuracy_cummax', 'length_cummax', 'relative_end']
            report_df = report_df.drop(drop_columns, axis=1)
            df_class = df_class.append(report_df)
        else:
            drop_columns = ['start', 'r2_cummax', 'length_cummax', 'relative_end']
            report_df = report_df.drop(drop_columns, axis=1)
            df_regr = df_regr.append(report_df)

    return df_class, df_regr

def pipeline_to_children(pipeline, automl):
    ''' Converts pipeline format of Gama log file to individual scikit-learn
            components.

    Parameters:
    -----------
    pipeline: pipeline set
        Contains pipeline in Gama format.
    automl: GamaClassifier or GamaRegressor
        Contains either a GamaClassifier object or GamaRegressor object.

    Returns:
    --------
    scikit-learn predictor, scikit-learn preprocessor (Optional), scikit-learn preprocessor (Optional), scikit-learn preprocessor (Optional)
        Contains the GAMA individuals converted to the respective pipeline scikit-learn components.
    '''
    ind = Individual.from_string(pipeline, automl._pset)
    inds = [p.str_nonrecursive for p in ind.primitives]
    if len(inds) == 1:
        return inds[0], np.nan, np.nan, np.nan
    elif len(inds) == 2:
        return inds[0], inds[1], np.nan, np.nan
    elif len(inds) == 3:
        return inds[0], inds[2], inds[1], np.nan
    else:
        return inds[0], inds[3], inds[2], inds[1]

def children_to_components(df, automl):
    ''' Sets individual components in the dataframe of the log file.

    Parameters:
    -----------
    df: pd.DataFrame
        Contains the Pandas dataframe format of the Gama log file.
    automl: GamaClassifier or GamaRegressor
        Contains either a GamaClassifier object or GamaRegressor object.

    Returns:
    --------
    pd.DataFrame
        Contains an updated pd.DataFrame where a GAMA individual is split into four columns.
    '''
    df['algorithm'], df['component_1'], df['component_2'], df['component_3'] = zip(
        *df['pipeline'].apply(lambda x: pipeline_to_children(x, automl)))
    return df

def prepare_df(log_path, filename_class, filename_regr):
    ''' Executes the transformation from Gama log to df for all the logs in a
            path.

    Parameters:
    -----------
    log_path: string
        Contains name of the path where the logs are stored.
    filename_class: string
        Contains the name for the csv file of the classification tasks.
    filename_regr: string
        Contains the name for the csv file of the regression tasks.

    Returns:
    --------
    str
        Contains a confirmation that the preparation of the dataframes was executed.
    '''
    classification, regression, clustering = get_dataset_ids(10000000)
    df_class, df_regr = log_to_df(path, classification, regression)

    df_class = df_class.reset_index(drop=True)
    df_regr = df_regr.reset_index(drop=True)

    automl_class = GamaClassifier(scoring='accuracy')
    automl_regr = GamaRegressor(scoring='r2')

    df_class = children_to_components(df_class, automl_class)
    df_regr = children_to_components(df_regr, automl_regr)

    df_class.to_csv(filename_class, index=False, sep=';')
    df_regr.to_csv(filename_regr, index=False, sep=';')

    return "Prepared the dataframes."

def log_to_df_file(log_file):
    ''' Converts a GAMA log file to a pd.DataFrame.

    Parameters:
    -----------
    log_file: str
        Contains the name of the file or the path to the file.

    Returns:
    --------
    pd.DataFrame
        Contains a pd.DataFrame for that specific log.
    '''
    report = GamaReport(logfile=log_file)
    return report.evaluations

if __name__ == "__main__":
    #single example:
    log_to_df_file("../data/a411.log")

    #multiple example:
    classification, regression, clustering = get_dataset_ids(10000000) 
    load_path = '../data/*.log'
    filename_class = '../data/testc.csv'
    filename_regr = '../data/testr.csv'
    df_class, df_regr = log_to_df(load_path, classification, regression)

    print(df_class, df_regr)
    df_class = df_class.reset_index(drop=True)
    df_regr = df_regr.reset_index(drop=True)

    automl_regr = GamaRegressor(scoring='r2')
    automl_class = GamaClassifier(scoring='accuracy')

    children_to_components(df_class, automl_class).to_csv(filename_class, index=False)
    children_to_components(df_regr, automl_regr).to_csv(filename_regr, index=False)
