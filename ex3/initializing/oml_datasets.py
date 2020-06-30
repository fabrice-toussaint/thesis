import pandas as pd
import openml as oml

def arff_and_first_version():
    ''' Returns the dataframe of the datasets from OpenML that are currently
    not depreciated (sparse-ARFF). Also, only the first versions of each data-
    set are kept.

    Parameters:
    -----------

    Returns:
    --------
    list
        Contains an overview of dataset IDs that are not sparse-ARFF and first version only. 
    '''
    #Only focus on ARFF as SPARSE ARFF IS DEPRECIATED
    openml_list = oml.datasets.list_datasets()
    openml_list = pd.DataFrame.from_dict(openml_list, orient='index')
    openml_list = openml_list[openml_list['NumberOfFeatures'].notna()]
    openml_list['format'] = openml_list['format'].str.upper()
    IDs = openml_list[openml_list['format'] == 'ARFF'][['name','did',
    'version']].groupby('name').min()
    openml_list = openml_list[openml_list['did'].isin(IDs['did'])]

    return openml_list

def get_dataset_ids(cell_amount):
    ''' Returns the dataset ids belonging to either classification, regression
    or clustering datasets.

    Parameters:
    -----------
    cell_amount: int
        Contains the amount of cells that should at most be included in the
        datasets extracted from OpenML.

    Returns:
    --------
    list, list, list
        Contains a list with classification dataset ids, contains a list with regression dataset ids,
            contains a list with clustering dataset ids.
    '''
    openml_list = arff_and_first_version()
    openml_list['cells'] = openml_list['NumberOfInstances'] * openml_list['NumberOfFeatures']
    openml_list = openml_list[openml_list['cells'] < cell_amount]
    openml_list = (openml_list).to_dict(orient='index')

    classification = []
    regression = []
    clustering = []

    for datasetID in openml_list.keys():
        if openml_list[datasetID]['NumberOfClasses'] > 1.0:
            classification.append(datasetID)
        elif openml_list[datasetID]['NumberOfClasses'] == 0.0:
            regression.append(datasetID)
        else:
            clustering.append(datasetID)

    return classification, regression, clustering
