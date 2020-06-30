import pandas as pd
from pymfe.mfe import MFE

def meta_features(X, y, groups=None, suppress=True):
    ''' Extracts and returns the meta-features from a dataset using the Pymfe
    package.

    Parameters:
    -----------
    X: pd.DataFrame
        Contains the dataframe of a given dataset excluding its target column.
    y: pd.Series
        Contains the series of the target of a given dataset.
    groups: list
        Contains the names of the meta-feature groups as available in the
        Pymfe package (pymfe.readthedocs.io).

    Returns:
    --------
    list
        Contains a list of lists where one list denotes the meta-feature names
            and the other denoted the meta-feature values respective to the names.
    '''
    try:
        X = X.to_numpy()
    except:
        pass

    try:
        y = y.to_numpy()
    except:
        pass
    
    if groups == None:
        mfe = MFE(suppress_warnings=suppress)
        mfe.fit(X, y)
        ft = mfe.extract()
    else:
        mfe = MFE(groups=groups, suppress_warnings=suppress)
        mfe.fit(X, y)
        ft = mfe.extract()

    return ft

def meta_features_datasets(datasets, classification, regression):
    ''' Extracts and returns the meta-features from the datasets available in
    OpenML.

    Parameters:
    -----------
    datasets: list
        Contains the dataset ids where meta-features will be extracted from.
    classification: list
        Contains the dataset ids that are classification datasets.
    regression: list
        Contains the datasets ids that are regression datasets.

    Returns:
    --------
    pd.DataFrame, pd.DataFrame
        Contains pd.DataFrame which contains all of the meta-features for the classification datasets of OpenML,
            contains pd.DataFrame which contains all of the meta-features for the regression datasets of OpenML.
    '''
    mfs_classification = pd.DataFrame(columns=['did'])
    mfs_regression = pd.DataFrame(columns=['did'])

    for did in datasets:
        ds = oml.datasets.get_dataset(did, download_data=False)
        try:
            try:
                X, y, categorical_indicator, attribute_names = ds.get_data(
                    dataset_format='array',
                    target=ds.default_target_attribute
                )
            except:
                X, y, categorical_indicator, attribute_names = ds.get_data(
                    dataset_format='dataframe',
                    target=ds.default_target_attribute
                )
            res = dict()
        except:
            print("Failed loading data.")
            res = dict()
        try:
            mf = meta_features(X, y)
            res = dict(zip(mf[0], mf[1]))
        except:
            print("Failed extracting meta-features.")
            pass

        res['did'] = did

        if did in classification:
            mfs_classification = mfs_classification.append(res, ignore_index=True)
        else:
            mfs_regression = mfs_regression.append(res, ignore_index=True)

    return mfs_classification, mfs_regression
