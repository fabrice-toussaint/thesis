import pandas as pd
import numpy as np

from pymfe.mfe import MFE
from itertools import chain, combinations

from sklearn.neighbors import NearestNeighbors

def columns_to_meta_features(df, mf_set):
    ''' Checks whether the meta-features from the Pymfe package are available in the meta-features
            set and only keep the ones specific to that group.

    Parameters:
    -----------
    df: pd.DataFrame
        Contains the whole set of meta-features in a pd.DataFrame.
    mf_set: list
        Contains a list with meta-feature names specific to that meta-feature group.

    Returns:
    --------
    list
        Contains a list with meta-features available in both meta-features set and meta-feature group.
    '''
    mfs = []
    for column in df.columns:
        for mf in mf_set:
            if mf in column and column not in mfs:
                mfs.append(column) 
    return mfs

def stat_to_meta_features(df, mf_set, all_mfs):
    ''' Checks whether the statistical meta-features from the Pymfe package are 
            available in the meta-features set and only keep the ones specific to that group.

    Parameters:
    -----------
    df: pd.DataFrame
        Contains the whole set of meta-features in a pd.DataFrame.
    mf_set: list
        Contains a list with meta-feature names specific to that statistical meta-feature group.
    all_mfs: list
        Contains a list with the names of all meta-feature groups instead of statistical.

    Returns:
    --------
    list
        Contains a list with meta-features available in both meta-features set and statistical meta-feature group.
    '''
    mfs = []
    for column in df.columns:
        for mf in mf_set:
            if mf in column and column not in mfs and column not in all_mfs:
                mfs.append(column) 
    return mfs

def return_knn_model(pt, metric):
    ''' Returns a nearest neighbor model that is trained on a pivot table as input.

    Parameters:
    -----------
    pt: pd.DataFrame
        Contains a pd.DataFrame with meta-features as features and pipeline configurations as index.
    metric: str
        Contains a string which is used to set the distance metric, more information can be found at:
            https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.DistanceMetric.html#sklearn.neighbors.DistanceMetric

    Returns:
    --------
    NearestNeighbor model
        Contains trained NearestNeighbor model.
    '''
    model = NearestNeighbors(algorithm='auto', metric=metric)
    model.fit(pt)
    return model

def all_subsets(ss):
    ''' Returns all combinations of a lists element.
            https://stackoverflow.com/questions/464864/how-to-get-all-possible-combinations-of-a-list-s-elements

    Parameters:
    -----------
    ss: list
        Contains a list with different elements.

    Returns:
    --------
    list
        Contains the combinations of all the elements available in the list ss.
    '''
    return chain(*map(lambda x: combinations(ss, x), range(0, len(ss)+1)))

def create_mf_subsets(mf):
    ''' Returns the unstacked combinations of a lists of lists of meta-feature groups combinations.

    Parameters:
    -----------
    mf: list
        Contains a list which contains lists of the names of the meta-features for the specific meta-feature groups.

    Returns:
    --------
    list
        Contains the combinations of all the different meta-feature groups.
    '''
    mf_subsets = []
    
    for subset in all_subsets(mf):
        subset_combined = [item for sublist in subset for item in sublist]
        if len(subset_combined) > 0:
            mf_subsets.append(subset_combined)
    
    return mf_subsets

def recommendations(mf, subsets, best):
    ''' Returns the recommendations for several test datasets.

    Parameters:
    -----------
    mf: pd.DataFrame
        Contains a pd.DataFrame with all the meta-features generated for the different datasets.d
    subsets: list
        Contains the list with the combinations of the meta-feature groups.
    best: pd.DataFrame
        Contains a pd.DataFrame with the best scoring pipelines for each dataset.

    Returns:
    --------
    pd.DataFrame
        Contains the recommendations of based on the meta-feature subsets and distance metrics.
    '''
    test_indices = np.random.permutation(mf.index)[0:5]
    recommendations = pd.DataFrame()
    distance_metrics = ['euclidean', 'manhattan', 'minkowski']
    for metric in distance_metrics:
        for subset in subsets:
            mf_subset = mf[subset]
            test = mf_subset.loc[mf_subset.index.isin(test_indices)]

            train = mf_subset.loc[~mf_subset.index.isin(test_indices)]
            model = return_knn_model(train, metric)
            for i in range(len(test)):
                distances, indices = model.kneighbors(test.iloc[i:i+1], 5)
                for j in range(len(indices[0])):
                    d = {}
                    recommendation = best[best['id'] == train.iloc[indices[0][j]].name].head(1)
                    d['metric'] = metric
                    d['subset'] = subset
                    d['test_id'] = int(test.iloc[i:i+1].index[0])
                    d['pipeline_recommendation'] = recommendation['pipeline'].values[0]
                    d['k_recommendation'] = recommendation['k'].values[0]
                    recommendations = recommendations.append(d, ignore_index=True)
    return recommendations

if __name__ == "__main__":
    mfc = pd.read_csv('../../data/all_mfc.csv', sep=';', index_col=[0])
    mfr = pd.read_csv('../../data/all_mfr.csv', sep=';', index_col=[0])

    mfc = mfc.fillna(0)
    mfr = mfr.fillna(0)

    bsc = pd.read_csv('../../data/best_C.csv', sep=';')
    bsr = pd.read_csv('../../data/best_R.csv', sep=';')

    mtfs_gen = list(MFE().valid_metafeatures(groups="general"))
    mtfs_stat = list(MFE().valid_metafeatures(groups="statistical"))
    mtfs_it = list(MFE().valid_metafeatures(groups="info-theory"))
    mtfs_mb = list(MFE().valid_metafeatures(groups="model-based"))
    mtfs_lm = list(MFE().valid_metafeatures(groups="landmarking"))
    mtfs_cl = list(MFE().valid_metafeatures(groups="clustering"))
    mtfs_con = list(MFE().valid_metafeatures(groups="concept"))
    mtfs_com = list(MFE().valid_metafeatures(groups="complexity"))

    gen = columns_to_meta_features(mfc, mtfs_gen)
    it = columns_to_meta_features(mfc, mtfs_it)
    mb = columns_to_meta_features(mfc, mtfs_mb)
    lm = columns_to_meta_features(mfc, mtfs_lm)
    cl = columns_to_meta_features(mfc, mtfs_cl)
    con = columns_to_meta_features(mfc, mtfs_con)
    com = columns_to_meta_features(mfc, mtfs_com)
    all_mfs = gen+it+mb+lm+cl+con+com
    stat = stat_to_meta_features(mfc, mtfs_stat, all_mfs)

    mfcs = [gen, it, mb, lm, cl, con, com, stat]
    mfrs = [gen, cl, stat]

    mfc_subsets = create_mf_subsets(mfcs)
    mfr_subsets = create_mf_subsets(mfrs)

    print(bsc.columns)
    print(mfr.columns)
    
    #print(len(recommendations(mfr, mfr_subsets, bsr)))

    