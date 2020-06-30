import pandas as pd

from prepare_recommender.knn_metric_mf_choice import return_knn_model
from prepare_recommender.prepare_metadata_knn import best_scoring_knn
from initializing.preprocessing import onehot_or_targ, category_numeric_or_string, impute
from configuration.config import *
from metafeatures.metafeatures import meta_features

from gama import GamaClassifier, GamaRegressor
from gama.genetic_programming.components import Individual
from copy import deepcopy
from sklearn.model_selection import train_test_split, cross_val_score
import openml as oml
import numpy as np

def prepare_best_scores(filename):
    ''' Loads and prepares the csv for the best scoring pipelines.

    Parameters:
    -----------
    filename: str
        Contains the path to the file of the best scoring pipelines.

    Returns:
    --------
    pd.DataFrame
        Contains an updated pd.DataFrame of the best scoring pipelines.
    '''
    best = pd.read_csv(filename, sep=';')
    best = best.drop('Unnamed: 0', axis=1)
    return best

def prepare_meta_features(filename, best):
    ''' Loads and prepares the csv for the meta-features.

    Parameters:
    -----------
    filename: str
        Contains the path to the file of the best scoring pipelines.
    best: pd.DataFrame
        Contains the pd.DataFrame of the best scoring pipelines.

    Returns:
    --------
    pd.DataFrame
        Contains an updated pd.DataFrame of the meta-features.
    '''
    mf = pd.read_csv(filename, sep=';', index_col = [0])
    mf.index = mf.index.astype(int)
    mf = mf[mf.index.isin(best['id'])]
    return mf

def pt_best(df):
    ''' Pivots the meta-features pd.DataFrame based on pipeline, k and dataset id.

    Parameters:
    -----------
    df: pd.DataFrame
        Contains pd.DataFrame of the meta-features.

    Returns:
    --------
    pd.DataFrame
        Contains pivoted pd.DataFrame of the meta-features.
    '''
    pt = pd.pivot_table(df, index=['pipeline', 'k', 'id'], 
                                     values=df.columns[18:len(df.columns)-1]).fillna(0)
    return pt

def get_distance_and_indices(values, model, neighbors):
    ''' Gets the index values and distances for given meta-feature values. 

    Parameters:
    -----------
    values: np.ndarray
        Contains the values of the meta-features.
    model: NearestNeighbor model
        Contains the trained NearestNeighbor model.
    neighbors: int
        Contains the amount of neighbors to include.

    Returns:
    --------
    list, list
        Contains the distances between the neighbors and the values, contains the indices of
            the neighbors in the meta-feature set.
    '''
    return model.kneighbors(values, n_neighbors=neighbors)

def get_pipeline_components(distances, indices, pt):
    ''' Gets the neighbor components based on the distance and indices list.

    Parameters:
    -----------
    distances: list
        Contains the distances between the neighbors and the meta-feature values
    indices: list
        Contains the indices of the neighbors of the meta-feature values.
    pt: pd.DataFrame
        Contains the meta-feature pivot table of the best performing pipelines.

    Returns:
    --------
    list
        Contains tuple elements which hold the closest neighbors.
    '''
    pipelines = []
    for i in range(0, len(distances.flatten())):
         pipelines.append(pt.index[indices.flatten()[i]])
    return pipelines

def make_recommendations(mf, best, task, mfc_subset, mfr_subset, n=1, k=10, metric=None):
    ''' Calculates the recommendations based on the learning task, number of neighbors,
            number of pipelines per neighbor, the distance metric and the meta-feature subset.

    Parameters:
    -----------
    mf: np.ndarray
        Contains the meta-feature values.
    best: pd.DataFrame
        Contains the best scoring pipelines.
    task: str
        Contains the learning task (i.e. "classification" or "regression")
    mfc_subset: list
        Contains the subset for the classification meta-features.
    mfr_subset: list
        Contains the subset for the regression meta-features.
    n: int
        Contains the amount of pipelines per neighbor.
    k: int
        Contains the amount of neighbors.
    metric: str
        Contains the distance metric choice.

    Returns:
    --------
    list
        Contains tuple elements which hold the closest neighbors.
    '''
    if task.lower() == "classification":
        if metric == None:
            metric = "euclidean"
        best = best_scoring_knn(best, n, "accuracy")
        subset = mfc_subset
    elif task.lower() == "regression":
        if metric == None:
            metric = "minkowski"
        best = best_scoring_knn(best, n, "r2")
        subset = mfr_subset
    else:
        return "{} is not implemented.".format(task)

    pt = pt_best(best)[subset]
    knn = return_knn_model(pt, metric=metric)
    dist, ind = get_distance_and_indices(mf, knn, n*k)
    
    return get_pipeline_components(dist, ind, pt)

def execute_recommendations(X, y, cat_ind, recommendations, task, n_jobs):
    ''' Executes the recommendations made by the nearest neighbor model based on
            a learning task and sets the number of jobs to n_jobs for the estimators
            and preprocessing algorithms.

    Parameters:
    -----------
    X: pd.DataFrame
        Contains the dataframe of a given dataset excluding its target column.
    y: pd.Series
        Contains the series of the target of a given dataset.
    cat_ind: list
        Contains boolean values to determine whether a column is categorical or
        not based.
    recommendations: list
        Contains the list with the recommendations made by the nearest neighbor model.
    task: str
        Contains the learning task (i.e. "classification" or "regression")
    n_jobs: int
        Contains what to set the number of jobs at for the estimators and preprocessing algorithms
            available in the recommended pipelines.

    Returns:
    --------
    list
        Contains scores of each pipeline run on X and y.
    '''
    categorical, numeric, string = category_numeric_or_string(X, cat_ind)
        
    if task.lower() == "classification":
        gama = GamaClassifier(scoring='accuracy')
    elif task.lower() == "regression":
        gama = GamaRegressor(scoring='r2')
    else:
        return "{} is not implemented, please try 'classification' or 'regression'".format(task)
    
    scores = []

    for recommendation in recommendations:
        pipeline, k, did = recommendation
        ind = Individual.from_string(pipeline, gama._pset)

        X_pipe = deepcopy(X)
        y_pipe = deepcopy(y)

        X_pipe, y_pipe = onehot_or_targ(X_pipe, y_pipe, categorical, k)

        pipeline = [eval(p.str_nonrecursive) for p in ind.primitives]
        pipeline.reverse()

        try:
            for component in pipeline:
                if pipeline.index(component) == len(pipeline)-1:
                    try:
                        setattr(component, 'n_jobs', n_jobs)
                    except:
                        pass

                    X_train, X_test, y_train, y_test = train_test_split(X_pipe, y_pipe, test_size=0.30, random_state=42)
                    cv_scores = cross_val_score(component, X_pipe, y_pipe, cv=10)
                    score = sum(cv_scores) / 10
                    #component.fit(X_train, y_train)
                    #score = component.score(X_test, y_test)
                    scores.append(score)
                else:
                    if isinstance(component, SelectPercentile) | isinstance(component, SelectFwe):
                        X_pipe = component.fit_transform(X_pipe, y_pipe)
                    else:
                        X_pipe = component.fit_transform(X_pipe)
        except:
            scores.append(0)

    return scores

def recommender(X, y, cat_ind, task, n=1, k=10, metric=None, n_jobs=1):
        ''' Executes the recommender and returns both recommendations and scores
                of the recommendation ran on the dataset (X and y).

    Parameters:
    -----------
    X: pd.DataFrame
        Contains the dataframe of a given dataset excluding its target column.
    y: pd.Series
        Contains the series of the target of a given dataset.
    cat_ind: list
        Contains boolean values to determine whether a column is categorical or
        not based.
    task: str
        Contains the learning task (i.e. "classification" or "regression")
    n: int
        Contains the amount of pipelines per neighbor.
    k: int
        Contains the amount of neighbors.
    metric: str
        Contains the distance metric choice.
    n_jobs: int
        Contains what to set the number of jobs at for the estimators and preprocessing algorithms
            available in the recommended pipelines.
    

    Returns:
    --------
    list, list
        Contains a list with GAMA individuals which represent the recommended pipelines,
            contains a list with the scoring of these pipelines.
    '''
        mfc_subset = ['c1', 'c2', 'f3.mean', 'f3.sd', 'f4.mean', 'f4.sd', 
             'l2.mean', 'l2.sd', 'n1', 'n4.mean', 'n4.sd', 
             't2', 't3', 't4']

        mfr_subset = ['nr_disc', 'ch', 'int', 'nre', 'pb', 'sc', 'sil', 
                    'vdb', 'vdu', 'can_cor.mean', 'can_cor.sd', 'cor.mean', 
                    'cor.sd', 'cov.mean', 'cov.sd', 'eigenvalues.mean', 'eigenvalues.sd', 
                    'g_mean.mean', 'g_mean.sd', 'gravity', 'h_mean.mean', 'h_mean.sd', 
                    'iq_range.mean', 'iq_range.sd', 'kurtosis.mean', 'kurtosis.sd', 
                    'mad.mean', 'mad.sd', 'max.mean', 'max.sd', 'mean.mean', 'mean.sd', 
                    'median.mean', 'median.sd', 'min.mean', 'min.sd', 'nr_cor_attr', 'nr_norm', 
                    'nr_outliers', 'range.mean', 'range.sd', 'sd.mean', 'sd.sd', 'sd_ratio', 
                    'skewness.mean', 'skewness.sd', 'sparsity.mean', 'sparsity.sd', 
                    't_mean.mean', 't_mean.sd', 'var.mean', 'var.sd', 'w_lambda']

        categorical, numeric, string = category_numeric_or_string(X, cat_ind)
        X, y = impute(X, y, categorical, numeric, string, "median")

        if task == "classification":
            best = prepare_best_scores('../data/best_C.csv')
            mfs = np.array(meta_features(X, y, groups=['complexity'], suppress=True)[1]).reshape(1, -1)
            recommendations = make_recommendations(mfs, best, "classification", mfc_subset, mfr_subset, n, k, metric=metric)
            scores = execute_recommendations(X, y, cat_ind, recommendations, "classification", n_jobs)
        else:
            best = prepare_best_scores('../data/best_R.csv')
            mfs = np.array(meta_features(X, y, groups=['statistical', 'general'], suppress=True)[1]).reshape(1, -1)
            recommendations = make_recommendations(mfs, best, "regression", mfc_subset, mfr_subset, n, k, metric=metric)
            scores = execute_recommendations(X, y, cat_ind, recommendations, "regression", n_jobs)

        return recommendations, scores


if __name__ == "__main__":

    mfc_subset = ['c1', 'c2', 'f3.mean', 'f3.sd', 'f4.mean', 'f4.sd', 
             'l2.mean', 'l2.sd', 'n1', 'n4.mean', 'n4.sd', 
             't2', 't3', 't4']

    mfr_subset = ['nr_disc', 'ch', 'int', 'nre', 'pb', 'sc', 'sil', 
                'vdb', 'vdu', 'can_cor.mean', 'can_cor.sd', 'cor.mean', 
                'cor.sd', 'cov.mean', 'cov.sd', 'eigenvalues.mean', 'eigenvalues.sd', 
                'g_mean.mean', 'g_mean.sd', 'gravity', 'h_mean.mean', 'h_mean.sd', 
                'iq_range.mean', 'iq_range.sd', 'kurtosis.mean', 'kurtosis.sd', 
                'mad.mean', 'mad.sd', 'max.mean', 'max.sd', 'mean.mean', 'mean.sd', 
                'median.mean', 'median.sd', 'min.mean', 'min.sd', 'nr_cor_attr', 'nr_norm', 
                'nr_outliers', 'range.mean', 'range.sd', 'sd.mean', 'sd.sd', 'sd_ratio', 
                'skewness.mean', 'skewness.sd', 'sparsity.mean', 'sparsity.sd', 
                't_mean.mean', 't_mean.sd', 'var.mean', 'var.sd', 'w_lambda']

    best_C = prepare_best_scores('../data/best_C.csv')
    best_R = prepare_best_scores('../data/best_R.csv')

    mfc = prepare_meta_features('../data/all_mfc.csv', best_C)
    mfr = prepare_meta_features('../data/all_mfr.csv', best_R)

    mfc = mfc[mfc_subset]
    mfr = mfr[mfr_subset]

    #mfc.to_csv('used_mfc.csv', sep=';')
    #mfr.to_csv('used_mfr.csv', sep=';')

    pt_C = pt_best(best_C)
    pt_R = pt_best(best_C)
    
    ds = oml.datasets.get_dataset(2, download_data=False)
    X, y, cat_ind, attribute_names = ds.get_data(
        dataset_format='DataFrame',
        target=ds.default_target_attribute
    )

    

    print(recommender(X, y, cat_ind, "classification"))