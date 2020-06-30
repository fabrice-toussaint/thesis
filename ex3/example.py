from recommender import *
import openml as oml

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

ds = oml.datasets.get_dataset(2, download_data=False)
X, y, cat_ind, attribute_names = ds.get_data(
    dataset_format='DataFrame',
    target=ds.default_target_attribute
)

recommendations_and_scoring = recommender(X, y, cat_ind, "classification")