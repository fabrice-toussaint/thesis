# Thesis

This repository contains the code for the experiments that I ran during my thesis. The folder EX1EX2 contains the code used in experiment 1 and experiment 2, which focused on random forest classification for predicting whether
a pipeline configuration with preprocessors improves over a baseline pipeline without preprocessors. The folder EX3 contains the code for the pipeline recommender system I built, which gives pipeline recommendations for a dataset.

## Getting Started

The following contains the data that was both generated and used during the experiments: https://drive.google.com/drive/folders/1woWz-NGYCFrkdV8nGEWd_t_24qCRaGxx?usp=sharing. In both cases, the data that is necessary for running the models,
excludes the larger csv files. In EX1EX2 these files are called "mdll_c.csv", "mdll_r.csv", "mdrf_c.csv" and "mdrf_r.csv". In EX3 these files are called "metadata_gama_C.csv" and "metadata_gama_R.csv". After cloning the packages, for the EX1EX2 models you need to redirect to this folder, while for EX3 you need to redirect to the folder for EX3.

The data needs to be saved in a folder called "data" in the "thesis" folder where data for EX1EX2 needs to be stored in a folder called "ex1ex2" and "ex3" for EX3.

### Prerequisites
```
pip install gama sklearn-deap category-encoders sklearn pandas numpy openml
```

### Example

The recommender is capable of giving pipeline recommendations and execute these recommendations manually or automatically. An example is given below:

```
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
```

## Built With

* [Gama](https://github.com/PGijsbers/gama) - The meta-data collector in EX3.
* [Sklearn-deap](https://github.com/rsteca/sklearn-deap) - The meta-data collector in EX1.

## Authors

* **Fabrice Toussaint** - *Initial work* - [Pipeline Recommender](https://github.com/fabrice-toussaint/thesis)



## Acknowledgments

* Martijn Willemsen, first thesis supervisor
* Joaquin Vanschoren, second thesis supervisor
* Sako Arts, supervisor from [Wolfpack](https://wolfpackit.nl/)
* Pieter Gijsbers, creator of GAMA
