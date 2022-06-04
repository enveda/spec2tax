# -*- coding: utf-8 -*-

"""
    Train classifier on MS data to predict taxonomic samples
    Run with:
        python -m spec2tax
"""

import json
import logging
import os
import time
from collections import Counter
from typing import List, Optional, Union

import click
import numpy as np
import pickle5 as pickle
from sklearn import linear_model, model_selection
from sklearn.metrics import roc_auc_score, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

logger = logging.getLogger(__name__)


@click.command()
@click.option('--input-directory', default='/opt/ml/processing/input')
@click.option('--output-directory', default='/opt/ml/processing/output')
@click.option('--cap')
def wrapper(input_directory, output_directory, cap):
    start_event = {
        'input_directory': input_directory,
    }
    print(json.dumps(start_event))

    # Ensure cap is cast properly
    cap = int(cap)

    output_directory = os.path.join(output_directory)
    os.makedirs(output_directory, exist_ok=True)

    start_time = time.time()

    tax_rank = 'class'

    logger.info(f'Starting spec2fam')
    X = pickle.load(open(f"{input_directory}/X.pkl", 'rb'))
    y = pickle.load(open(f"{input_directory}/y.pkl", 'rb'))
    logger.info(f'loaded X and y')
    logger.info(Counter(y))

    new_X, new_y = [], []
    for x, label in zip(X, y):
        if new_y.count(label) > cap:
            continue
        else:
            # Check some arrays that are empty
            try:
                x = x.tolist()
            except:
                print(f"empty sample: {x}")
                continue

            new_X.append(x)
            new_y.append(label)

    logger.info(f'reduced the dataset to just the largest classes')
    logger.info(Counter(new_y))

    X = np.array(new_X)
    y = np.array(new_y)

    label_to_int_map = {result[0]: result[1] for result in zip(np.unique(y), range(0, len(np.unique(y))))}
    y_with_ints = [label_to_int_map[label] for label in y]
    outer_cv_splits = 5
    inner_cv_splits = 5

    # TODO: Remove later
    # np.random.shuffle(y_with_ints)

    max_iter = 1000

    this_auc_scores, this_f1s, this_recalls, this_class_info = train_elastic_net_model(
        X,
        y_with_ints,
        outer_cv_splits,
        inner_cv_splits,
        [0.01, 0.07, 0.17, 0.29, 0.37, 0.47, 0.59, 0.71, 0.76, 0.83, 0.88, 0.92, 0.97],
        max_iter,
    )
    minutes = (time.time() - start_time) / 60.0
    logger.info(f'Embedding finished after {minutes:.2f} minutes')
    logger.info("model trained, congrats!")

    logger.info('here is the auc performacnce')
    logger.info(this_auc_scores)
    logger.info('here is the f1 performance')
    logger.info(this_f1s)
    logger.info('here is the recall performance')
    logger.info(this_recalls)
    logger.info('here is the class information')
    logger.info(this_class_info)
    performance_metrics = {
        "AUC": this_auc_scores,
        "F1": this_f1s,
        "recall": this_recalls,
        "classes": this_class_info,
    }
    metrics_json = json.dumps(performance_metrics)
    with open(f"{output_directory}/metrics.json", 'w') as outfile:
        outfile.write(metrics_json)


def get_l1_ratios():
    """Return a list of values that are used by the elastic net as hyperparameters."""
    return [
        i / 100
        for i in range(0, 101)
        if not _skip_index(i)
    ]


def _skip_index(i):
    return (i < 70 and (i % 2) == 0) or ((i % 3) == 0) or ((i % 5) == 0)


def train_elastic_net_model(
    x,
    y,
    outer_cv_splits: int,
    inner_cv_splits: int,
    l1_ratio: List[float],
    max_iter: Optional[int] = None,
):
    """Train elastic net model via a nested cross validation given expression data.
    Uses a defined hyperparameter space for l1_ratio.
    :param numpy.array x: 2D matrix of pathway scores and samples
    :param list y: class labels of samples
    :param outer_cv_splits: number of folds for cross validation split in outer loop
    :param inner_cv_splits: number of folds for cross validation split in inner loop
    :param l1_ratio: list of hyper-parameters for l1 and l2 priors
    :param model_name: name of the model
    :param max_iter: default to 1000 to ensure convergence
    :param export: Export the models using :mod:`joblib`
    :return: A list of AUC-ROC scores
    """

    test_info = []
    auc_scores = []
    f1_scores = []
    recall_scores = []

    it = _help_train_elastic_net_model(
        x=x,
        y=y,
        outer_cv_splits=outer_cv_splits,
        inner_cv_splits=inner_cv_splits,
        l1_ratio=l1_ratio,
        max_iter=max_iter,
    )

    # Iterator to calculate metrics for each CV step
    for i, (glm_elastic, y_test, y_pred, y_prob) in enumerate(it):
        # auc_scores.append(roc_auc_score(y_test, y_prob, multi_class='ovr'))
        f1_scores.append(f1_score(y_test, y_pred, average='weighted'))
        recall_scores.append(recall_score(y_test, y_pred, average='weighted'))
        test_info.append({
            "test_classes": dict(Counter(y_test)),
            "predicted_classes": dict(Counter(y_pred))
        })

    # Return a list with all AUC/AUC-PR scores for each CV step    
    return auc_scores, f1_scores, recall_scores, test_info


def _help_train_elastic_net_model(
    x,
    y,
    outer_cv_splits: int,
    inner_cv_splits: int,
    l1_ratio: Union[float, List[float]],
    max_iter: Optional[int] = None,
):
    max_iter = max_iter or 1000
    # Use variation of KFold cross validation that returns stratified folds for outer loop in the CV.
    # The folds are made by preserving the percentage of samples for each class.
    skf = StratifiedKFold(n_splits=outer_cv_splits, shuffle=True)

    # tqdm wrapper to print the current CV state
    iterator = tqdm(skf.split(x, y), desc='Outer CV for classification', total=outer_cv_splits)

    # Parameter Grid
    param_grid = dict(l1_ratio=l1_ratio)

    for train_indexes, test_indexes in iterator:
        # Splice the entire data set so only the training and test sets for this CV iter are used
        x_train, x_test = x[train_indexes], x[test_indexes]
        y_train = [y[train_index] for train_index in train_indexes]
        y_test = [y[test_index] for test_index in test_indexes]

        # Instantiate the model fitting along a regularization path (CV).
        # Inner loop
        estimator = linear_model.LogisticRegression(
            penalty='elasticnet',
            class_weight='balanced',
            solver='saga',
            multi_class='multinomial',
            max_iter=max_iter,
            C=1,
        )

        glm_elastic = model_selection.GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            cv=inner_cv_splits,
            scoring='roc_auc_ovo_weighted',
            n_jobs=-1,
        )

        # Fit model with train data
        glm_elastic.fit(x_train, y_train)

        # Predict trained model with test data
        y_prob = glm_elastic.predict_proba(x_test)
        y_pred = glm_elastic.predict(x_test)

        # Return model and y test y predict to calculate prediction metrics
        yield glm_elastic, y_test, y_pred, y_prob



if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    wrapper()
