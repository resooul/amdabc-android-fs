"""The 7 ML classifiers with the paper's fixed hyperparameters (Table ml_hyperparameters).

Values come from the archived ML-Run-2 notebook; no grid search.
"""
from __future__ import annotations

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

SEED = 42


def make(name: str):
    if name == "XGB":
        from xgboost import XGBClassifier
        return XGBClassifier(tree_method="hist", max_depth=5, n_estimators=20, learning_rate=0.1,
                             random_state=SEED, n_jobs=1, verbosity=0)
    if name == "LR":
        return LogisticRegression(C=1.0, solver="liblinear")
    if name == "GBM":
        return GradientBoostingClassifier(max_depth=5, n_estimators=20, random_state=SEED)
    if name == "RF":
        return RandomForestClassifier(max_depth=5, n_estimators=10, criterion="gini",
                                      random_state=SEED, n_jobs=1)
    if name == "CAT":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(depth=5, iterations=20, learning_rate=0.1, random_seed=SEED,
                                  verbose=False, thread_count=1, allow_writing_files=False)
    if name == "LDA":
        return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    if name == "MLP":
        return MLPClassifier(hidden_layer_sizes=(10, 20, 10), activation="relu", solver="adam",
                             learning_rate="adaptive", max_iter=100, random_state=SEED)
    raise ValueError(name)
