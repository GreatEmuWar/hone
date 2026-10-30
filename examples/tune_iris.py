"""Tune a support vector classifier on the iris dataset with Bayesian optimization.

Needs scikit-learn: pip install scikit-learn
"""
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from hone import BayesianConfig, BayesianOptimizer, ChoiceParameter, RangeParameter

X, y = load_iris(return_X_y=True)
folds = StratifiedKFold(5, shuffle=True, random_state=0)


def accuracy(C, gamma, kernel):
    """Mean cross-validated accuracy."""
    model = make_pipeline(StandardScaler(), SVC(C=C, gamma=gamma, kernel=kernel))
    return cross_val_score(model, X, y, cv=folds).mean()


search_space = [
    # C and gamma span orders of magnitude, so search them on a log scale.
    RangeParameter("C", 1e-3, 1e3, log=True),
    RangeParameter("gamma", 1e-4, 10, log=True),
    ChoiceParameter("kernel", ["rbf", "sigmoid"]),
]

if __name__ == "__main__":
    optimizer = BayesianOptimizer(search_space, BayesianConfig(max_evaluations=30, maximize=True, seed=0))
    result = optimizer.optimize(accuracy)
    print(f"accuracy {result.best_fitness:.3f} with {result.best_params}")
