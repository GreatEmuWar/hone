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


def accuracy(log10_C, log10_gamma, kernel):
    """Mean cross-validated accuracy; C and gamma are searched on a log scale."""
    model = make_pipeline(StandardScaler(), SVC(C=10**log10_C, gamma=10**log10_gamma, kernel=kernel))
    return cross_val_score(model, X, y, cv=folds).mean()


search_space = [
    RangeParameter("log10_C", -3, 3),
    RangeParameter("log10_gamma", -4, 1),
    ChoiceParameter("kernel", ["rbf", "sigmoid"]),
]

if __name__ == "__main__":
    optimizer = BayesianOptimizer(search_space, BayesianConfig(max_evaluations=30, maximize=True, seed=0))
    result = optimizer.optimize(accuracy)
    print(f"accuracy {result.best_fitness:.3f} with {result.best_params}")
