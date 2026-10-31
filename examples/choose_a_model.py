"""Choose a classifier and its settings in one search, with conditional parameters.

Each model has its own settings, and some settings only matter for some of a model's options:
gamma only for the SVM's rbf and poly kernels, degree only for poly. Conditions tell hone
which settings belong to which choices, so it only searches, and only passes the objective,
the settings that apply. A conditional constraint keeps the forest small enough to train quickly.

Needs scikit-learn: pip install scikit-learn
"""
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from hone import BayesianConfig, BayesianOptimizer, ChoiceParameter, RangeParameter, SearchSpace

X, y = load_breast_cancer(return_X_y=True)
folds = StratifiedKFold(5, shuffle=True, random_state=0)

model = ChoiceParameter("model", ["svm", "forest", "logistic"])
C = RangeParameter("C", 1e-3, 1e3, log=True)  # used by both the SVM and logistic regression
kernel = ChoiceParameter("kernel", ["rbf", "poly", "linear"])
gamma = RangeParameter("gamma", 1e-5, 1, log=True)
degree = RangeParameter("degree", 2, 4, integer=True)
trees = RangeParameter("trees", 10, 300, integer=True, log=True)
depth = RangeParameter("depth", 2, 20, integer=True)

search_space = SearchSpace(
    [model, C, kernel, gamma, degree, trees, depth],
    constraints=[model["forest"] >> (trees * depth <= 2000)],  # a forest that trains in a second or so
    conditions=[
        model["svm", "logistic"] >> C,
        model["svm"] >> kernel,
        kernel["rbf", "poly"] >> gamma,  # only with an SVM, since kernel is
        kernel["poly"] >> degree,
        model["forest"] >> [trees, depth],
    ],
)


def accuracy(model, C=None, kernel=None, gamma=None, degree=None, trees=None, depth=None):
    """Mean cross-validated accuracy. Settings that don't apply to the chosen model aren't passed."""
    if model == "svm":
        estimator = SVC(C=C, kernel=kernel, gamma=gamma or "scale", degree=degree or 3)
    elif model == "logistic":
        estimator = LogisticRegression(C=C, max_iter=2000)
    else:
        estimator = RandomForestClassifier(n_estimators=trees, max_depth=depth, random_state=0, n_jobs=1)
    return cross_val_score(make_pipeline(StandardScaler(), estimator), X, y, cv=folds).mean()


if __name__ == "__main__":
    optimizer = BayesianOptimizer(search_space, BayesianConfig(max_evaluations=60, maximize=True, seed=0))
    result = optimizer.optimize(accuracy)
    print(f"accuracy {result.best_fitness:.3f} with {result.best_params}")
