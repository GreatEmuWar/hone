"""Machine learning tuning problems: choosing a model's hyperparameters by cross-validation.

Each problem trains a scikit-learn model on one of its bundled datasets and scores it by
5-fold cross-validation with fixed folds, so every evaluation is a real training run and the
same settings always give the same score. Each problem minimizes 1 − score: 1 − accuracy for
classification, 1 − R² for regression.

The true optimum is unknown, so each problem's optimum is the best value a long reference
search found (see REFERENCE_SEARCH), and error is measured from there. Settings that span
orders of magnitude are log-scale ranges.

These problems need scikit-learn, which hone itself doesn't depend on: `pip install scikit-learn`.
"""
from functools import cache
from typing import Callable

from threadpoolctl import threadpool_limits

from hone.parameters import ChoiceParameter, Condition, RangeParameter, SearchSpace

from .problems import Problem

# How each problem's optimum was found: the best of these runs, each with seeds 0 to 2.
REFERENCE_SEARCH = (
    "BayesianOptimizer with 200 evaluations, and CMA-ES (where it applies), the genetic optimizer "
    "and random search with 1,000"
)

# Model tuning is done on small budgets, so these problems run at these budgets only.
BUDGETS = (10, 20, 50)


def _require_sklearn() -> None:
    try:
        import sklearn  # noqa: F401
    except ImportError as error:
        raise ImportError("the machine learning problems need scikit-learn: pip install scikit-learn") from error


@cache
def _data(name: str):
    from sklearn import datasets

    return getattr(datasets, f"load_{name}")(return_X_y=True)


def _cross_validate(model, dataset: str, classification: bool) -> float:
    """1 − the mean cross-validated score: accuracy for a classifier, R² for a regressor."""
    from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

    X, y = _data(dataset)
    folds = (StratifiedKFold if classification else KFold)(5, shuffle=True, random_state=0)
    # The datasets are small, and threads only add overhead.
    with threadpool_limits(limits=1):
        score = cross_val_score(model, X, y, cv=folds, scoring="accuracy" if classification else "r2").mean()
    return 1.0 - float(score)


def svm_breast_cancer(C: float, gamma: float) -> float:
    """A support vector classifier with an RBF kernel, on 569 breast tumour samples with 30 features."""
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    model = make_pipeline(StandardScaler(), SVC(C=C, gamma=gamma))
    return _cross_validate(model, "breast_cancer", classification=True)


def kernel_ridge_diabetes(alpha: float, gamma: float, kernel: str) -> float:
    """Kernel ridge regression of disease progression from 10 measurements, on 442 diabetes patients."""
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(StandardScaler(), KernelRidge(alpha=alpha, gamma=gamma, kernel=kernel))
    return _cross_validate(model, "diabetes", classification=False)


def random_forest_wine(max_depth: int, min_samples_leaf: int, max_features: float, criterion: str) -> float:
    """A 20-tree random forest classifying 178 wines into 3 cultivars from 13 chemical measurements."""
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(
        n_estimators=20,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        criterion=criterion,
        random_state=0,
        n_jobs=1,
    )
    return _cross_validate(model, "wine", classification=True)


def model_selection_breast_cancer(
    model: str,
    C: float | None = None,
    gamma: float | None = None,
    max_depth: int | None = None,
    min_samples_leaf: int | None = None,
    max_features: float | None = None,
    n_neighbors: int | None = None,
    weights: str | None = None,
) -> float:
    """One of three classifiers, and its settings, on the 569 breast tumour samples svm-breast-cancer uses.

    A support vector classifier with an RBF kernel, a 20-tree random forest, or k nearest
    neighbours. With conditions, only the chosen model's settings are searched.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    if model == "svm":
        estimator = make_pipeline(StandardScaler(), SVC(C=C, gamma=gamma))
    elif model == "forest":
        estimator = RandomForestClassifier(
            n_estimators=20, max_depth=max_depth, min_samples_leaf=min_samples_leaf, max_features=max_features, random_state=0, n_jobs=1
        )
    else:
        estimator = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights))
    return _cross_validate(estimator, "breast_cancer", classification=True)


_MODEL = ChoiceParameter("model", ["svm", "forest", "knn"])
_SVM = [RangeParameter("C", 1e-3, 1e3, log=True), RangeParameter("gamma", 1e-5, 10, log=True)]
_FOREST = [
    RangeParameter("max_depth", 1, 12, integer=True),
    RangeParameter("min_samples_leaf", 1, 20, integer=True),
    RangeParameter("max_features", 0.05, 1.0),
]
_KNN = [RangeParameter("n_neighbors", 1, 50, integer=True), ChoiceParameter("weights", ["uniform", "distance"])]
MODEL_SELECTION = [_MODEL, *_SVM, *_FOREST, *_KNN]
MODEL_CONDITIONS: list[Condition] = [_MODEL["svm"] >> _SVM, _MODEL["forest"] >> _FOREST, _MODEL["knn"] >> _KNN]

# The best value of each problem found by REFERENCE_SEARCH.
OPTIMA = {
    "svm-breast-cancer": 0.014066138798323302,  # accuracy 98.6%, found by CMA-ES
    "kernel-ridge-diabetes": 0.4961748264794844,  # R² 0.504, found by Bayesian optimization
    "random-forest-wine": 0.005555555555555536,  # accuracy 99.4%, found by Bayesian optimization
    "model-selection-breast-cancer": 0.014066138798323302,  # PLACEHOLDER
}

SPACES: dict[str, tuple[Callable[..., float], list]] = {
    "svm-breast-cancer": (
        svm_breast_cancer,
        [RangeParameter("C", 1e-3, 1e3, log=True), RangeParameter("gamma", 1e-5, 10, log=True)],
    ),
    "kernel-ridge-diabetes": (
        kernel_ridge_diabetes,
        [
            RangeParameter("alpha", 1e-4, 100, log=True),
            RangeParameter("gamma", 1e-5, 10, log=True),
            ChoiceParameter("kernel", ["rbf", "laplacian"]),
        ],
    ),
    "random-forest-wine": (
        random_forest_wine,
        [
            RangeParameter("max_depth", 1, 12, integer=True),
            RangeParameter("min_samples_leaf", 1, 20, integer=True),
            RangeParameter("max_features", 0.05, 1.0),
            ChoiceParameter("criterion", ["gini", "entropy"]),
        ],
    ),
}


def ml_problems() -> list[Problem]:
    """The machine learning tuning problems. Needs scikit-learn.

    model-selection-breast-cancer comes twice: with conditions, and as model-selection-breast-cancer-flat
    with every setting always searched, as before conditions, to show what they are worth.
    """
    _require_sklearn()
    problems = [
        Problem(name=name, function=function, search_space=SearchSpace(parameters), optimum=OPTIMA[name], tags=frozenset({"ml"}))
        for name, (function, parameters) in SPACES.items()
    ]
    optimum = OPTIMA["model-selection-breast-cancer"]
    return problems + [
        Problem(
            name="model-selection-breast-cancer",
            function=model_selection_breast_cancer,
            search_space=SearchSpace(MODEL_SELECTION, conditions=MODEL_CONDITIONS),
            optimum=optimum,
            tags=frozenset({"ml", "conditional"}),
        ),
        Problem(
            name="model-selection-breast-cancer-flat",
            function=model_selection_breast_cancer,
            search_space=SearchSpace(MODEL_SELECTION),
            optimum=optimum,
            tags=frozenset({"ml", "conditional", "flat"}),
        ),
    ]
