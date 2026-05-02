"""Scikit-learn baseline models."""

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import LinearSVC, LinearSVR

from quasim.ownai.determinism import set_seed


def get_logistic_regression(seed: int = 42) -> LogisticRegression:
    """Get deterministic Logistic Regression classifier.

    Parameters
    ----------
    seed : int
        Random seed

    Returns
    -------
    LogisticRegression
        Configured classifier
    """

    set_seed(seed)
    return LogisticRegression(
        random_state=seed,
        max_iter=200,
        solver="lbfgs",
    )


def get_linear_svc(seed: int = 42) -> LinearSVC:
    """Get deterministic Linear SVC classifier.

    Parameters
    ----------
    seed : int
        Random seed

    Returns
    -------
    LinearSVC
        Configured classifier
    """

    set_seed(seed)
    return LinearSVC(
        random_state=seed,
        max_iter=200,
        dual="auto",
    )


def get_linear_svr(seed: int = 42) -> LinearSVR:
    """Get deterministic Linear SVR regressor.

    Parameters
    ----------
    seed : int
        Random seed

    Returns
    -------
    LinearSVR
        Configured regressor
    """

    set_seed(seed)
    return LinearSVR(
        random_state=seed,
        max_iter=200,
    )


def get_random_forest_classifier(seed: int = 42, n_estimators: int = 100) -> RandomForestClassifier:
    """Get deterministic Random Forest classifier.

    Parameters
    ----------
    seed : int
        Random seed
    n_estimators : int
        Number of trees

    Returns
    -------
    RandomForestClassifier
        Configured classifier
    """

    set_seed(seed)
    return RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=1,
        max_depth=10,
    )


def get_random_forest_regressor(seed: int = 42, n_estimators: int = 100) -> RandomForestRegressor:
    """Get deterministic Random Forest regressor.

    Parameters
    ----------
    seed : int
        Random seed
    n_estimators : int
        Number of trees

    Returns
    -------
    RandomForestRegressor
        Configured regressor
    """

    set_seed(seed)
    return RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=1,
        max_depth=10,
    )


def get_linear_regression() -> LinearRegression:
    """Get deterministic Linear Regression.

    Returns
    -------
    LinearRegression
        Configured regressor
    """

    return LinearRegression()
