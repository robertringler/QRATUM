"""XGBoost and LightGBM baseline models with tiny configs."""

from quasim.ownai.determinism import set_seed


def get_xgboost_classifier(seed: int = 42, n_estimators: int = 50):
    """Get XGBoost classifier with tiny config.

    Parameters
    ----------
    seed : int
        Random seed
    n_estimators : int
        Number of boosting rounds

    Returns
    -------
    XGBClassifier
        Configured classifier
    """
    try:
        from xgboost import XGBClassifier

        set_seed(seed)
        return XGBClassifier(
            n_estimators=n_estimators,
            max_depth=3,
            learning_rate=0.1,
            random_state=seed,
            n_jobs=1,
            verbosity=0,
        )
    except ImportError:
        raise ImportError("XGBoost not installed. Install with: pip install xgboost")


def get_xgboost_regressor(seed: int = 42, n_estimators: int = 50):
    """Get XGBoost regressor with tiny config.

    Parameters
    ----------
    seed : int
        Random seed
    n_estimators : int
        Number of boosting rounds

    Returns
    -------
    XGBRegressor
        Configured regressor
    """
    try:
        from xgboost import XGBRegressor

        set_seed(seed)
        return XGBRegressor(
            n_estimators=n_estimators,
            max_depth=3,
            learning_rate=0.1,
            random_state=seed,
            n_jobs=1,
            verbosity=0,
        )
    except ImportError:
        raise ImportError("XGBoost not installed. Install with: pip install xgboost")


def get_lightgbm_classifier(seed: int = 42, n_estimators: int = 50):
    """Get LightGBM classifier with tiny config.

    Parameters
    ----------
    seed : int
        Random seed
    n_estimators : int
        Number of boosting rounds

    Returns
    -------
    LGBMClassifier
        Configured classifier
    """
    try:
        from lightgbm import LGBMClassifier

        set_seed(seed)
        return LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=3,
            learning_rate=0.1,
            random_state=seed,
            n_jobs=1,
            verbosity=-1,
            force_row_wise=True,  # For determinism
        )
    except ImportError:
        raise ImportError("LightGBM not installed. Install with: pip install lightgbm")


def get_lightgbm_regressor(seed: int = 42, n_estimators: int = 50):
    """Get LightGBM regressor with tiny config.

    Parameters
    ----------
    seed : int
        Random seed
    n_estimators : int
        Number of boosting rounds

    Returns
    -------
    LGBMRegressor
        Configured regressor
    """
    try:
        from lightgbm import LGBMRegressor

        set_seed(seed)
        return LGBMRegressor(
            n_estimators=n_estimators,
            max_depth=3,
            learning_rate=0.1,
            random_state=seed,
            n_jobs=1,
            verbosity=-1,
            force_row_wise=True,  # For determinism
        )
    except ImportError:
        raise ImportError("LightGBM not installed. Install with: pip install lightgbm")
