"""Model Registry and Factory Module for Phase 2 Baselines.

Provides standardized instantiations for:
- RUL Regression: Linear Regression, Random Forest, Gradient Boosting, XGBoost
- Early-Warning & Machine Failure Classification: Logistic Regression, Random Forest, Gradient Boosting, XGBoost
"""

from typing import Any, Dict, Optional, Union
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
import xgboost as xgb


def get_regression_models(
    random_state: int = 42,
    n_jobs: int = 4
) -> Dict[str, Any]:
    """Return dictionary of configured baseline regressors for RUL prediction."""
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_leaf=3,
            n_jobs=n_jobs,
            random_state=random_state,
        ),
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            min_samples_leaf=3,
            random_state=random_state,
        ),
        "XGBoost Regressor": xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.08,
            max_depth=5,
            subsample=0.85,
            colsample_bytree=0.85,
            n_jobs=n_jobs,
            random_state=random_state,
            objective="reg:squarederror",
        ),
    }


def get_classification_models(
    class_weight: Optional[str] = "balanced",
    scale_pos_weight: float = 1.0,
    random_state: int = 42,
    n_jobs: int = 4
) -> Dict[str, Any]:
    """Return dictionary of configured baseline classifiers for failure detection."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight=class_weight,
            random_state=random_state,
            solver="lbfgs",
        ),
        "Random Forest Classifier": RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_leaf=2,
            class_weight=class_weight,
            n_jobs=n_jobs,
            random_state=random_state,
        ),
        "Gradient Boosting Classifier": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.08,
            max_depth=4,
            min_samples_leaf=3,
            random_state=random_state,
        ),
        "XGBoost Classifier": xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.08,
            max_depth=4,
            scale_pos_weight=scale_pos_weight,
            subsample=0.85,
            colsample_bytree=0.85,
            n_jobs=n_jobs,
            random_state=random_state,
            eval_metric="logloss",
        ),
    }
