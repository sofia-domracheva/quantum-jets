import numpy as np
import logging
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from qjets.config import ModelConfig

log = logging.getLogger(__name__)

def fit_classical(X_train: np.ndarray, y_train: np.ndarray, config: ModelConfig, seed: int) -> GridSearchCV:
    svc = SVC(kernel="rbf")
    param_grid = {"C": list(config.c_grid), "gamma": list(config.gamma_grid)}
    cv = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=seed)
    grid = GridSearchCV(svc, param_grid, cv=cv, scoring=config.scoring, n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid

def evaluate(model: GridSearchCV, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
    y_pred = model.predict(X_test)
    y_score = model.decision_function(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_score),
    }

def run_baseline(datasets: dict[int, dict[str, np.ndarray]], y_train: np.ndarray, y_test: np.ndarray, config: ModelConfig, seed: int) -> dict[int, dict]:
    results = {} 
    for n, data in datasets.items():
        model = fit_classical(data["train"], y_train, config, seed)
        metrics = evaluate(model, data["test"], y_test)
        results[n] = {"metrics": metrics, "best_params": model.best_params_, "cv_score": model.best_score_}
        log.info(f"Results for {n} qubits: {metrics}, best params: {model.best_params_}, cv score: {model.best_score_}")
    return results