"""Воспроизведение финального стекинга без исследовательских переборов."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

PROJECT_DIR = Path(__file__).resolve().parent
RANDOM_STATE = 23


def make_features(df, frequency_maps=None):
    """Та же подготовка признаков, что в финальном блоке research.ipynb."""
    X = df.drop(columns=["id", "target"], errors="ignore").copy()
    X = X.loc[:, ~X.columns.str.startswith("ps_calc_")].copy()
    base_cols = X.columns.tolist()
    categorical_cols = [col for col in base_cols if col.endswith("_cat")]
    numeric_cols = [col for col in base_cols if col not in categorical_cols]
    ind_cols = [col for col in base_cols if col.startswith("ps_ind_")]
    ind_key = pd.util.hash_pandas_object(X[ind_cols], index=False)

    if frequency_maps is None:
        frequency_maps = {
            col: X[col].value_counts(dropna=False) for col in categorical_cols
        }
        frequency_maps["new_ind"] = ind_key.value_counts()

    for col in categorical_cols:
        X[f"{col}_count"] = (
            X[col].map(frequency_maps[col]).fillna(0).astype(np.float32)
        )
    X["new_ind_count"] = (
        ind_key.map(frequency_maps["new_ind"])
        .fillna(0)
        .to_numpy(dtype=np.float32)
    )
    X["missing_count"] = (X[base_cols].eq(-1) | X[base_cols].isna()).sum(axis=1)
    ind_bin_cols = [
        col for col in base_cols
        if col.startswith("ps_ind_") and col.endswith("_bin")
    ]
    X["ind_bin_sum"] = X[ind_bin_cols].eq(1).sum(axis=1)
    X[numeric_cols] = X[numeric_cols].replace(-1, np.nan)
    X[categorical_cols] = X[categorical_cols].fillna(-1)
    X = X.replace([np.inf, -np.inf], np.nan)
    numeric_cols = [col for col in X.columns if col not in categorical_cols]
    preprocessor = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", dtype=np.float32),
             categorical_cols),
            ("num", SimpleImputer(strategy="median"), numeric_cols),
        ],
        sparse_threshold=1.0,
    )
    return X, preprocessor, frequency_maps


def build_stacking(preprocessor):
    """Параметры и значения seed сохранены из финального блока ноутбука."""
    def model_pipeline(model):
        return Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("model", model),
        ])

    return StackingClassifier(
        estimators=[
            ("lgbm", model_pipeline(LGBMClassifier(
                n_estimators=500, learning_rate=0.1, max_depth=3,
                reg_alpha=10, n_jobs=-1, verbose=-1,
            ))),
            ("cat", model_pipeline(CatBoostClassifier(
                iterations=500, learning_rate=0.1, depth=5, verbose=0,
                allow_writing_files=False,
            ))),
            ("xgboost", model_pipeline(XGBClassifier(
                n_estimators=500, learning_rate=0.1, max_depth=3,
                reg_alpha=10, eval_metric="mlogloss", n_jobs=-1, verbosity=0,
            ))),
        ],
        final_estimator=LogisticRegression(
            random_state=RANDOM_STATE, max_iter=500,
        ),
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        stack_method="predict_proba",
        passthrough=False,
        n_jobs=1,
    )


def validate(train):
    train_part, valid_part = train_test_split(
        train, test_size=0.2, stratify=train["target"],
        random_state=RANDOM_STATE,
    )
    X_train, preprocessor, frequency_maps = make_features(train_part)
    X_valid, _, _ = make_features(valid_part, frequency_maps)
    model = build_stacking(preprocessor)
    print("Обучение стекинга для validation (5 фолдов)…", flush=True)
    model.fit(X_train, train_part["target"].to_numpy())
    probability = model.predict_proba(X_valid)[:, 1]
    gini = 2 * roc_auc_score(valid_part["target"], probability) - 1
    print(f"Validation Gini: {gini:.5f}", flush=True)
    return gini


def make_submission(train, test, output):
    X_train, preprocessor, frequency_maps = make_features(train)
    X_test, _, _ = make_features(test, frequency_maps)
    if not X_train.columns.equals(X_test.columns):
        raise ValueError("Наборы признаков train и test должны совпадать.")
    model = build_stacking(preprocessor)
    print("Обучение стекинга на полном train (5 фолдов)…", flush=True)
    model.fit(X_train, train["target"].astype(np.int8))
    probability = model.predict_proba(X_test)[:, 1]
    if not np.isfinite(probability).all() or not (
        (probability >= 0) & (probability <= 1)
    ).all():
        raise ValueError("Прогноз содержит некорректные вероятности.")
    submission = pd.DataFrame({"id": test["id"].to_numpy(), "target": probability})
    output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output, index=False)
    print(f"Сохранено {len(submission):,} строк: {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=["validate", "submit", "all"], default="all",
        help="validate: локальный Gini; submit: создание CSV; all: оба шага",
    )
    parser.add_argument(
        "--data-dir", type=Path,
        default=PROJECT_DIR / "porto-seguro-safe-driver-prediction",
        help="каталог с train.csv и test.csv",
    )
    parser.add_argument(
        "--output", type=Path, default=PROJECT_DIR / "submission_reproduced.csv",
        help="путь для нового CSV (существующий файл по этому пути заменяется)",
    )
    args = parser.parse_args()
    required_files = [args.data_dir / "train.csv"]
    if args.mode in {"submit", "all"}:
        required_files.append(args.data_dir / "test.csv")
    for file in required_files:
        if not file.is_file():
            parser.error(f"Не найден {file}. Подготовьте данные по инструкции в README.md.")

    started = perf_counter()
    train = pd.read_csv(required_files[0])
    if "target" not in train or set(train["target"].unique()) != {0, 1}:
        parser.error("train.csv должен содержать target с классами 0 и 1.")
    test = None
    if args.mode in {"submit", "all"}:
        test = pd.read_csv(args.data_dir / "test.csv")
        if "id" not in test or test["id"].isna().any() or test["id"].duplicated().any():
            parser.error("test.csv должен содержать уникальные непустые id.")
        if not train.drop(columns=["target"]).columns.equals(test.columns):
            parser.error("Состав и порядок колонок train и test должны совпадать, кроме target.")
    if args.mode in {"validate", "all"}:
        validate(train)
    if args.mode in {"submit", "all"}:
        make_submission(train, test, args.output)
    print(f"Время выполнения: {(perf_counter() - started) / 60:.1f} мин.")


if __name__ == "__main__":
    main()
