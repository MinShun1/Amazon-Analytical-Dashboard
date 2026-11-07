"""Amazon order-cancellation experiments, reproduced from the notebook.

* Experiment 1   (cells 200-214): order-level features, temporal split
  (train <= 31 May 2022, test = June), 10 model configurations.
* Experiment 1.1 (cells 216-234): richer order-level features, April train
  undersampled to 65:35, May validation; 3 baseline + 3 SMOTENC models.
  The notebook evaluated 1.1 on the *validation* month only.

Configurations, seeds and preprocessing are unchanged. No tuning.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE, SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.config import (
    EXP11_SPLITS, EXP11_TARGET_CANCEL_RATE, EXP1_CUTOFF, RANDOM_STATE,
    STATE_MAPPING, UNRESOLVED_STATUSES,
)

TARGET_NAMES = ["Not Cancelled", "Cancelled"]


# ---------------------------------------------------------------------------
# Shared evaluation
# ---------------------------------------------------------------------------
def evaluate(y_true, y_pred, y_prob) -> dict[str, Any]:
    return {
        "metrics": {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1-Score": f1_score(y_true, y_pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_true, y_prob),
            "PR-AUC": average_precision_score(y_true, y_prob),
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, target_names=TARGET_NAMES, zero_division=0, output_dict=True),
    }


@dataclass
class ExperimentResult:
    name: str
    eval_split: str
    metrics: pd.DataFrame
    confusion_matrices: dict[str, list[list[int]]]
    reports: dict[str, dict]
    predictions: pd.DataFrame            # one row per evaluated order, prob_/pred_ per model
    split_summary: pd.DataFrame
    models: dict[str, Any] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)


def _split_row(name: str, df: pd.DataFrame, target: str = "cancelled") -> dict[str, Any]:
    return {
        "split": name, "orders": len(df),
        "cancelled": int(df[target].sum()),
        "cancel_rate_pct": df[target].mean() * 100,
        "start": df["Date"].min(), "end": df["Date"].max(),
    }


# ===========================================================================
# Experiment 1
# ===========================================================================
def build_exp1_dataset(amazon: pd.DataFrame) -> pd.DataFrame:
    """Cells 200-204: one row per resolved order with multi-hot category/size
    flags, month and weekday, and the order Date (for the temporal split)."""
    prep = amazon.copy()
    prep.columns = prep.columns.str.strip()
    source = prep.loc[~prep["Status"].isin(UNRESOLVED_STATUSES)].copy()

    base_cols = ["Order ID", "Date", "Fulfilment", "Sales Channel",
                 "ship-service-level", "B2B", "ship-state"]
    base = source[base_cols + ["Status"]].drop_duplicates(subset="Order ID").copy()
    base["cancelled"] = base["Status"].eq("Cancelled").astype(int)
    base = base.drop(columns="Status")

    cat = pd.get_dummies(source["Category"], prefix="category", dtype=int)
    cat["Order ID"] = source["Order ID"]
    cat = cat.groupby("Order ID").max()
    size = pd.get_dummies(source["Size"], prefix="size", dtype=int)
    size["Order ID"] = source["Order ID"]
    size = size.groupby("Order ID").max()

    ml = base.set_index("Order ID").join(cat, how="left").join(size, how="left").reset_index()
    ml["Date"] = pd.to_datetime(ml["Date"], errors="coerce")
    ml["order_month"] = ml["Date"].dt.month
    ml["order_dayofweek"] = ml["Date"].dt.dayofweek
    ml = ml.drop(columns="Date")

    order_dates = prep[["Order ID", "Date"]].drop_duplicates(subset="Order ID").copy()
    order_dates["Date"] = pd.to_datetime(order_dates["Date"], errors="coerce")
    return ml.merge(order_dates, on="Order ID", how="left", validate="one_to_one")


EXP1_CATEGORICAL = ["Fulfilment", "Sales Channel", "ship-service-level", "B2B",
                    "ship-state", "order_month", "order_dayofweek"]


def _to_object_nan(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for col in EXP1_CATEGORICAL:
        X[col] = X[col].astype(object).where(X[col].notna(), np.nan)
    return X


def exp1_preprocessor(numeric_features: list[str]) -> ColumnTransformer:
    """Cell 206."""
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    return ColumnTransformer([
        ("categorical", categorical, EXP1_CATEGORICAL),
        ("numeric", numeric, numeric_features),
    ], remainder="drop")


def exp1_models(scale_pos_weight: float) -> dict[str, Any]:
    """Cells 207, 208, 210, 212 (in that order)."""
    xgb_kwargs = dict(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8,
                      colsample_bytree=0.8, objective="binary:logistic",
                      eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1)
    rf_kwargs = dict(n_estimators=300, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1)

    def smote(clf):
        return ImbPipeline([
            ("smote", SMOTE(sampling_strategy="auto", random_state=RANDOM_STATE, k_neighbors=5)),
            ("classifier", clf),
        ])

    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Balanced Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(**rf_kwargs),
        "Balanced Random Forest": RandomForestClassifier(class_weight="balanced", **rf_kwargs),
        "XGBoost": XGBClassifier(**xgb_kwargs),
        "Balanced XGBoost": XGBClassifier(scale_pos_weight=scale_pos_weight, **xgb_kwargs),
        "SMOTE Logistic Regression": smote(LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        "SMOTE Random Forest": smote(RandomForestClassifier(**rf_kwargs)),
        "SMOTE XGBoost": smote(XGBClassifier(**xgb_kwargs)),
        "SMOTE CatBoost": smote(CatBoostClassifier(
            iterations=300, depth=5, learning_rate=0.05, loss_function="Logloss",
            random_seed=RANDOM_STATE, verbose=0, thread_count=-1,
            allow_writing_files=False)),  # no catboost_info/ logs; model unchanged
    }


def run_experiment_1(amazon: pd.DataFrame, log=print) -> ExperimentResult:
    ml = build_exp1_dataset(amazon)
    cutoff = pd.Timestamp(EXP1_CUTOFF)
    train, test = ml.loc[ml["Date"] <= cutoff].copy(), ml.loc[ml["Date"] > cutoff].copy()
    excluded = ["Order ID", "Date", "cancelled"]
    X_train, y_train = train.drop(columns=excluded), train["cancelled"]
    X_test, y_test = test.drop(columns=excluded), test["cancelled"]

    numeric_features = [c for c in X_train.columns if c.startswith(("category_", "size_"))]
    pre = exp1_preprocessor(numeric_features)
    Xtr = pre.fit_transform(_to_object_nan(X_train))
    Xte = pre.transform(_to_object_nan(X_test))

    spw = (y_train == 0).sum() / (y_train == 1).sum()
    rows, cms, reports, models = [], {}, {}, {}
    preds = test[["Order ID", "Date", "cancelled"]].reset_index(drop=True)
    for name, model in exp1_models(spw).items():
        log(f"  [Exp 1] training {name} ...")
        model.fit(Xtr, y_train)
        y_pred = model.predict(Xte)
        y_prob = model.predict_proba(Xte)[:, 1]
        ev = evaluate(y_test, y_pred, y_prob)
        rows.append({"Model": name, **ev["metrics"]})
        cms[name], reports[name] = ev["confusion_matrix"], ev["classification_report"]
        preds[f"prob__{name}"] = y_prob
        preds[f"pred__{name}"] = np.asarray(y_pred).astype(int)
        models[name] = model

    preds = preds.merge(
        test[["Order ID", "Fulfilment", "Sales Channel", "ship-service-level", "B2B", "ship-state"]],
        on="Order ID", how="left")
    return ExperimentResult(
        name="Experiment 1", eval_split="Test (June 2022)",
        metrics=pd.DataFrame(rows),
        confusion_matrices=cms, reports=reports, predictions=preds,
        split_summary=pd.DataFrame([_split_row("Train", train), _split_row("Test", test)]),
        models={"preprocessor": pre, **models},
        extras={"dataset": ml, "scale_pos_weight": float(spw),
                "numeric_features": numeric_features},
    )


# ===========================================================================
# Experiment 1.1
# ===========================================================================
def build_exp11_features(amazon: pd.DataFrame) -> pd.DataFrame:
    """Cells 216-221: order-level features for resolved orders."""
    raw = amazon.copy()
    raw.columns = raw.columns.str.strip()
    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")

    source = raw.loc[~raw["Status"].isin(UNRESOLVED_STATUSES)].copy()
    orders = source[["Order ID", "Date", "Status"]].drop_duplicates(subset="Order ID").copy()
    orders["cancelled"] = orders["Status"].eq("Cancelled").astype(int)

    feat_src = raw.loc[raw["Order ID"].isin(set(orders["Order ID"]))].copy()
    features = feat_src.groupby("Order ID").agg(
        Date=("Date", "first"),
        Fulfilment=("Fulfilment", "first"),
        Sales_Channel=("Sales Channel", "first"),
        Ship_Service_Level=("ship-service-level", "first"),
        Qty=("Qty", "sum"),
        Item_Count=("sku", "size"),
        Ship_City=("ship-city", "first"),
        Ship_State=("ship-state", "first"),
        B2B=("B2B", "first"),
    )
    cat = pd.crosstab(feat_src["Order ID"], feat_src["Category"]).clip(upper=1)
    cat.columns = [f"Category_{c}" for c in cat.columns]
    size = pd.crosstab(feat_src["Order ID"], feat_src["Size"]).clip(upper=1)
    size.columns = [f"Size_{c}" for c in size.columns]
    features = features.join(cat).join(size)
    features["cancelled"] = orders.set_index("Order ID")["cancelled"].reindex(features.index)
    return features.reset_index()


def _between(df: pd.DataFrame, bounds: tuple[str, str]) -> pd.DataFrame:
    return df.loc[df["Date"].between(*bounds)].copy()


def rare_states_from_train(train_features: pd.DataFrame) -> list[str]:
    """Cell 225: states with < 100 April orders become 'Other'."""
    clean = train_features["Ship_State"].fillna("Unknown").replace(STATE_MAPPING)
    counts = clean.value_counts()
    return counts[(counts < 100) & (counts.index != "Unknown")].index.tolist()


def undersample_by_date(train: pd.DataFrame) -> pd.DataFrame:
    """Cell 228: keep all cancellations, sample non-cancellations per date so
    the cancellation share is 35 % while preserving the daily distribution."""
    majority = train[train["cancelled"] == 0].copy()
    minority = train[train["cancelled"] == 1].copy()
    target = int(len(minority) * (1 - EXP11_TARGET_CANCEL_RATE) / EXP11_TARGET_CANCEL_RATE)

    per_date = majority.groupby("Date").size()
    samples = (per_date / len(majority) * target).round().astype(int)
    diff = target - samples.sum()
    if diff != 0:
        idx = samples.sort_values(ascending=False).index[:abs(diff)]
        samples.loc[idx] += 1 if diff > 0 else -1

    sampled = (
        majority.groupby("Date", group_keys=False)
        .apply(lambda g: g.sample(n=int(samples.loc[g.name]), random_state=RANDOM_STATE))
        .reset_index(drop=True)
    )
    balanced = pd.concat([sampled, minority], ignore_index=True)
    return balanced.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


EXP11_CATEGORICAL = ["Fulfilment", "Sales_Channel", "Ship_Service_Level",
                     "Ship_State_Grouped", "Day_of_Week"]


def run_experiment_11(amazon: pd.DataFrame, log=print) -> ExperimentResult:
    features = build_exp11_features(amazon)

    april = _between(features, EXP11_SPLITS["train"])
    rare_states = rare_states_from_train(april)

    category_features = [c for c in features.columns if c.startswith("Category_")]
    size_features = [c for c in features.columns if c.startswith("Size_")]
    base_features = ["Fulfilment", "Sales_Channel", "Ship_Service_Level", "B2B",
                     "Item_Count", "Day_of_Week"]

    md = features.copy()                                                    # cell 226
    md["Day_of_Week"] = md["Date"].dt.dayofweek
    md["Ship_State"] = md["Ship_State"].fillna("Unknown").replace(STATE_MAPPING)
    md["Ship_State_Grouped"] = md["Ship_State"].where(~md["Ship_State"].isin(rare_states), "Other")
    selected = base_features + ["Ship_State_Grouped"] + category_features + size_features
    md = md[["Order ID", "Date"] + selected + ["cancelled"]].copy()

    train = _between(md, EXP11_SPLITS["train"])                            # cell 227
    val = _between(md, EXP11_SPLITS["validation"])
    test = _between(md, EXP11_SPLITS["test"])
    balanced = undersample_by_date(train)                                   # cell 228

    binary = ["B2B"] + category_features + size_features                   # cell 229
    numerical = ["Item_Count"]

    def cast(df):
        df = df.copy()
        for c in EXP11_CATEGORICAL:
            df[c] = df[c].astype(str)
        for c in binary:
            df[c] = df[c].astype(int)
        return df

    tr_src, val_src = cast(balanced), cast(val)
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(train[EXP11_CATEGORICAL].astype(str))
    names = encoder.get_feature_names_out(EXP11_CATEGORICAL)

    def matrix(df):
        enc = pd.DataFrame(encoder.transform(df[EXP11_CATEGORICAL]), columns=names, index=df.index)
        return pd.concat([enc, df[numerical + binary].copy()], axis=1)

    X_train, X_val = matrix(tr_src), matrix(val_src)
    y_train, y_val = tr_src["cancelled"], val_src["cancelled"]

    scaler = StandardScaler()                                               # cell 230
    X_train_s, X_val_s = X_train.copy(), X_val.copy()
    X_train_s[["Item_Count"]] = scaler.fit_transform(X_train[["Item_Count"]])
    X_val_s[["Item_Count"]] = scaler.transform(X_val[["Item_Count"]])

    baseline = {                                                            # cell 231
        "Undersampled Logistic Regression": (LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE), X_train_s, X_val_s),
        "Undersampled Random Forest": (RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
            X_train, X_val),
        "Undersampled HistGradientBoosting": (HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.1, random_state=RANDOM_STATE), X_train, X_val),
    }

    # SMOTENC on the *original* April training data (cells 232-233)
    X_orig = train[selected].copy()
    for c in EXP11_CATEGORICAL:
        X_orig[c] = X_orig[c].astype(str)
    smotenc = SMOTENC(categorical_features=[c for c in selected if c != "Item_Count"],
                      sampling_strategy=3 / 7, random_state=RANDOM_STATE, k_neighbors=5)
    X_sm_arr, y_sm = smotenc.fit_resample(X_orig, train["cancelled"].copy())
    X_sm_raw = pd.DataFrame(X_sm_arr, columns=selected)
    y_sm = pd.Series(np.asarray(y_sm), name="cancelled")
    for c in binary:
        X_sm_raw[c] = X_sm_raw[c].astype(int)
    X_sm_raw["Item_Count"] = X_sm_raw["Item_Count"].round().clip(lower=1).astype(int)

    sm_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    sm_encoder.fit(train[EXP11_CATEGORICAL].astype(str))
    sm_names = sm_encoder.get_feature_names_out(EXP11_CATEGORICAL)

    def sm_matrix(cat_df, rest_df):
        cat_df = cat_df.copy()
        for c in EXP11_CATEGORICAL:
            cat_df[c] = cat_df[c].astype(str)
        enc = pd.DataFrame(sm_encoder.transform(cat_df), columns=sm_names)
        return pd.concat([enc, rest_df[["Item_Count"] + binary].reset_index(drop=True)], axis=1)

    X_sm = sm_matrix(X_sm_raw[EXP11_CATEGORICAL], X_sm_raw)
    X_val_m = sm_matrix(val[EXP11_CATEGORICAL], val)
    X_sm_s, X_val_m_s = X_sm.copy(), X_val_m.copy()
    X_sm_s["Item_Count"] = scaler.transform(X_sm[["Item_Count"]]).ravel()
    X_val_m_s["Item_Count"] = scaler.transform(X_val_m[["Item_Count"]]).ravel()
    y_val_m = val["cancelled"].reset_index(drop=True)

    smote_models = {                                                        # cell 234
        "SMOTENC Logistic Regression": (LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE), X_sm_s, X_val_m_s),
        "SMOTENC Random Forest": (RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1), X_sm, X_val_m),
        "SMOTENC HistGradientBoosting": (HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.1, random_state=RANDOM_STATE), X_sm, X_val_m),
    }

    rows, cms, reports, models = [], {}, {}, {}
    preds = val[["Order ID", "Date", "cancelled"]].reset_index(drop=True)
    for group, y_fit, y_eval in [(baseline, y_train, y_val), (smote_models, y_sm, y_val_m)]:
        for name, (model, X_fit, X_eval) in group.items():
            log(f"  [Exp 1.1] training {name} ...")
            model.fit(X_fit, y_fit)
            y_pred = model.predict(X_eval)
            y_prob = model.predict_proba(X_eval)[:, 1]
            ev = evaluate(np.asarray(y_eval), y_pred, y_prob)
            rows.append({"Model": name, **ev["metrics"]})
            cms[name], reports[name] = ev["confusion_matrix"], ev["classification_report"]
            preds[f"prob__{name}"] = y_prob
            preds[f"pred__{name}"] = np.asarray(y_pred).astype(int)
            models[name] = model

    preds = preds.merge(val[["Order ID", "Fulfilment", "Sales_Channel", "Ship_Service_Level",
                             "B2B", "Ship_State_Grouped", "Item_Count"]], on="Order ID", how="left")
    split_summary = pd.DataFrame([
        _split_row("Train (April, original)", train),
        {**_split_row("Train (April, undersampled 65:35)", balanced)},
        {"split": "Train (April, SMOTENC 70:30)", "orders": len(y_sm), "cancelled": int(y_sm.sum()),
         "cancel_rate_pct": y_sm.mean() * 100, "start": train["Date"].min(), "end": train["Date"].max()},
        _split_row("Validation (May)", val),
        _split_row("Test (June, not evaluated in notebook)", test),
    ])
    return ExperimentResult(
        name="Experiment 1.1", eval_split="Validation (May 2022)",
        metrics=pd.DataFrame(rows),
        confusion_matrices=cms, reports=reports, predictions=preds,
        split_summary=split_summary,
        models={"encoder": encoder, "smote_encoder": sm_encoder, "scaler": scaler, **models},
        extras={"features": features, "april_features": april, "rare_states": rare_states},
    )
