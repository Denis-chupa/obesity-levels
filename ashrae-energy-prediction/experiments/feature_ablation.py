"""Feature ablation on a global LightGBM, 3 month blocks (results/feature_ablation.csv).

Run from the project root after prepare_data.py and the validation stage (needs
research.ipynb and artifacts/boosting_config.joblib):  python experiments/feature_ablation.py
"""
import json, time, joblib, numpy as np, pandas as pd, lightgbm
from sklearn.preprocessing import StandardScaler
nb = json.load(open("research.ipynb", encoding="utf-8")); by = {c["id"]: "".join(c["source"]) for c in nb["cells"]}
ns = {"np": np, "pd": pd, "StandardScaler": StandardScaler}
ns.update(joblib.load("artifacts/boosting_config.joblib"))
for cid in ["resume-common", "77f0fdc2", "6e5ba85e"]:
    exec(by[cid], ns)
tf = pd.read_parquet("artifacts/train_full.parquet")
bad = (((tf.site_id == 0) & (tf.meter == 0) & (tf.timestamp < ns["BAD_SITE0_END"])) | tf.constant_run).to_numpy()
month = tf.timestamp.dt.month.to_numpy()
schemes = {
    "blocks of 4 months": [np.isin(month, m) for m in ([1,2,3,4], [5,6,7,8], [9,10,11,12])],
    "interleaved months": [(month - 1) % 3 == k for k in range(3)],
}
WIN = ["raw_air_temperature_mean_3h", "raw_air_temperature_mean_24h", "raw_air_temperature_lag_24h", "raw_air_temperature_diff_72h"]
CAL = ["is_holiday", "month", "dayofyear"]
rng = np.random.default_rng(42)
rows = []
for scheme, folds in schemes.items():
    for fi, held in enumerate(folds):
        fit_pos = np.flatnonzero(~held & ~bad); fit_pos = np.sort(rng.choice(fit_pos, 1_000_000, replace=False))
        ev_pos = np.flatnonzero(held); ev_pos = np.sort(rng.choice(ev_pos, 300_000, replace=False))
        fit, ev = tf.iloc[fit_pos].reset_index(drop=True), tf.iloc[ev_pos].reset_index(drop=True)
        scaler = StandardScaler()
        for s in range(0, len(fit), 50_000): scaler.partial_fit(ns["cyclic_numeric_batch"](fit.iloc[s:s+50_000]))
        cmap = {c: {v: i for i, v in enumerate(fit[c].dropna().unique())} for c in ns["BOOSTING_CATEGORICAL"]}
        Xf, Xe = ns["make_boosting_features"](fit, scaler, cmap), ns["make_boosting_features"](ev, scaler, cmap)
        yf, ye = np.log1p(fit.meter_reading.to_numpy(np.float64)), np.log1p(ev.meter_reading.to_numpy(np.float64))
        base = [c for c in Xf.columns if c not in WIN + CAL]
        sets = {"current": base, "+windows": base + WIN, "+windows+holiday": base + WIN + ["is_holiday"],
                "+windows+holiday+month": base + WIN + ["is_holiday", "month"],
                "+windows+holiday+dayofyear": base + WIN + ["is_holiday", "dayofyear"],
                "+windows+holiday+month+dayofyear": base + WIN + CAL}
        for name, cols in sets.items():
            for label, params in [("100 trees", {}), ("400 trees, 127 leaves", {"n_estimators": 400, "num_leaves": 127, "learning_rate": 0.05})]:
                if label != "100 trees" and scheme != "blocks of 4 months": continue
                mdl = lightgbm.LGBMRegressor(**{**ns["LGBM_PARAMS"], "n_jobs": 4, **params}).fit(Xf[cols], yf, categorical_feature=ns["BOOSTING_CATEGORICAL"])
                rows.append({"scheme": scheme + ", " + label, "features": name, "fold": fi, "rmse": float(np.sqrt(np.mean((np.maximum(mdl.predict(Xe[cols]), 0) - ye) ** 2)))})
        print(scheme, fi, "done", flush=True)
df = pd.DataFrame(rows)
for scheme, part in df.groupby("scheme", sort=False):
    t = part.pivot(index="features", columns="fold", values="rmse").loc[list(sets)]; t["mean"] = t.mean(axis=1)
    print("\n" + scheme); print(t.round(4).to_string())
