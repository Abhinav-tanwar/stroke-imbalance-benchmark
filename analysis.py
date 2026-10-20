"""Training-only mixed-type resampling with explicit minority-class metrics."""

import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler, SMOTENC
from imblearn.under_sampling import TomekLinks
from utils import ROOT, save, clf


def run():
    raw = pd.read_csv(ROOT / "data/input.csv")
    d = raw.drop_duplicates()
    y = d.pop("stroke").astype(int)
    X = d
    cat = list(X.select_dtypes(exclude=np.number).columns)
    num = [c for c in X if c not in cat]
    groups = pd.util.hash_pandas_object(X, index=False).to_numpy()
    a, b = next(
        StratifiedGroupKFold(5, shuffle=True, random_state=24).split(X, y, groups)
    )
    train = X.iloc[a]
    test = X.iloc[b]
    yt = y.iloc[a]
    yv = y.iloc[b]

    def prepare():
        return ColumnTransformer(
            [
                (
                    "numeric",
                    Pipeline(
                        [
                            ("impute", SimpleImputer(strategy="median")),
                            ("scale", StandardScaler()),
                        ]
                    ),
                    num,
                ),
                (
                    "categorical",
                    Pipeline(
                        [
                            ("impute", SimpleImputer(strategy="most_frequent")),
                            (
                                "ordinal",
                                OrdinalEncoder(
                                    handle_unknown="use_encoded_value", unknown_value=-1
                                ),
                            ),
                        ]
                    ),
                    cat,
                ),
            ]
        )

    catidx = list(range(len(num), len(num) + len(cat)))
    encode = lambda: ColumnTransformer(
        [
            ("num", "passthrough", list(range(len(num)))),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                catidx,
            ),
        ]
    )
    variants = {
        "Logistic": ("passthrough", LogisticRegression(max_iter=2000)),
        "Weighted_logistic": (
            "passthrough",
            LogisticRegression(max_iter=2000, class_weight="balanced"),
        ),
        "Random_oversampling": (
            RandomOverSampler(random_state=24),
            LogisticRegression(max_iter=2000),
        ),
        "Tomek_links": (TomekLinks(), LogisticRegression(max_iter=2000)),
        "Mixed_type_SMOTE": (
            SMOTENC(categorical_features=catidx, random_state=24),
            LogisticRegression(max_iter=2000),
        ),
        "Random_forest": (
            "passthrough",
            RandomForestClassifier(
                n_estimators=100,
                max_depth=12,
                min_samples_leaf=5,
                n_jobs=2,
                random_state=24,
            ),
        ),
    }
    result = dict(
        rows=len(X),
        duplicates_removed=len(raw) - len(X),
        train_rows=len(a),
        test_rows=len(b),
        test_positive=int(yv.sum()),
        test_prevalence=float(yv.mean()),
        majority_accuracy=float(max(yv.mean(), 1 - yv.mean())),
        threshold=0.5,
        group_overlap=int(len(set(groups[a]) & set(groups[b]))),
        models={},
    )
    for name, (sampler, model) in variants.items():
        stages = [("prepare", prepare())]
        if name == "Tomek_links":
            stages.extend([("encode", encode()), ("sample", sampler)])
        else:
            stages.extend([("sample", sampler), ("encode", encode())])
        stages.append(("model", model))
        pipe = ImbPipeline(stages)
        pipe.fit(train, yt)
        prob = pipe.predict_proba(test)[:, 1]
        result["models"][name] = clf(yv, prob >= 0.5, prob)
    return save(result)


if __name__ == "__main__":
    run()
