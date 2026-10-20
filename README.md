# Stroke Imbalance Benchmark

Mixed-type resampling and minority-class evaluation with leakage-controlled pipelines.

**Author and maintainer:** Abhinav Tanwar.

## Run

Requires Python 3.12. Install dependencies with `python -m pip install -r requirements.txt`.

```sh
python analysis.py
python -m unittest discover -s tests
```

## Methods and limitations

The supplied 43,400-row dataset was located beside the notebook. ID is removed, and identical predictor profiles are kept within the same split with StratifiedGroupKFold. Numeric imputation/scaling and categorical encoding are fitted on training rows. SMOTENC generates mixed-type samples before final one-hot encoding; it does not interpolate one-hot categories. Baseline, class weighting, random oversampling, Tomek links, mixed-type SMOTE, and random forest share the same test records. Tomek cleaning runs after one-hot encoding so nominal categories are not treated as ordered distances. Default threshold is 0.5; high accuracy can coexist with zero positive recall. Average precision must be assessed against the test prevalence. This retrospective label benchmark does not predict individual future clinical risk.

## Measured results

- **Logistic:** roc auc = 0.8692; average precision = 0.1016; recall = 0; precision = 0; accuracy = 0.9819.
- **Weighted logistic:** roc auc = 0.8678; average precision = 0.1008; recall = 0.8471; precision = 0.05532; accuracy = 0.7356.
- **Random oversampling:** roc auc = 0.8675; average precision = 0.1011; recall = 0.8471; precision = 0.05516; accuracy = 0.7348.
- **Tomek links:** roc auc = 0.8692; average precision = 0.1016; recall = 0; precision = 0; accuracy = 0.9819.
- **Mixed type SMOTE:** roc auc = 0.8632; average precision = 0.1024; recall = 0.828; precision = 0.05556; accuracy = 0.7423.
- **Random forest:** roc auc = 0.8508; average precision = 0.07937; recall = 0; precision = 0; accuracy = 0.9819.

[Full calculated metrics](results/metrics.json) · [Input provenance](data/README.md) · [Method changes](docs/methodology.md). Open `report.html` for the visual summary.

The clean notebook in `notebooks/` delegates to the same module. Notebook cells were syntax-checked with outputs cleared; native scripts and tests were executed.
