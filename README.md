# Predicting Airbnb Listing Prices

A data-analysis project that predicts the log-price of an Airbnb listing from its
characteristics, comparing a linear regression with a support vector regressor. The work
covers the full workflow: exploration, cleaning, preprocessing, modelling and tuning.

## Data

The training set holds about 22,000 labelled listings; a separate test set of roughly 52,000
listings is used to produce predictions. Each listing is described by 28 columns: property and
room type, capacity (accommodates, bathrooms, bedrooms, beds), bed and cancellation policy,
location (city, neighbourhood), review scores, host response rate, and several boolean host
attributes. The target is the log-price of the listing, and the deliverable is a submission file
pairing each test id with its predicted log-price.

## Workflow

Exploration first checks for missing values and duplicates. Cleaning maps the textual
true/false fields to booleans, parses the host response rate, and drops columns judged
uninformative for price (identifiers, raw text fields, geographic coordinates, review dates).
A correlation analysis against the target guides feature selection.

Preprocessing is handled by a `ColumnTransformer` pipeline: numeric columns are imputed and
standardized, categorical columns are one-hot encoded, and boolean columns are passed through.
The data is then split 80/20 for training and testing.

## Models and results

Two models are compared. A linear regression provides a baseline; a support vector regressor
with an RBF kernel is then tuned with grid search over its `C` and `gamma` parameters.

| Model | R² | RMSE |
|---|---|---|
| Linear regression | 0.63 | 0.44 |
| SVR (RBF) | 0.65 | 0.43 |
| SVR (tuned: C=10, gamma=0.01) | 0.65 | 0.42 |

The tuned SVR gives the best fit, explaining about 65% of the variance in log-price. The gain
over the linear baseline is modest, suggesting that the relationship is close to linear once
the categorical features are properly encoded.

## Running the project

The analysis is a Jupyter notebook. Dependencies:

```
pip install pandas numpy scikit-learn matplotlib
```

The notebook expects training and test data. The file paths are currently hard-coded to a
local machine and should be changed to a relative location (for example a `data/` folder next
to the notebook) before running elsewhere.

Because the raw CSVs are large, they are stored compressed (`airbnb_train.csv.gz`,
`airbnb_test.csv.gz`). `pandas` reads them directly, with no manual decompression:

```python
train = pd.read_csv("data/airbnb_train.csv.gz")
test  = pd.read_csv("data/airbnb_test.csv.gz")
```

## Possible improvements

- Use a randomized train/test split or cross-validation rather than slicing the rows in order.
- Try a gradient-boosting model, which often improves on SVR for tabular data of this kind.
- Encode the `amenities` field (currently dropped), which likely carries price information.
