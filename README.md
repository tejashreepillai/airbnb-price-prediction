# Melbourne Airbnb Price Prediction

An independent machine-learning rebuild that predicts nightly Airbnb listing prices in Melbourne using reproducible, leakage-safe preprocessing and model evaluation.

## Project Summary

This project uses listing capacity, location, property characteristics, host information, availability and review history to estimate nightly prices. The final model was developed as decision support for mainstream listings rather than as an automatic pricing system for unusual or luxury properties.

The analysis compares a median baseline, Ridge Regression, Random Forest and Gradient Boosting. Models are evaluated using mean absolute error (MAE) on the original dollar scale.

[View the complete analysis notebook](notebooks/airbnb_price_model.ipynb)

## Results

| Model                                | Validation MAE |
| ------------------------------------ | -------------: |
| Median baseline                      |        $198.88 |
| Log-target Ridge Regression          |        $159.30 |
| Log-target Random Forest             |        $150.20 |
| Untuned log-target Gradient Boosting |        $144.53 |
| Tuned log-target Gradient Boosting   |    **$144.10** |

The tuned model reduced validation MAE by approximately **27.5%** compared with the median baseline.

Five-fold cross-validation produced:

| Model                     | Mean CV MAE |
| ------------------------- | ----------: |
| Tuned Gradient Boosting   | **$111.52** |
| Untuned Gradient Boosting |     $112.81 |
| Random Forest             |     $116.37 |
| Ridge Regression          |     $126.46 |

Hyperparameter tuning improved holdout MAE by only $0.43. This suggests that the preprocessing and model choice contributed more than the final tuning step.

## Performance by Price Range

The target contained a small number of exceptionally high prices:

* 95% of training listings were priced at or below $555
* 99% were priced at or below approximately $1,106
* Six listings above $2,000 generated 64.89% of the tuned model’s total validation error
* For the 1,389 validation listings priced at or below approximately $1,106, MAE was **$48.33**

The available data could not establish whether the extreme prices were genuine luxury listings or data errors. They were retained in the primary evaluation and reported separately through sensitivity analysis.

## Strongest Predictors

Permutation importance identified the following as the most useful predictors for mainstream listings:

1. Bedrooms
2. Guest capacity
3. Bathrooms
4. Neighbourhood
5. Thirty-day availability
6. Room type
7. Review value score
8. Longitude

These are predictive relationships rather than causal effects. Correlated variables may share importance.

## Data Preparation

Key preparation improvements included:

* Converting currency and percentage text into numeric values
* Extracting numeric bathroom quantities, including half-bath descriptions
* Creating reproducible host-tenure and review-recency features using a fixed reference date
* Safely parsing amenities and host-verification lists with `ast.literal_eval`
* Counting actual verification methods rather than characters in stored list strings
* Creating word-count features from listing, neighbourhood and host descriptions
* Converting boolean fields to numeric indicators
* Learning imputation and category mappings from training data only
* Applying preprocessing independently inside every cross-validation fold
* Grouping infrequent categorical values and safely handling unseen categories

## Model Development

The modelling process included:

* An 80/20 stratified training and validation split
* A median-price baseline
* Log-target Ridge Regression
* Log-target Random Forest
* Log-target Histogram Gradient Boosting
* Five-fold cross-validation
* Randomised hyperparameter search across 100 fitted models
* Holdout evaluation and price-band sensitivity analysis
* Permutation-based feature importance

The selected Gradient Boosting model used:

* Learning rate: `0.1`
* Maximum iterations: `600`
* Maximum leaf nodes: `31`
* Minimum samples per leaf: `10`
* L2 regularisation: `1.0`

## Project Scope and Authorship

This is an independent post-course rebuild of a Melbourne Airbnb project originally completed as a university group assignment.

In the original assignment, my assessed responsibility covered:

* Numerical-data cleaning
* Feature engineering
* Missing-value treatment
* Categorical encoding

All model development in this repository—including validation design, baseline comparison, three regression models, tuning, outlier analysis and feature-importance interpretation—was completed independently after the course.

The original group notebook and my teammates’ work are not included or represented as my individual contribution.

## Data Access and Privacy

The supplied files contained 7,000 labelled training records, 3,000 unlabelled competition records and 60 original predictors.

The data originate from the [ASBA Predictive Analytics Competition](https://www.kaggle.com/competitions/asba-predictive-analytics-competition/data). Raw competition data and generated prediction files are intentionally excluded from this repository. Authorised users must obtain the files directly through Kaggle.

Expected private files:

```text
data/
├── train.csv
├── test.csv
├── metaData.csv
└── sample-solution.csv
```

## Repository Structure

```text
airbnb-price-prediction/
├── notebooks/
│   └── airbnb_price_model.ipynb
├── .gitignore
├── README.md
└── requirements.txt
```

The private `data/` and `outputs/` directories are excluded by `.gitignore`.

## Running the Project

Clone the repository:

```bash
git clone https://github.com/tejashreepillai/airbnb-price-prediction.git
cd airbnb-price-prediction
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS or Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Download the authorised competition files from Kaggle, place them inside `data/`, and start Jupyter:

```bash
jupyter notebook
```

Open `notebooks/airbnb_price_model.ipynb` and run all cells.

## Limitations

* Extreme target prices could not be independently verified.
* The validation set was inspected during preliminary model comparison, so its final score is not equivalent to a completely untouched external test.
* Competition test labels are unavailable, preventing an independently verified test score.
* Final predictions ranged from $40.69 to $956.59, indicating limited suitability for exceptionally expensive listings.
* Historical Melbourne relationships may not generalise to newer periods, other cities or changed market conditions.
* Free text was represented using word counts rather than full natural-language modelling.
* Feature importance does not establish causation.

A stronger production evaluation would use newer externally labelled listings, time-based testing and a separate strategy for exceptionally expensive properties.
