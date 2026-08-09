# 📊 House Price Prediction --- Model Evaluation Report

> **Task 9 --- Week 9: Introduction to Machine Learning Concepts**

## 📌 Project Overview

This project presents an end-to-end machine learning workflow for
predicting residential house prices using supervised regression
techniques.

The workflow covers data validation, exploratory data analysis, feature
preprocessing, model development, model comparison, hyperparameter
optimization, evaluation, residual diagnostics, feature-importance
analysis, and interpretation of the final model.

The primary objective is to develop a reliable regression model that
predicts house prices from property characteristics while following
sound machine-learning practices such as train-test separation,
leakage-safe preprocessing, cross-validation, and evaluation on unseen
data.

### Project Highlights

-   300 residential property observations
-   Target variable: `Price`
-   Numerical predictors: `Area`, `Bedrooms`, `Bathrooms`, `Age`
-   Categorical predictors: `Location`, `Property_Type`
-   80:20 train-test split
-   Linear Regression implemented from scratch
-   Scikit-learn used for ML implementation and validation
-   Polynomial Regression
-   Decision Tree Regression
-   Random Forest Regression
-   Random Forest hyperparameter optimization using 5-fold
    cross-validation
-   Evaluation using MAE, MSE, RMSE, and R²
-   Permutation Feature Importance
-   Actual-vs-predicted analysis
-   Residual diagnostics
-   Final model selection based on unseen test-set performance

------------------------------------------------------------------------

## 🎯 Objectives

1.  Understand and validate the structure and quality of the house-price
    dataset.
2.  Explore relationships between property characteristics and price.
3.  Identify influential variables associated with house prices.
4.  Prepare numerical and categorical variables for machine learning.
5.  Implement Multiple Linear Regression mathematically from scratch.
6.  Validate the custom implementation against Scikit-learn.
7.  Compare linear and nonlinear regression approaches.
8.  Optimize Random Forest hyperparameters using cross-validation.
9.  Evaluate models using MAE, MSE, RMSE, and R².
10. Diagnose the final model using residual analysis.
11. Select the strongest model using unseen test data.
12. Translate statistical and machine-learning findings into practical
    insights.

------------------------------------------------------------------------

# 📚 Dataset Description

The project uses a residential property dataset containing **300
observations**.

### Target Variable

`Price`

The models learn the relationship between property characteristics and
selling price.

### Predictor Variables

  Type          Feature           Description
  ------------- ----------------- -------------------------------------
  Numerical     `Area`            Size of the property
  Numerical     `Bedrooms`        Number of bedrooms
  Numerical     `Bathrooms`       Number of bathrooms
  Numerical     `Age`             Age of the property
  Categorical   `Location`        Geographic/location category
  Categorical   `Property_Type`   Type/category of property
  Identifier    `Property_ID`     Identifier; excluded from modelling

`Property_ID` is excluded because it is an identifier rather than a
meaningful predictive characteristic.

------------------------------------------------------------------------

# 🔄 Methodology

``` text
Raw Dataset
     ↓
Data Loading & Validation
     ↓
Data Quality Checks
     ↓
Exploratory Data Analysis
     ↓
Feature / Target Separation
     ↓
Train-Test Split
     ↓
Leakage-Safe Preprocessing
     ↓
Model Training
     ↓
Model Evaluation
     ↓
Random Forest Hyperparameter Optimization
     ↓
Final Model Evaluation
     ↓
Feature Importance
     ↓
Residual Diagnostics
     ↓
Model Interpretation
     ↓
Conclusion
```

## 1. Data Loading and Validation

The dataset was loaded using Pandas and inspected for:

-   Number of observations
-   Data types
-   Missing values
-   Duplicate records
-   Numerical and categorical variables
-   Target-variable characteristics

A preprocessing summary was exported to:

`results/preprocessing_summary.csv`

## 2. Exploratory Data Analysis

EDA examined:

-   Price distribution
-   Area vs. Price
-   Bedrooms and Bathrooms vs. Price
-   Location and Property Type distribution
-   Age vs. Price
-   Numerical correlation matrix
-   Area-Price relationship by location

The resulting visualizations are stored in the `visualizations/`
directory.

## 3. Train-Test Split

The dataset was divided into:

  Dataset      Observations   Percentage
  ---------- -------------- ------------
  Training              240          80%
  Testing                60          20%
  Total                 300         100%

The test set was kept separate for final evaluation.

## 4. Leakage-Safe Preprocessing

Categorical variables were transformed using one-hot encoding.

Reference categories were used:

-   `City Center` → Location reference
-   `Apartment` → Property Type reference

The resulting encoded categorical variables include:

-   `Location_Rural`
-   `Location_Suburb`
-   `Property_Type_House`
-   `Property_Type_Villa`

Together with:

-   `Area`
-   `Bedrooms`
-   `Bathrooms`
-   `Age`

This produces **8 model features** after preprocessing.

Preprocessing transformations were fitted on training data only and then
applied to the test data.

------------------------------------------------------------------------

# 🤖 Model Development

## 1. Linear Regression --- From Scratch

Multiple Linear Regression was implemented manually using the Ordinary
Least Squares approach.

The implementation uses a design matrix and estimates regression
coefficients mathematically. The Moore-Penrose pseudoinverse was used
for numerical stability.

This demonstrates understanding of the mathematical foundation of Linear
Regression rather than relying exclusively on a library implementation.

## 2. Linear Regression --- Scikit-learn

Scikit-learn's `LinearRegression` was trained using the same processed
data.

The custom implementation was compared with the Scikit-learn
implementation using:

-   MAE
-   MSE
-   RMSE
-   R²

The predictions were effectively equivalent, validating the custom
implementation.

Results:

`results/scratch_vs_sklearn_comparison.csv`

## 3. Polynomial Regression

A degree-2 Polynomial Regression model was used to investigate whether
nonlinear relationships could improve predictive performance.

The degree-2 model achieved:

-   RMSE ≈ **₹3.18 million**
-   R² = **0.9290**

It did not outperform Linear Regression on the held-out test set.

## 4. Decision Tree Regression

A controlled Decision Tree was used to model nonlinear relationships and
feature interactions.

Configuration:

  Parameter                 Value
  ----------------------- -------
  Maximum Depth                 6
  Minimum Samples Split        10
  Minimum Samples Leaf          5
  Random State                 42

Performance:

-   RMSE ≈ **₹3.29 million**
-   R² = **0.9238**

## 5. Initial Random Forest Regression

An initial Random Forest was evaluated before optimization.

Performance:

-   RMSE ≈ **₹4.81 million**
-   R² = **0.8372**

This demonstrated that an ensemble model does not automatically perform
better without suitable hyperparameter configuration.

------------------------------------------------------------------------

# ⚙️ Hyperparameter Optimization

Random Forest hyperparameters were optimized using randomized search
with **5-fold cross-validation**.

The test set remained untouched during optimization.

The best configuration was:

  Hyperparameter             Value
  ----------------------- --------
  Number of Trees              300
  Maximum Depth                 12
  Minimum Samples Split          2
  Minimum Samples Leaf           2
  Max Features                 1.0
  Random State                  42
  Cross-Validation          5-Fold

The optimized configuration substantially improved the initial Random
Forest performance.

------------------------------------------------------------------------

# 📈 Model Evaluation

The final models were evaluated on the unseen test set using MAE, MSE,
RMSE, and R².

### Metric Interpretation

**MAE:** Average absolute prediction error. Lower is better.

**MSE:** Average squared prediction error. Lower is better and large
errors receive greater weight.

**RMSE:** Square root of MSE, expressed in the same monetary unit as
price. Lower is better.

**R²:** Proportion of target variation explained by the model. Higher is
better.

## Model Comparison

  Model                                 RMSE           R²
  ----------------------------- ------------ ------------
  **Optimized Random Forest**     **₹2.22M**   **0.9654**
  Linear Regression               **₹2.91M**   **0.9406**
  Polynomial Regression           **₹3.18M**   **0.9290**
  Decision Tree                   **₹3.29M**   **0.9238**
  Initial Random Forest           **₹4.81M**   **0.8372**

## R² Comparison

![R² Model Comparison](visualizations/model_r2_comparison.png)

The Optimized Random Forest achieved the highest R² score.

## RMSE Comparison

![RMSE Model Comparison](visualizations/model_rmse_comparison.png)

The Optimized Random Forest achieved the lowest RMSE.

------------------------------------------------------------------------

# 🏆 Final Model Performance

## Optimized Random Forest Regression

  Metric                Final Test Score
  ---------- ---------------------------
  **MAE**              **₹1,611,259.09**
  **MSE**      **₹4,923,749,816,934.83**
  **RMSE**             **₹2,218,952.41**
  **R²**                      **0.9654**

An R² of **0.9654** means the final model explains approximately
**96.54% of the variation in house prices within the held-out test
dataset**.

The MAE of approximately **₹1.61 million** represents the average
absolute prediction error.

The RMSE of approximately **₹2.22 million** represents the overall scale
of prediction error, with larger errors receiving greater influence.

------------------------------------------------------------------------

# 🎯 Model Diagnostics

## 1. Actual vs. Predicted Prices

![Actual vs Predicted House
Prices](visualizations/actual_vs_predicted.png)

The predictions are generally concentrated around the perfect-prediction
diagonal, indicating strong agreement between actual and predicted
prices.

A small number of observations show larger deviations and therefore
contribute more strongly to the RMSE.

## 2. Residual Distribution

![Residual Distribution](visualizations/residual_distribution.png)

Residuals are generally distributed around zero.

This suggests that the model does not show a strong overall tendency to
consistently overpredict or underpredict.

## 3. Residuals vs. Predicted Price

![Residuals vs Predicted
Price](visualizations/residuals_vs_predicted.png)

Residuals occur on both sides of zero without a pronounced systematic
curve.

Some increase in error spread is visible at higher predicted prices,
suggesting that expensive properties can be somewhat harder to predict
accurately.

------------------------------------------------------------------------

# 🧠 Feature Importance

Permutation Feature Importance was used to evaluate how much predictive
performance changes when individual features are randomly shuffled.

![Permutation Feature
Importance](visualizations/permutation_feature_importance.png)

### Main Findings

1.  **Area** is the strongest predictive feature.
2.  **Location** has substantial predictive importance.
3.  **Bedrooms** provide moderate additional predictive information.
4.  **Age** contributes relatively little compared with the strongest
    variables.
5.  **Bathrooms** and **Property Type** show very low standalone
    permutation importance in this dataset.

The complete results are available in:

`results/permutation_feature_importance.csv`

> Permutation importance indicates predictive contribution within this
> model and dataset. It does not establish causation.

------------------------------------------------------------------------

# 📐 Linear Regression Coefficients

![Linear Regression
Coefficients](visualizations/linear_regression_coefficients.png)

Linear Regression coefficients provide directional information relative
to the reference categories.

-   Location effects are interpreted relative to `City Center`.
-   Property Type effects are interpreted relative to `Apartment`.

The complete coefficient output is stored in:

`results/linear_regression_coefficients.csv`

Coefficient magnitude should not be treated as a universal
feature-importance ranking because features have different units and
scales.

------------------------------------------------------------------------

# 💡 Key Findings

1.  **Area is the strongest numerical predictor of house price.**
2.  **Location has substantial pricing influence.**
3.  **Bedrooms have a moderate positive relationship with price.**
4.  **Age has a weak negative relationship with price.**
5.  **Bathrooms have limited standalone linear association with price.**
6.  **Nonlinear models do not automatically perform better.**
7.  **Hyperparameter optimization substantially improved Random Forest
    performance.**
8.  **The Optimized Random Forest was the strongest model on the
    held-out test set.**

The EDA and correlation analysis also indicate an approximate Pearson
correlation of **0.80 between Area and Price**, reinforcing the
importance of property size in this dataset.

------------------------------------------------------------------------

# 💼 Practical Interpretation

The analysis suggests that property area and location should receive
particular attention when estimating house prices within this dataset.

The optimized Random Forest can capture nonlinear relationships and
interactions that may not be represented by a simple linear model.

However, predictions should be treated as estimates rather than exact
professional valuations.

------------------------------------------------------------------------

# ⚠️ Limitations

-   The dataset contains only 300 observations.
-   Available features do not represent every factor affecting property
    prices.
-   Important variables such as amenities, schools, hospitals,
    transportation, parking, and neighborhood characteristics are
    absent.
-   Economic and housing-market trends are not represented.
-   Evaluation uses one held-out test set.
-   Results may not generalize to other cities, markets, or time
    periods.
-   Some high-priced properties show larger prediction errors.

------------------------------------------------------------------------

# 🚀 Future Scope

Possible improvements include:

-   Collecting a larger and more diverse dataset.
-   Adding geographic and neighborhood features.
-   Including distances to schools, hospitals, transport, and commercial
    areas.
-   Adding market-time and economic indicators.
-   Testing Gradient Boosting, XGBoost, LightGBM, and Extra Trees.
-   Performing broader hyperparameter optimization.
-   Using repeated cross-validation.
-   Applying SHAP or other explainable-AI methods.
-   Building a web-based prediction interface.
-   Deploying the model as an API.
-   Monitoring model performance after deployment.

------------------------------------------------------------------------

# 🛠️ Setup and Installation

Install the required dependencies:

``` bash
pip install -r requirements.txt
```

Launch Jupyter:

``` bash
jupyter notebook
```

Open:

``` text
house_price_prediction.ipynb
```

Alternatively, open the notebook in VS Code with the Jupyter extension.

Run the notebook from top to bottom.

The notebook will:

1.  Load the dataset.
2.  Validate and preprocess the data.
3.  Generate EDA outputs.
4.  Train the regression models.
5.  Optimize Random Forest.
6.  Evaluate the final model.
7.  Generate diagnostics and visualizations.
8.  Save analysis results into `results/`.

------------------------------------------------------------------------

# 📁 Project Structure

``` text
HOUSE-PRICE-PREDICTION/
│
├── house_price_prediction.ipynb
├── house_data.csv
├── house_prices.csv
├── README.md
├── model_evaluation_report.md
├── requirements.txt
│
├── results/
│   ├── preprocessing_summary.csv
│   ├── numerical_correlation_matrix.csv
│   ├── age_price_correlation.csv
│   ├── area_price_correlation.csv
│   ├── area_price_correlation_by_location.csv
│   ├── bedrooms_bathrooms_price_analysis.csv
│   ├── location_property_type_price_analysis.csv
│   ├── linear_regression_scratch_results.csv
│   ├── scratch_vs_sklearn_comparison.csv
│   ├── linear_regression_coefficients.csv
│   ├── permutation_feature_importance.csv
│   ├── polynomial_regression_results.csv
│   ├── decision_tree_results.csv
│   ├── random_forest_results.csv
│   └── optimized_random_forest_results.csv
│
└── visualizations/
    ├── 01_price_distribution.png
    ├── 02_area_vs_price.png
    ├── 03_bedrooms_bathrooms_vs_price.png
    ├── 04_location_property_type_distribution.png
    ├── 05_age_vs_price.png
    ├── 06_correlation_heatmap.png
    ├── 07_area_price_by_location.png
    ├── actual_vs_predicted.png
    ├── linear_regression_coefficients.png
    ├── model_r2_comparison.png
    ├── model_rmse_comparison.png
    ├── permutation_feature_importance.png
    ├── residual_distribution.png
    └── residuals_vs_predicted.png
```

------------------------------------------------------------------------

# 🔐 Reproducibility and Data-Leakage Controls

The project follows these practices:

-   Fixed `random_state = 42` where applicable.
-   Separate training and testing datasets.
-   Test data kept unseen during training.
-   Preprocessing fitted on training data only.
-   Learned transformations applied to test data.
-   Hyperparameter optimization performed using training data and
    cross-validation.
-   Test data excluded from hyperparameter selection.
-   Final model evaluated only after model selection.
-   Consistent metrics used across models.
-   Numerical results exported as CSV files.
-   Visualizations exported as PNG files.

These controls reduce evaluation bias and help ensure that the final
performance represents an independent test-set evaluation.

------------------------------------------------------------------------

# 📦 Output Artifacts

The project produces two main categories of artifacts.

### `results/`

Machine-readable CSV outputs containing preprocessing summaries,
correlations, model results, coefficients, and feature-importance
results.

### `visualizations/`

PNG visualizations covering EDA, model comparison, prediction accuracy,
feature importance, and residual diagnostics.

------------------------------------------------------------------------

# 🧪 Requirement Coverage

  Requirement                      Status
  -------------------------------- -----------------------
  Scikit-learn ML implementation   ✅
  Proper train-test split          ✅
  At least 3 evaluation metrics    ✅ MAE, MSE, RMSE, R²
  Multiple regression models       ✅
  Cross-validation                 ✅ 5-Fold
  Hyperparameter optimization      ✅
  Model evaluation                 ✅
  Model interpretation             ✅
  Residual diagnostics             ✅
  Project documentation            ✅
  Reproducibility controls         ✅
  Saved CSV results                ✅
  Saved PNG visualizations         ✅

------------------------------------------------------------------------

# 🏁 Final Conclusion

This project demonstrates a complete and reproducible machine-learning
workflow for residential house-price prediction.

The workflow progresses from data validation and exploratory analysis
through preprocessing, model development, comparison, hyperparameter
optimization, evaluation, diagnostics, and interpretation.

The **Optimized Random Forest Regression** model achieved the strongest
performance on the unseen test dataset:

-   **R² = 0.9654**
-   **RMSE = ₹2,218,952.41**
-   **MAE = ₹1,611,259.09**

The results demonstrate that careful preprocessing, appropriate
evaluation methodology, model comparison, and systematic hyperparameter
optimization can substantially improve predictive performance.

> **Disclaimer:** This model is intended for educational and analytical
> purposes. Predictions should be treated as estimates and should not be
> considered professional property valuations.

------------------------------------------------------------------------

## 👤 Author

**Vaibhav Shukla**

**BCA Graduate \| Data Analytics & Machine Learning**
