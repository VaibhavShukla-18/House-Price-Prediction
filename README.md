# 🏠 House Price Prediction — Advanced Machine Learning Project

<p align="center">
  <img src="visualizations/actual_vs_predicted.png" alt="Actual vs Predicted House Prices" width="850">
</p>

<p align="center">
  <b>End-to-End House Price Prediction using Regression, Ensemble Learning, Hyperparameter Optimization & Model Diagnostics</b>
</p>

---

## 📌 Project Overview

This project develops an end-to-end machine learning workflow for predicting residential property prices.

The project goes beyond a basic regression implementation by combining:

- Data loading and validation
- Data quality checks
- Exploratory Data Analysis (EDA)
- Correlation and relationship analysis
- Categorical feature encoding
- Train-test splitting
- Leakage-safe preprocessing
- Linear Regression implemented **from scratch**
- Validation against Scikit-learn Linear Regression
- Polynomial Regression
- Decision Tree Regression
- Random Forest Regression
- Random Forest hyperparameter optimization using **5-fold cross-validation**
- Permutation Feature Importance
- Actual vs. Predicted analysis
- Residual diagnostics
- R² and RMSE model comparison
- Business interpretation
- Limitations and future scope

The objective is not only to obtain a high prediction score, but also to demonstrate a **complete, reproducible and interpretable machine learning workflow**.

---

## 🎯 Objectives

The major objectives of this project are:

1. Understand the structure and quality of the house-price dataset.
2. Explore relationships between property characteristics and price.
3. Identify the most influential variables associated with house prices.
4. Prepare numerical and categorical variables for machine learning.
5. Implement Multiple Linear Regression mathematically from scratch.
6. Validate the custom implementation against Scikit-learn.
7. Compare linear and nonlinear regression approaches.
8. Optimize Random Forest hyperparameters using cross-validation.
9. Evaluate models using MAE, MSE, RMSE and R².
10. Diagnose the final model using residual analysis.
11. Select the best-performing model based on unseen test data.
12. Translate statistical and machine-learning findings into practical insights.

---

## 📊 Dataset

The project uses a residential property dataset containing:

- **300 observations**
- **Price** as the prediction target
- Numerical predictors:
  - Area
  - Bedrooms
  - Bathrooms
  - Age
- Categorical predictors:
  - Location
  - Property Type
- `Property_ID` is treated as an identifier and excluded from predictive modelling.

### Target Variable

**Price**

The model learns patterns between property characteristics and the selling price of the property.

---

## 🧰 Technology Stack

| Category | Technology |
|---|---|
| Programming Language | Python |
| Data Manipulation | Pandas |
| Numerical Computing | NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Development Environment | Jupyter Notebook / VS Code |
| Output Format | CSV, PNG |
| Documentation | Markdown |

---

# 🔄 Project Workflow

```text
Dataset
   ↓
Data Loading & Validation
   ↓
Data Cleaning / Quality Checks
   ↓
Exploratory Data Analysis
   ↓
Feature Selection
   ↓
Train-Test Split
   ↓
Leakage-Safe Preprocessing
   ↓
Linear Regression From Scratch
   ↓
Scikit-learn Validation
   ↓
Polynomial Regression
   ↓
Decision Tree Regression
   ↓
Random Forest Regression
   ↓
Hyperparameter Optimization
   ↓
Final Model Evaluation
   ↓
Feature Importance
   ↓
Residual Diagnostics
   ↓
Model Comparison
   ↓
Business Insights & Conclusion
🔍 Exploratory Data Analysis

The EDA phase investigates the distribution of house prices and the relationship between price and the available property characteristics.

EDA Visualizations
1. Price Distribution

2. Area vs Price

3. Bedrooms & Bathrooms vs Price

4. Location & Property Type

5. Age vs Price

6. Correlation Heatmap

7. Area-Price Relationship by Location

🧹 Data Preprocessing

The preprocessing workflow was designed to avoid data leakage and maintain consistency between training and testing data.

Steps performed
Separated predictors and target variable.
Removed identifier information from the feature set.
Split the dataset into:
80% training data = 240 observations
20% testing data = 60 observations
Identified numerical and categorical features.
Applied one-hot encoding to categorical variables.
Used drop="first" to avoid the dummy-variable trap.
Fitted preprocessing transformations only on the training data.
Applied the learned transformations to the test data.
Final encoded feature structure

The categorical variables use reference categories:

City Center → reference category for Location
Apartment → reference category for Property Type

Resulting encoded categorical features include:

Location_Rural
Location_Suburb
Property_Type_House
Property_Type_Villa

Together with:

Area
Bedrooms
Bathrooms
Age

This produces 8 model features after preprocessing.

🤖 Machine Learning Models
1. Linear Regression — From Scratch

Multiple Linear Regression was implemented manually using the Ordinary Least Squares approach.

The implementation used the design matrix and solved for the regression coefficients without using a regression estimator.

The solution was additionally checked using the Moore-Penrose pseudoinverse for numerical stability.

Validation

The custom implementation was compared against Scikit-learn Linear Regression using:

MAE
MSE
RMSE
R²

The two implementations produced effectively identical predictions, validating the mathematical implementation.

2. Linear Regression — Scikit-learn

Scikit-learn's LinearRegression was trained using the same processed training data and evaluated on the same unseen test set.

This model serves as a benchmark for the from-scratch implementation.

3. Polynomial Regression

A degree-2 Polynomial Regression model was used to investigate whether nonlinear relationships could improve predictive performance.

Only numerical predictors were polynomial-expanded, while categorical variables remained one-hot encoded.

The test results showed that the degree-2 polynomial model did not outperform Linear Regression on this dataset.

4. Decision Tree Regression

A controlled Decision Tree model was developed to capture nonlinear relationships and feature interactions.

The initial configuration used:

Maximum depth = 6
Minimum samples split = 10
Minimum samples leaf = 5
Random state = 42

The tree performed below the linear benchmark on the held-out test set.

5. Random Forest Regression

Random Forest Regression was evaluated as an ensemble approach combining multiple decision trees.

An initial constrained Random Forest configuration was tested first. Its weaker performance demonstrated that model configuration had a substantial effect on predictive accuracy.

⚙️ Random Forest Hyperparameter Optimization

To improve the Random Forest, Randomized Search with 5-fold cross-validation was performed.

Search dimensions included
Number of estimators
Maximum tree depth
Minimum samples required for a split
Minimum samples per leaf
Number of features considered at each split

The test set remained completely untouched during hyperparameter optimization.

Best configuration
Hyperparameter	Value
Number of Trees	300
Maximum Depth	12
Minimum Samples Split	2
Minimum Samples Leaf	2
Max Features	1.0
Cross-Validation	5-fold
Random State	42
🏆 Final Model Performance

The optimized Random Forest produced the best test-set performance.

Model	RMSE	R²
Optimized Random Forest	₹2.219M	0.9654
Linear Regression	₹2.908M	0.9406
Polynomial Regression	₹3.181M	0.9290
Decision Tree	₹3.293M	0.9238
Initial Random Forest	₹4.815M	0.8372
Final Optimized Random Forest Metrics
Metric	Score
MAE	₹1,611,259.09
MSE	₹4,923,749,816,934.83
RMSE	₹2,218,952.41
R²	0.9654
Interpretation

An R² of 0.9654 means the optimized Random Forest explains approximately 96.54% of the variation in house prices on the held-out test dataset.

The RMSE of approximately ₹2.22 million represents the scale of prediction error in the same monetary unit as the target.

Important: These are predictive associations within this dataset and should not be interpreted as causal effects.

📈 Model Comparison
R² Comparison

RMSE Comparison

R² is better when higher, whereas RMSE is better when lower.

Both metrics independently identify the Optimized Random Forest as the strongest model among the evaluated approaches.

🎯 Model Diagnostics
Actual vs Predicted Prices

The predictions are concentrated around the perfect-prediction diagonal, indicating strong agreement between observed and predicted prices.

Residual Distribution

Residuals are generally distributed around zero, although a small number of observations show larger errors.

Residuals vs Predicted Price

Residuals occur on both sides of zero without a pronounced systematic curve. Some increase in error spread is visible at higher predicted prices, indicating that a small number of expensive properties are harder to predict accurately.

🧠 Model Interpretation
Linear Regression Coefficients

The coefficient analysis provides directional information relative to the reference categories.

For example:

Positive coefficients indicate an estimated positive association with price, holding other model variables constant.
Negative coefficients indicate an estimated negative association.
Categorical coefficients are interpreted relative to their reference categories.

Raw coefficient magnitudes should not be treated as direct feature-importance rankings because the predictors have different units and scales.

🔬 Permutation Feature Importance

Permutation importance evaluates how much model performance decreases when an input feature is disrupted.

The analysis indicates that:

Area is the strongest predictive feature.
Location is another major contributor.
Bedrooms provide additional predictive information.
Age, Bathrooms and Property Type contribute less predictive information in this particular dataset.

Permutation importance should be interpreted as predictive contribution rather than causation.

💡 Key Findings
1. Area is the strongest numerical predictor

Area has a strong positive relationship with price, with a Pearson correlation of approximately 0.80.

Larger properties generally command higher prices in this dataset.

2. Location has substantial pricing influence

Properties in different locations show significant differences in price levels.

Location also contributes strongly to predictive performance.

3. Bedrooms have a moderate relationship with price

Bedrooms show a positive but considerably weaker relationship with price compared with Area.

4. Age has a weak negative relationship

Older properties tend to have somewhat lower prices in the dataset, although Age is not among the strongest predictors.

5. Bathrooms have limited standalone linear association

The number of bathrooms shows little linear correlation with price in this dataset.

6. Nonlinear models do not automatically perform better

Polynomial Regression and the initial Decision Tree did not outperform Linear Regression.

This demonstrates that model complexity alone does not guarantee better generalization.

7. Hyperparameter optimization was highly valuable

The initial Random Forest achieved:

R² = 0.8372

After optimization:

R² = 0.9654

This represents a substantial improvement in predictive performance.

8. Optimized Random Forest is the final model

The optimized Random Forest achieved the highest R² and lowest RMSE among the evaluated models and was therefore selected as the final predictive model.

💼 Business Recommendations

Based on the analysis:

Consider property area strongly during valuation.
Account for location when estimating market price.
Use multiple property characteristics rather than relying on a single variable.
Use the optimized Random Forest when predictive accuracy is the primary objective.
Treat model predictions as estimates rather than exact professional valuations.
⚠️ Limitations

This project has several limitations:

The dataset contains only 300 observations.
The available features may not represent every factor affecting property prices.
Important variables such as proximity to schools, hospitals, public transport, amenities and neighborhood quality are not included.
Economic conditions and broader housing-market trends are not represented.
Evaluation uses one held-out test set.
Results may not generalize to other cities, markets or time periods.
Some high-priced properties exhibit larger prediction errors.
🚀 Future Scope

Future improvements could include:

Collecting a substantially larger dataset.
Adding geographic and neighborhood features.
Incorporating distance to schools, hospitals, transport and commercial areas.
Adding market-time and economic indicators.
Testing Gradient Boosting, XGBoost or other ensemble methods.
Performing broader hyperparameter optimization.
Using repeated cross-validation for more robust estimates.
Creating a web interface for real-time price prediction.
Deploying the model as an API or application.
Monitoring model performance after deployment.
📂 Project Structure
HOUSE-PRICE-PREDICTION/
│
├── house_price_prediction.ipynb
├── house_prices.csv
├── README.md
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
    ├── permutation_feature_importance.png
    ├── residual_distribution.png
    ├── residuals_vs_predicted.png
    ├── model_r2_comparison.png
    └── model_rmse_comparison.png
    
▶️ How to Run
1. Clone or download the project

Place the complete project folder on your local machine.

2. Install dependencies
pip install -r requirements.txt
3. Open the notebook
jupyter notebook house_price_prediction.ipynb

Alternatively, open the notebook directly in VS Code with the Jupyter extension.

4. Run the notebook

Execute the notebook from top to bottom.

The notebook will:

Load the dataset
Perform preprocessing
Generate EDA outputs
Train all models
Optimize Random Forest
Evaluate the final model
Generate visualizations
Save analysis outputs into results/
📦 Output Artifacts

The project deliberately separates outputs into two folders.

results/

Contains machine-readable CSV outputs such as:

Correlation results
EDA summaries
Regression metrics
Coefficients
Model comparisons
Permutation importance
Optimized Random Forest results
visualizations/

Contains high-resolution PNG files used for:

EDA
Model interpretation
Model diagnostics
Performance comparison

This separation makes the project easier to inspect, reproduce and reuse.

✅ Quality & Reproducibility Practices

The project follows several reproducibility and modelling best practices:

Fixed random_state=42 where applicable.
Separate training and testing datasets.
Test data kept unseen during hyperparameter optimization.
Preprocessing fitted only on training data.
Reference-category encoding used to avoid dummy-variable multicollinearity.
Same evaluation metrics used across models.
Custom Linear Regression validated against Scikit-learn.
Cross-validation used for Random Forest optimization.
Final model evaluated on an untouched test set.
Results and visualizations exported as separate artifacts.
🏁 Final Conclusion

The project demonstrates a complete machine-learning workflow for house-price prediction, from exploratory analysis and preprocessing to model development, optimization and diagnostic evaluation.

Although simpler models such as Linear Regression performed strongly, the optimized Random Forest achieved the best generalization performance on the held-out test data.

🏆 Final Model

Optimized Random Forest Regression

R² = 0.9654
RMSE = ₹2.22M
MAE = ₹1.61M

The project therefore demonstrates that careful preprocessing, model comparison and hyperparameter optimization can substantially improve predictive performance.

👨‍💻 Author

Vaibhav Shukla

BCA Graduate | Data Analytics & Machine Learning