## 📁 Project Structure

The project is organized into separate files and directories for analysis, results, visualizations, and documentation.

**Project structure:**

    HOUSE-PRICE-PREDICTION/
    │
    ├── README.md
    ├── requirements.txt
    ├── house_price_prediction.ipynb
    ├── house_prices.csv
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

### 📂 Directory Description

| File / Directory | Purpose |
|---|---|
| `house_price_prediction.ipynb` | Complete analysis, modelling, evaluation, and documentation |
| `house_prices.csv` | Original dataset used for the project |
| `README.md` | Project documentation and results summary |
| `requirements.txt` | Python dependencies required to run the project |
| `results/` | Machine-readable CSV outputs from the analysis and models |
| `visualizations/` | EDA, model interpretation, diagnostics, and comparison charts |

---

## 📦 Output Artifacts

### `results/`

The `results/` directory contains machine-readable CSV files generated during the analysis, including:

- Preprocessing summaries
- Correlation analysis
- Feature relationship analysis
- Linear Regression results
- Scratch vs Scikit-learn comparison
- Linear Regression coefficients
- Polynomial Regression results
- Decision Tree results
- Random Forest results
- Optimized Random Forest results
- Permutation Feature Importance

### `visualizations/`

The `visualizations/` directory contains presentation-ready PNG files covering:

- Exploratory Data Analysis
- Correlation analysis
- Feature relationships
- Linear Regression interpretation
- Feature importance
- Actual vs Predicted analysis
- Residual diagnostics
- R² model comparison
- RMSE model comparison

Separating numerical outputs and visual outputs keeps the project organized and makes the results easy to reuse.

---

## ▶️ Installation & Setup

### 17.1 Clone the Repository

Clone the repository using Git:

    git clone <YOUR-GITHUB-REPOSITORY-URL>

Navigate into the project directory:

    cd House-Price-Prediction

### 17.2 Create a Virtual Environment

A virtual environment is recommended to keep project dependencies isolated.

**Windows:**

    python -m venv venv
    venv\Scripts\activate

**macOS / Linux:**

    python3 -m venv venv
    source venv/bin/activate

### 17.3 Install Dependencies

    pip install -r requirements.txt

The project requires:

- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter

---

## 📋 Requirements

All required Python dependencies are listed in:

`requirements.txt`

Current dependencies:

    numpy
    pandas
    matplotlib
    seaborn
    scikit-learn
    jupyter

Using a requirements file makes it easier to recreate the project environment.

---

## ▶️ Running the Project

### Option 1 — Jupyter Notebook

Launch Jupyter Notebook:

    jupyter notebook

Then open:

`house_price_prediction.ipynb`

Run the notebook sequentially from the first cell to the final cell.

### Option 2 — VS Code

1. Open the project folder in VS Code.
2. Install the Jupyter extension if required.
3. Open `house_price_prediction.ipynb`.
4. Select the appropriate Python environment/kernel.
5. Run the notebook from top to bottom.

### Execution Flow

The notebook performs the following operations:

**Dataset → Validation → EDA → Preprocessing → Model Training → Hyperparameter Optimization → Evaluation → Diagnostics → Model Comparison → Insights → Conclusion**

### Generated Outputs

After execution:

- CSV results are saved in `results/`
- PNG visualizations are saved in `visualizations/`

---

## 🔁 Reproducibility

Several practices were followed to make the analysis consistent and reproducible:

- Fixed `random_state=42` where applicable.
- Used a consistent train-test split.
- Kept the test dataset unseen during model training.
- Kept the test dataset isolated during hyperparameter optimization.
- Fitted preprocessing transformations using training data only.
- Applied the same transformations to the unseen test data.
- Used consistent evaluation metrics across models.
- Validated the custom Linear Regression implementation against Scikit-learn.
- Used 5-fold cross-validation during Random Forest optimization.
- Saved numerical outputs as CSV files.
- Saved visualizations as PNG files.
- Documented the complete workflow inside the Jupyter Notebook.

---

## 🧪 Evaluation Metrics

Multiple regression metrics were used to evaluate predictive performance.

### Mean Absolute Error — MAE

MAE measures the average absolute difference between actual and predicted prices.

**Lower MAE indicates better performance.**

### Mean Squared Error — MSE

MSE measures the average squared prediction error.

Because the errors are squared, larger errors receive greater weight.

**Lower MSE indicates better performance.**

### Root Mean Squared Error — RMSE

RMSE is the square root of MSE and is expressed in the same units as the target variable.

**Lower RMSE indicates better performance.**

### R² Score

R² measures the proportion of variation in the target variable explained by the model.

**Higher R² indicates better predictive performance.**

The final Optimized Random Forest achieved:

**R² = 0.9654**

---

## 📊 Metric Summary

| Metric | Preferred Direction | Purpose |
|---|---|---|
| MAE | Lower | Average absolute prediction error |
| MSE | Lower | Penalizes larger errors |
| RMSE | Lower | Error magnitude in target units |
| R² | Higher | Proportion of explained variance |

Using multiple metrics provides a more complete evaluation than relying on a single performance score.

---

## 🏁 Final Conclusion

This project demonstrates a complete end-to-end machine-learning workflow for residential house-price prediction.

The analysis progressed from data validation and exploratory analysis through feature preparation, model development, hyperparameter optimization, model diagnostics, and final interpretation.

The evaluated approaches included:

- Linear Regression From Scratch
- Scikit-learn Linear Regression
- Polynomial Regression
- Decision Tree Regression
- Initial Random Forest Regression
- Optimized Random Forest Regression

The comparison demonstrated that greater model complexity does not automatically guarantee better predictive performance.

While Linear Regression provided a strong baseline, systematic Random Forest hyperparameter optimization produced a substantial improvement in predictive performance.

### Key Conclusions

- **Area** is the strongest numerical predictor.
- **Location** provides substantial predictive information.
- Linear Regression provides a strong and interpretable baseline.
- Polynomial Regression did not improve upon the linear baseline.
- The initial Random Forest configuration performed poorly before optimization.
- Hyperparameter optimization substantially improved Random Forest performance.
- Residual diagnostics showed generally well-distributed prediction errors with some larger errors among higher-priced properties.
- The Optimized Random Forest achieved the best overall predictive performance among the evaluated models.

---

## 🏆 Final Model

### Optimized Random Forest Regression

| Metric | Final Score |
|---|---:|
| **R²** | **0.9654** |
| **RMSE** | **₹2.22M** |
| **MAE** | **₹1.61M** |
| **MSE** | **₹4.92T** |

The final model explains approximately **96.54% of the variation in house prices within the held-out test dataset**.

The Optimized Random Forest is therefore selected as the final predictive solution for this dataset.

> **Disclaimer:** This project is intended for educational and analytical purposes. Model predictions should be treated as estimates and should not be considered professional property valuations.

---

## 📌 Project Highlights

| Area | Achievement |
|---|---|
| Dataset | 300 property records |
| Train-Test Split | 80:20 |
| Models Evaluated | 6 approaches |
| Hyperparameter Optimization | Randomized Search + 5-Fold CV |
| Best Model | Optimized Random Forest |
| Best R² | **0.9654** |
| Best RMSE | **₹2.22M** |
| Feature Importance | Permutation Importance |
| Model Diagnostics | Actual vs Predicted + Residual Analysis |
| Output Artifacts | CSV + PNG |
| Documentation | Complete README + Jupyter Notebook |

---

## 👤 Author

**Vaibhav Shukla**

**BCA Graduate | Data Analytics & Machine Learning**

---

⭐ If you find this project useful, feel free to explore the notebook, results, and visualizations.
