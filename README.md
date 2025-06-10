## Overview

This project delves into the fascinating world of **predictive analytics** within the social media advertising landscape. The core objective is to develop and evaluate a robust machine learning model capable of predicting whether a social network user is likely to purchase a product based on their demographic information (gender, age, and estimated salary). By accurately identifying potential customers, businesses can significantly refine their marketing strategies, leading to more efficient ad spending and a higher return on investment. This repository contains the code for data preprocessing, model training, evaluation, and the final serialized model pipeline.

---

## 📊 Dataset

The project utilizes the `Social_Network_Ads.csv` dataset, a simulated dataset common in machine learning tutorials. It provides a simplified yet effective scenario to demonstrate classification tasks.

### Dataset Columns:

* **`User ID`**: A unique numerical identifier assigned to each user. While useful for distinguishing individuals, this column is typically dropped during model training as it doesn't contribute to predictive power.
* **`Gender`**: Categorical variable indicating the user's gender (Male or Female). This feature is crucial for understanding potential differences in purchasing behavior across genders.
* **`Age`**: A numerical variable representing the user's age. Age is a significant factor in consumer behavior, often correlating with purchasing power and preferences.
* **`EstimatedSalary`**: A numerical variable indicating the user's estimated annual salary. Salary is a direct indicator of purchasing capacity and is expected to heavily influence the likelihood of a purchase.
* **`Purchased`**: The **target variable** for our prediction task. This is a binary categorical variable (0 for No, 1 for Yes), indicating whether the user ultimately purchased the product.

---

## 🛠️ Project Structure

* `Social_Network_Ads.csv`: The primary dataset used for all analyses and model training.
* `purchase_prediction.ipynb` (or similar name if converted to a Python script): The Jupyter Notebook (or Python script) containing the complete workflow from data loading to model evaluation and saving.
* `rf_pipeline.joblib`: The serialized machine learning pipeline. This file encapsulates both the data scaling mechanism (`StandardScaler`) and the trained **Random Forest Classifier**, ensuring that future predictions are made using the exact same preprocessing steps applied during training.

---

## ✨ Key Steps Performed

This project follows a standard machine learning pipeline to ensure data integrity, model efficiency, and reliable predictions.

1.  **Data Loading and Initial Inspection**:
    * The `Social_Network_Ads.csv` file is loaded into a Pandas DataFrame.
    * Initial checks are performed using `data.head()`, `data.shape`, and `data.dtypes` to understand the dataset's structure, dimensions, and data types.
    * Missing values are identified using `data.isnull().sum()` to ensure data quality before preprocessing. In this case, the dataset is clean with no missing values.

2.  **Data Preprocessing**:
    * **Categorical Feature Encoding**: The `Gender` column, being a categorical feature, is transformed into numerical format using `sklearn.preprocessing.LabelEncoder`. This is a necessary step as machine learning models primarily work with numerical input. 'Male' and 'Female' are converted to 0 and 1, respectively.
    * **Feature-Target Split**: The dataset is divided into features (`X`, containing 'Gender', 'Age', 'EstimatedSalary') and the target variable (`Y`, 'Purchased').
    * **Data Splitting**: The dataset is split into training and testing sets using `sklearn.model_selection.train_test_split` with an 80/20 ratio (80% for training, 20% for testing) and `random_state=0` for reproducibility. This ensures that the model is trained on one portion of the data and evaluated on unseen data.
    * **Feature Scaling**: `sklearn.preprocessing.StandardScaler` is applied to `X_train` and `X_test`. This step is crucial for numerical features (Age, EstimatedSalary) as it transforms them to have a mean of 0 and a standard deviation of 1. Scaling prevents features with larger numerical ranges from dominating the learning process and helps optimization algorithms converge faster.

3.  **Model Training**:
    * **Random Forest Classifier**: An ensemble learning method, `RandomForestClassifier` from `sklearn.ensemble`, is chosen for its robustness, ability to handle non-linear relationships, and good performance on various datasets. It builds multiple decision trees and merges their predictions to improve accuracy and control overfitting. The model is trained on the scaled training data (`X_train`, `Y_train`).
    * **Logistic Regression**: As a baseline for comparison, a `LogisticRegression` model from `sklearn.linear_model` is also trained. Logistic Regression is a simple yet powerful linear model for binary classification, providing a good reference point for the more complex Random Forest.

4.  **Model Evaluation**:
    * **Predictions**: Both models make predictions on the unseen test data (`X_test`). For Random Forest, `Y_pred` gives class predictions (0 or 1), and `y_proba` provides the probability of a positive class (purchase).
    * **Classification Report**: A detailed `classification_report` from `sklearn.metrics` provides a summary of:
        * **Precision**: The proportion of positive identifications that were actually correct. High precision means fewer false positives.
        * **Recall (Sensitivity)**: The proportion of actual positives that were identified correctly. High recall means fewer false negatives.
        * **F1-Score**: The harmonic mean of Precision and Recall, offering a balance between the two metrics.
        * **Support**: The number of actual occurrences of each class in the specified dataset.
    * **Confusion Matrix**: Generated using `sklearn.metrics.confusion_matrix` and visualized using `seaborn.heatmap`. This matrix provides a clear breakdown of:
        * **True Positives (TP)**: Correctly predicted positive cases.
        * **True Negatives (TN)**: Correctly predicted negative cases.
        * **False Positives (FP)**: Incorrectly predicted positive cases (Type I error).
        * **False Negatives (FN)**: Incorrectly predicted negative cases (Type II error).
        The heatmap offers an intuitive visual representation of the model's performance across different prediction outcomes.
    * **Key Performance Metrics**: Individual calculations for:
        * **Accuracy**: The proportion of correctly classified instances out of the total instances.
        * **Recall**: (as described above)
        * **Precision**: (as described above)
        * **F1 Score**: (as described above)
    * **ROC Curve and AUC**: The Receiver Operating Characteristic (ROC) curve is plotted, and the Area Under the Curve (AUC) is calculated using `sklearn.metrics.roc_curve` and `roc_auc_score`.
        * The **ROC curve** illustrates the diagnostic ability of a binary classifier system as its discrimination threshold is varied.
        * The **AUC** represents the degree or measure of separability. It tells us how much the model is capable of distinguishing between classes. A higher AUC means the model is better at predicting 0s as 0s and 1s as 1s.

5.  **Model Comparison**: A direct comparison of the `accuracy_score` between the Random Forest Classifier and Logistic Regression model is provided to highlight the performance difference.

6.  **Pipeline Creation and Saving**:
    * A `sklearn.pipeline.Pipeline` is constructed, chaining the `StandardScaler` with the trained `RandomForestClassifier`. This is a best practice for deployment, ensuring that any new data fed to the model will automatically undergo the same scaling transformation before prediction.
    * The entire pipeline is then saved to disk using `joblib.dump` as `rf_pipeline.joblib`. This allows for easy loading and deployment of the trained model without needing to manually re-apply preprocessing steps.

---

## 📈 Results

The Random Forest Classifier consistently demonstrated strong performance on this dataset. The detailed classification report, confusion matrix, and ROC AUC curve provide a comprehensive view of its predictive capabilities.

*(Here, you could insert specific numerical results obtained from running the notebook, e.g.:)*

* **Random Forest Accuracy**: `[Your RF Accuracy]`
* **Logistic Regression Accuracy**: `[Your LR Accuracy]`
* **Random Forest ROC AUC**: `[Your RF ROC AUC Score]`

These metrics indicate that the Random Forest model is effective in distinguishing between users who will and will not purchase the product, making it a valuable tool for targeted advertising.

---

## ⚙️ How to Run

To run this project and reproduce the results, please follow these steps:

1.  **Clone the Repository**:
    ```bash
    git clone <your-repository-url-here>
    cd Social-Network-Ads-Purchase-Prediction
    ```
2.  **Create a Virtual Environment (Recommended)**:
    This isolates your project's dependencies from your global Python environment.
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```
3.  **Install Dependencies**:
    All necessary libraries can be installed using `pip`.
    ```bash
    pip install -r requirements.txt
    ```
    (If `requirements.txt` is not present, see the **Dependencies** section below to create it.)
4.  **Execute the Code**:
    * **Jupyter Notebook**: If you plan to run the `.ipynb` file:
        ```bash
        jupyter notebook
        ```
        Then, open `purchase_prediction.ipynb` in your browser and run all cells.
    * **Python Script**: If you have converted the notebook to a `.py` file (e.g., `purchase_prediction.py`):
        ```bash
        python purchase_prediction.py
        ```

---

## 📚 Dependencies

This project relies on the following Python libraries:

* **`pandas`**: For data manipulation and analysis.
* **`numpy`**: For numerical operations.
* **`scikit-learn`**: The primary machine learning library for model building, preprocessing, and evaluation.
* **`matplotlib`**: For creating static, interactive, and animated visualizations.
* **`seaborn`**: A data visualization library based on matplotlib, providing a high-level interface for drawing attractive and informative statistical graphics.
* **`joblib`**: For efficiently saving and loading Python objects, especially large NumPy arrays or scikit-learn pipelines.

You can generate a `requirements.txt` file from your environment using:

```bash
pip freeze > requirements.txt
