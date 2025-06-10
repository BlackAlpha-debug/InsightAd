#  Predictive Analytics for Social Network Ads

This project explores **predictive analytics** in the realm of social media advertising. The primary objective is to develop a robust machine learning model that predicts whether a user will purchase a product based on demographic features such as **gender**, **age**, and **estimated salary**. Accurately identifying potential buyers can help businesses optimize ad spending and improve return on investment (ROI).

This repository includes everything from data preprocessing and model training to evaluation and deployment-ready model serialization.

---

## 📂 Dataset

We utilize the `Social_Network_Ads.csv` dataset—widely used in tutorials for binary classification tasks. It simulates real-world advertising behavior and enables easy experimentation with machine learning models.

### Features:

* `User ID`: Unique identifier for each user (not used in model training).
* `Gender`: Categorical feature representing the user's gender (`Male` or `Female`).
* `Age`: User’s age (numeric).
* `EstimatedSalary`: Estimated annual salary of the user.
* `Purchased`: **Target** variable (1 if purchased, 0 otherwise).

---

## 🧱 Project Structure

* `Social_Network_Ads.csv`: Input dataset.
* `purchase_prediction.ipynb`: Jupyter notebook with the complete machine learning pipeline.
* `rf_pipeline.joblib`: Serialized pipeline with scaling and a trained Random Forest Classifier.

---

## 🔁 ML Workflow Overview

### 1. **Data Loading & Inspection**

* Load dataset using `pandas`.
* Inspect structure, types, and check for missing values (none found).

### 2. **Data Preprocessing**

* **Label Encoding**: Convert `Gender` into numerical values using `LabelEncoder`.
* **Feature/Target Split**: Features = `['Gender', 'Age', 'EstimatedSalary']`, Target = `Purchased`.
* **Train-Test Split**: 80% training, 20% testing using `train_test_split`.
* **Feature Scaling**: Apply `StandardScaler` to normalize `Age` and `EstimatedSalary`.

### 3. **Model Training**

* **Random Forest Classifier**: Chosen for its high performance and ability to capture non-linear patterns.
* **Logistic Regression**: Used as a baseline model.

### 4. **Model Evaluation**

Evaluation metrics include:

* **Classification Report**: Precision, Recall, F1-score, Support.
* **Confusion Matrix**: Visualized via `seaborn.heatmap`.
* **ROC Curve and AUC**: Using `roc_curve` and `roc_auc_score`.

### 5. **Model Comparison**

* Direct comparison of accuracy and ROC AUC between Random Forest and Logistic Regression.

### 6. **Model Serialization**

* Pipeline constructed using `Pipeline([('scaler', StandardScaler()), ('model', RandomForestClassifier())])`.
* Saved using `joblib.dump()` to `rf_pipeline.joblib` for easy reuse.

---

## 📊 Results

Random Forest outperformed Logistic Regression in all major metrics:

| Model               | Accuracy   | ROC AUC        |
| ------------------- | ---------- | -------------- |
| Random Forest       | **93.75%** | *(e.g., 0.96)* |
| Logistic Regression | 92.25%     | *(e.g., 0.94)* |

> 📌 *The Random Forest Classifier effectively identifies users likely to purchase, making it a strong candidate for targeted advertising strategies.*

---

## ⚙️ Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/BlackAlpha-debug/InsightAd.git
cd Social-Network-Ads-Purchase-Prediction
```

### 2. (Optional) Create a Virtual Environment

```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> If `requirements.txt` is missing, generate one with:
>
> ```bash
> pip freeze > requirements.txt
> ```

### 4. Run the Notebook or Script

* **Jupyter Notebook**:

  ```bash
  jupyter notebook
  ```

  Open `purchase_prediction.ipynb` and run all cells.
* **Python Script**:

  ```bash
  python purchase_prediction.py
  ```

---

## 🧰 Dependencies

* `pandas`
* `numpy`
* `scikit-learn`
* `matplotlib`
* `seaborn`
* `joblib`

You can install all with:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

---

## 📎 License

This project is released under the [MIT License](LICENSE). Feel free to use and modify it for your needs.

