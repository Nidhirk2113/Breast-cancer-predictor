# Breast Cancer Prediction using Machine Learning

This repository contains a machine learning project focused on predicting breast cancer diagnoses (malignant or benign) using various classification algorithms. The project leverages the Breast Cancer Wisconsin dataset, exploring multiple models and comparing their performance.

## 📂 Project Structure

```
breast_cancer_prediction/
├── breast_cancer_prediction.ipynb  # Jupyter notebook with complete code
├── README.md                       # Project overview and documentation
```

## 🔍 Problem Statement

Breast cancer is a significant public health concern, and early diagnosis is critical to effective treatment. This project aims to develop an accurate prediction model that assists in classifying tumors as malignant or benign based on features extracted from digitized images of fine needle aspirate (FNA) of breast masses.

## 📊 Dataset

- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
- **Features:** 30 numerical features computed from digitized images (e.g., radius, texture, perimeter, area, smoothness, etc.)
- **Target:** Diagnosis (M = malignant, B = benign)

## 📌 Key Features of the Notebook

- Data loading and preprocessing
- Exploratory Data Analysis (EDA)
- Feature scaling and selection
- Model training using:
  - Logistic Regression
  - K-Nearest Neighbors (KNN)
  - Support Vector Machine (SVM)
  - Random Forest Classifier
- Performance evaluation (accuracy, confusion matrix, classification report)
- Model comparison

## 📈 Results

All models were evaluated using:
- **Accuracy score**
- **Confusion Matrix**
- **Classification Report (Precision, Recall, F1-Score)**

> The Random Forest and SVM classifiers showed strong performance, with accuracy often exceeding 95% on the test set.

## 🚀 Getting Started

### Prerequisites

Ensure you have the following Python packages installed:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### Running the Notebook

1. Clone this repository.
2. Open `breast_cancer_prediction.ipynb` in Jupyter Notebook or JupyterLab.
3. Run the cells sequentially to see data analysis, model building, and results.

## 📚 Future Work

- Hyperparameter tuning using GridSearchCV
- Model serialization with `joblib` or `pickle`
- Integration with a web UI (e.g., using Streamlit or Flask)
- Handling imbalanced datasets using SMOTE

## 🧠 Author

- **Nidhi Kulkarni**
- Engineering Student,DSATM

## 📜 License

This project is open source and available under the [MIT License](LICENSE).
