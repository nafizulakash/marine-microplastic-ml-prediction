# 🌊 Marine Microplastics Concentration Prediction using Machine Learning

> **Undergraduate Artificial Intelligence Course Project**  
> Department of Computer Science  
> Independent University, Bangladesh (IUB)

This project was completed as part of my undergraduate Artificial Intelligence course. The main goal was to compare different machine learning models and evaluate how well they can predict **marine microplastic concentration (pieces/m³)** using environmental data. Since this was my first complete machine learning project, I improved the implementation over multiple iterations with guidance from my course instructor and his research assistant.

Instead of focusing on only one model, I wanted to compare different learning approaches and understand their strengths and weaknesses on the same dataset.

**Original Dataset:**  
https://www.kaggle.com/datasets/mselimozen/microplastics-in-the-oceans  
The Kaggle dataset is based on publicly available marine microplastic observations synthesized from multiple research projects. 

> **Note:**  
> The original Kaggle dataset was cleaned and preprocessed before training the models. The cleaned dataset used in this project is included in this repository so the results can be reproduced easily.

---

# 📑 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Features Used](#-features-used)
- [Data Preparation](#-data-preparation)
- [Models Used](#-models-used)
- [Project Workflow](#-project-workflow)
- [Model Evaluation](#-model-evaluation)
- [Repository Structure](#-repository-structure)
- [How to Run](#-how-to-run)
- [Future Improvements](#-future-improvements)
- [What I Learned](#-what-i-learned)


---

# 📖 Project Overview

Microplastic pollution has become a growing environmental problem around the world. Collecting samples from the ocean takes time, money, and human effort. One possible way to support future monitoring is to use machine learning models to estimate microplastic concentration from environmental information.

In this project, I trained and compared four different regression models using the same dataset and evaluation process. I also converted the regression output into low and high concentration classes using a fixed threshold so I could compare their classification performance as well.

---

# 🌍 Dataset

The original dataset was downloaded from Kaggle.

It contains marine environmental observations together with measured microplastic concentrations collected from different parts of the world's oceans. The dataset available on Kaggle is based on a published global synthesis of pelagic microplastic observations.

For this project I used a cleaned version of the dataset after preprocessing.

---

# 📊 Features Used

The final model uses environmental variables including:

- Latitude
- Longitude
- Wind Speed
- Significant Wave Height
- Year
- Month (converted into cyclic features)

**Target Variable**

- Microplastic Concentration (pieces/m³)

---

# 🧹 Data Preparation

Before training the models, several preprocessing steps were applied.

- Removed unnecessary columns
- Handled missing values
- Converted month into sine and cosine values to represent seasonal patterns
- Selected only the features used for training
- Split the dataset into training and testing sets based on year
- Standardized the features before training (mainly for KNN)

The same train and test split was used for every model so the comparison would be fair.

---

# 🤖 Models Used

I selected four different regression models to compare different learning approaches.

### Random Forest Regressor

Random Forest was chosen because it performs well on structured datasets and can learn complex patterns without requiring feature scaling.

### XGBoost Regressor

XGBoost was included because it is one of the most popular boosting algorithms and is known for producing strong prediction results on tabular datasets.

### K-Nearest Neighbors (KNN)

KNN was selected because it predicts values based on the similarity between nearby samples. I wanted to compare this approach with the tree based models.

### Stacking Regressor

The stacking model combines the predictions from Random Forest, XGBoost, and KNN using Linear Regression as the final estimator. I wanted to see whether combining multiple models could improve prediction performance.

---

# 🔄 Project Workflow

```text
Original Dataset
        │
        ▼
Data Cleaning
        │
        ▼
Feature Engineering
        │
        ▼
Train / Test Split
        │
        ▼
Feature Scaling
        │
        ▼
Train Four ML Models
        │
        ▼
Evaluate Performance
        │
        ▼
Compare Results
```

---

# 📈 Model Evaluation

The regression models were evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

To better understand the prediction performance, the regression outputs were also converted into binary classes using a threshold of **0.1**.

The following classification metrics were calculated:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

# 📂 Repository Structure

```text
marine-microplastics-ml-prediction/
│
├──  dataset.csv
├── LICENSE
├── marine_microplastics_prediction.py
└──  README.md

```

---

# 🚀 How to Run

1. Download this repository as a ZIP file or clone it to your computer.

2. Open **Google Colab**: https://colab.research.google.com/

3. Upload the project file (`marine_microplastics_prediction.py`) to Google Colab.

4. Download `dataset.csv` from this repository and upload it using the **Files** panel on the left side of Colab.

5. Make sure the uploaded file is named **`dataset.csv`**. If you change the filename, update the dataset path in the code before running the project.

6. Run the Python file from top to bottom.

> **Note:**  
> Files uploaded to Google Colab are only available during the current session. If the runtime is restarted or disconnected, you will need to upload `dataset.csv` again before running the project.

# 🚧 Future Improvements

There are several things I would like to improve in the future.

- Perform hyperparameter tuning
- Include more environmental variables
- Add variables related to ocean physics such as distance from shoreline and ocean currents
- Experiment with deep learning models
- Improve the overall prediction performance
- Test the models on larger and more recent datasets

---

# 📚 What I Learned

This project helped me understand the complete machine learning workflow, including:

- Data preprocessing
- Feature engineering
- Working with environmental datasets
- Comparing different regression models
- Building a stacking ensemble
- Evaluating model performance using multiple metrics
- Creating clear visualizations for model comparison

---

