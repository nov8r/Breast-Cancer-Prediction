# Breast Cancer Prediction

This project develops and evaluates several machine learning models to accurately predict the diagnosis (malignant or benign) of a breast tumor based on physical characteristics. It was initially created as a final project for *DAT 350: Data Management and Data Science* and later polished for improved performance.

Colab link: https://colab.research.google.com/drive/1PQXJYlSIaBWE-YCAGOBnnuH4eMikPoTf?usp=sharing

## Project Overview

The primary goal is to apply machine learning techniques on a breast cancer dataset to:

- Clean and preprocess real-world medical data.
- Conduct exploratory data analysis (EDA).
- Train and compare different classification models.
- Evaluate performance using various metrics.

## Dataset

The dataset used is the **Breast Cancer Wisconsin (Diagnostic) Data Set**

It contains attributes related to the physical measurements of breast tumors and a diagnosis label.

## Features and Tools Used

### Libraries:
- **Pandas**, **Numpy**, **Matplotlib**, **Seaborn** – Data Manipulation & Visualization
- **Scikit-Learn** – Model Building, Preprocessing, Evaluation
- **Google Colab** – Development Environment
- **Warnings** - Misc. Libraries

### Models:
- Logistic Regression
- Support Vector Machines (SVM)
- Multi-layer Perceptron (Neural Network)
- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)

## Evaluation:

To assess the performance of each model, the following evaluation metrics were used

- **Accuracy**: Overall proportion of correctly predicted instances.
- **Precision**: The proportion of positive identifications that were actually correct (i.e., low false positive rate).
- **Recall**: The proportion of actual positives that were correctly identified (i.e., low false negative rate).
- **F1 Score**: Harmonic mean of precision and recall, useful for imbalanced datasets.

Confusion Matrices and Model Comparison graphs were also used to visualize model performance.

## Exploratory Data Analysis

Performed tasks include:
- Handling missing values
- Removing irrelevant features
- Analyzing feature distribution
- Visualizing class imbalance and correlation
- Handling outliers with IQR

**Class Distribution**:  
![Class Distribution](https://github.com/nov8r/Breast-Cancer-Prediction/blob/main/Visualizations/class_distribution.png)  
**Correlation Matrix**:  
![Correlation Matrix](https://github.com/nov8r/Breast-Cancer-Prediction/blob/main/Visualizations/correlation_matrix_small.png)

## Model Building

Models were trained using scikit-learn pipelines and evaluated using:
- Train-test splits
- Cross-validation
- Hyperparameter tuning (GridSearchCV)

## Results

Each model's performance was assessed using classification reports and confusion matrices. Comparisons were made to select the most effective model for this classification task.

**Confusion Matrices**:  
![Confusion Matrices](https://github.com/nov8r/Breast-Cancer-Prediction/blob/main/Visualizations/confusion_matrices.png)

**Model Performance Comparison**:  
![Model Performance Comparison](https://github.com/nov8r/Breast-Cancer-Prediction/blob/main/Visualizations/model_performance_comparison.png)

**Overall Model Performance**:  
| Model               | Accuracy | Precision | f1-Score | Recall  |
| :------------------ | :------: | :-------: | :------: | :-----: |
| Logistic Regression | 0.979    | 1.0000    | 0.9709   | 0.9434  |
| SVM                 | 0.979    | 1.0000    | 0.9709   | 0.9434  |
| Neural Network      | 0.965    | 0.9800    | 0.9515   | 0.9245  |
| Decision Tree       | 0.958    | 0.9796    | 0.9412   | 0.9057  |
| Random Forest       | 0.958    | 1.0000    | 0.9400   | 0.8868  |
| K-NN                | 0.958    | 1.0000    | 0.9400   | 0.8868  |

**Best Model:** Logistic Regression (F1 Score: 0.9709)

# Final Thoughts

This project was my entry point into gaining hands-on experience with machine learning. The class from which this project originated also showed me my passion for machine learning in general. 
While these models and this project aren't perfect, I am proud of what I was able to accomplish and create. I learned a lot about machine learning from the creation of this project to the continual improvements I'm making, and I also strengthened my skills in data visualization.
Overall, I am grateful for the class I took and the opportunities it has presented me such as this project here.

## Author

Project by **Ethan Posey**  
Original coursework: DAT 350 – Data Management and Data Science
