# Breast Cancer Prediction Project Summary

This project aims to develop a cancer prediction model using machine learning techniques, trained to accurately predict the outcome (malignant vs benign) - based on tumor features from the dataset described below.
The project consists of several notebooks that cover different aspects of the model development process.

## About the dataset

This dataset contains characteristics derived from digitized imaging of fine needle aspirates of a breast tumor cell mass.
Ten real-valued features were computed for each cell nucleus:

a) radius (mean of distances from center to points on the perimeter)    
b) texture (standard deviation of gray-scale values)    
c) perimeter    
d) area    
e) smoothness (local variation in radius lengths)    
f) compactness (perimeter^2 / area - 1.0)    
g) concavity (severity of concave portions of the contour)   
h) concave points (number of concave portions of the contour)    
i) symmetry    
j) fractal dimension ("coastline approximation" - 1)    
For each feature there are three related values: its mean , standart error (se) , and greatest value (worst - mean value of three largest values) , in this way there are total 30 features in a dataset plus an identification (observation id) field and the diagnosis field.   
* Original Dataset is available at the UCA Machine Learning Repository:  https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)

## Notebooks

1. **Data Preprocessing**: This notebook focuses on cleaning and preparing the dataset for model training. It includes steps such as handling missing values, feature scaling, and data normalization.

2. **Exploratory Data Analysis (EDA)**: In this notebook, we explore the dataset to gain insights and understand the relationships between variables. We visualize the data using various plots and statistical analysis techniques.

3. **Feature Selection**: This notebook deals with selecting the most relevant features for the prediction model. We use techniques like correlation analysis, feature importance, and dimensionality reduction algorithms.

4. **Model Training**: Here, we train different machine learning models on the preprocessed dataset. We compare their performance using evaluation metrics such as accuracy, precision, recall, and F1-score.

5. **Model Evaluation**: This notebook focuses on evaluating the trained models using cross-validation techniques. We assess their generalization performance and identify the best-performing model.

6. **Model Deployment**: In this final notebook, we deploy the selected model and provide instructions on how to use it for cancer prediction. We also discuss potential future improvements and extensions to the project.

## Conclusion

This project provides a comprehensive overview of the cancer prediction model development process. By following the notebooks in this project, users can gain insights into the data, select relevant features, train and evaluate models, and deploy the final model for practical use.


## Intermidiate Files:
https://drive.google.com/file/d/1evBRddDER4D3St13TbSksQdvcAUzl1tW/view?usp=sharing

## Breast cancer diagnosis can be predicted with roughly 95% accuracy using these 30 histological tumor characteristics.
Machine learning algorithms that performed the best include linear models: logistic regression and linear discriminant analysis and two ensemble algorithms: random forest and an ensemble voting classifier.
The most predictive features found with recursive feature elimination and cross validation using logistic regression as a classifier are: mean texture, perimeter and area, and worst texture, perimeter, and area.
The most predictive features using random forest classification are: mean texture perimeter and area, fractal dimension, and worst area perimeter.