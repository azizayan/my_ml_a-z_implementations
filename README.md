
## Machine Learning Learning Exercises

This repository documents a series of machine learning experiments conducted as part of a structured, hands-on learning process. Each submodule focuses on the "why" behind the "how," covering data preprocessing, model behavior, and identifying common pitfalls like data leakage and target imbalance. Collection of ML-A-Z style implementations plus small side projects. Covers preprocessing, core regression/classification algorithms, model evaluation, and a few applied mini-projects (pumpkin pricing/color, breast cancer baseline, UFO exploration).

## Regression Methods: 
multiple linear regression, polynomial regression, support vector regression , decision tree regression, random forest regression; model evaluation scripts (train/test splits, scores).
## Classification Methods: 
Logistic regression, k-NN, SVM , kernel SVM, decision tree, random forest, naive Bayes; classification evaluation datasets and scripts with confusion matrices and reports.

---

## 📊 Additional Projects: 
1) Pumpkin Color Prediction: Using USDA pumpkin data, this project cleans and filters bushel sales, builds linear and polynomial regression models to track pie-type pumpkin prices across the growing season, and trains logistic regression classifiers (with tailored encodings and preprocessing) to predict pumpkin color from location, variety, origin, size, and date features. It includes light EDA (seaborn/matplotlib), categorical encodings (ordinal + one-hot), imputation where needed, and reports standard regression errors plus classification accuracy/F1 and full reports on held-out splits.

2) Breast Cancer Detection: Logistic regression baseline for the Wisconsin reast cancer dataset with cross validation

3) UFOS(ongoing): Early stage UFO sightings exploration. Currently loads the sightings CSV, keeps a trimmed set of columns (duration seconds, country, latitude, longitude), inspects unique countries, and drops missing values as a first cleaning step. End goal: complete a geospatial/temporal analysis , map sighting hotspots, model duration patterns by region/time, and build simple classifiers or anomaly detectors to flag unusual sightings, once the cleaned dataset and feature set are finalized.


