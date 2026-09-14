# HDB Resale Price Prediction

## Overview

This project explores the factors that influence the resale prices of HDB flats in Singapore and develops machine learning models to predict resale prices based on property characteristics and location-related attributes.

The project is formulated as a **regression problem**, with **Root Mean Squared Error (RMSE)** used as the primary evaluation metric.

### Objectives

* Predict the resale price of an HDB flat.
* Identify the attributes that have the greatest influence on resale prices.
* Compare different regression techniques and their performance.
* Analyse model errors and limitations.
* Explore potential improvements and extensions to the prediction approach.

1. Create Python environment.
```
conda create -n KDDM14 python=3.13 -y
conda activate KDDM14
pip install -r requirements.txt
```
2. Computing walking distance from auxiliary data.
```
cd walking_dist
# Follow instructions in README.md there
cd ..
```
3. Data preprocessing. Run `src/DataPrep.ipynb`.

4. Exploratory data analysis. Run `src/EDA.ipynb`.

5. Model Training. Uncomment respective lines to try models we tried but are suboptimal.
```
cd src
python ModelFitting.py
cd ..
```
6. Model validation.
```
cd src
python "Cross Validation.py"
cd ..
```
7. Feature analysis. First, save our best models:
```
cd src
python "Save Models.py"
cd ..
```
Then run `src/FeatureImportance.ipynb` and `src/FeatureInteraction.ipynb`.
8. Submission: run script to make prediction and submit `pred.csv`:
```
cd src
python submit.py
cd ..
```

Subfolders:
1. data/: original data provided
2. walking_dist/: walking distance matrix generation
3. src/: all source codes
4. models/: the best models (best hyperparameter): 
    - "model3_HistGB_model.pkl": partitioned train set, no cross-validation, for experimentation
    - "model3_HistGradientBoostingRegressor_model.pkl": full train set (without train-validation partition), cross-validation, for testing
