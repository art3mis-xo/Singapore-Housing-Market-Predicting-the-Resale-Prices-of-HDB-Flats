Link to Google shared drive: https://drive.google.com/drive/folders/1x-HOzK80HzgznGlfGTi5bJY9BBTJR23U?usp=drive_link 
Predicting Resale Prices of HDB flats in Singapore (CS5228, 7 November 2025)
Done by: Group 14 ANCHovYs (Charissa, Nicholas, Yiwei, Akshaya, Haiwei)

This document records how to produce the data in our report.
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
3. intermediate_csvs/: intermediate data processed csv files
4. src/: all source codes
5. models/: our best models (best hyperparameter): 
    - "model3_HistGB_model.pkl": partitioned train set, no cross-validation, for experimentation
    - "model3_HistGradientBoostingRegressor_model.pkl": full train set (without train-validation partition), cross-validation, for testing
6. pred.csv: our final Kaggle submission