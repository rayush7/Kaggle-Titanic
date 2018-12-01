# Kaggle-Titanic
The repository contains the code for Kaggle Competition on Titanic Dataset. The train.csv and test.csv files should be in the same folder as pipeline.py , apply_ml.py and feature_engineering.py. The output file will be predicted_class_hold_one_out.csv and predicted_class_optim.csv. 

Note: 
* predicted_class_hold_one_out.csv is the file without hyperparameter tuning. This is not the file we used for submission.
* The predicted_class_optim.csv is the file which reflects our highest public score in the competition. We obtained this result by applying grid search over gradient boosting classifier in xgboost library. This is the file we used for submission.

To run the code
* python pipeline.py