import numpy as np 
import pandas as pd 
from apply_ml import *
from feature_engineering import *
from sklearn.model_selection import KFold

#--------------------------------------------------------------------------------------------------------------------------------------------------

def main():

	train_file_path = './Data/train.csv'
	test_file_path = './Data/test.csv'

	cross_val_type = 'Leave_One_Out'
	#cross_val_type = 'KFold'

	final_training_type = 'train_test_split'
	final_training_type = 'train_split'

	#nsplit = 5

	#Train Data
	train_data = pd.read_csv(train_file_path)
	num_train_obs = train_data.shape[0]

	#Combining Train and Test Data 
	merged_data = get_merged_data(train_file_path,test_file_path)
	merged_data = get_titles(merged_data)
	merged_data = processing_age_features(merged_data,num_train_obs)
	merged_data = processing_names_features(merged_data)
	merged_data = processing_fares_features(merged_data,num_train_obs)
	merged_data = processing_embarked_features(merged_data)
	merged_data = processing_cabin_features(merged_data)
	merged_data = processing_sex_features(merged_data)
	merged_data = processing_pclass_features(merged_data)
	merged_data = processing_ticket_features(merged_data)
	merged_data = processing_family_features(merged_data)

	train_features, test_features, train_targets = recover_train_test_target(merged_data,train_data,num_train_obs)

	feature_list = list(train_features.columns)

	if cross_val_type == 'Leave_One_Out':
		# Hold Out Cross Validation
		train_features_full_reduced, test_features_reduced, model_crossval = apply_hold_one_out_crossval(train_features,test_features,train_targets,feature_list)

	#elif cross_val_type == 'KFold':
		#Apply KFold Cross Validation
		#apply_kfold_crossval(train_features,test_features,train_targets,nsplit)

	if final_training_type == 'train_test_split':
		#Training on Full Dataset
		final_model = complete_training(train_features_full_reduced, test_features_reduced, train_targets, test_file_path)
	elif final_training_type == 'train_split':
		final_model = model_crossval

	#Prediction
	get_final_prediction(final_model, test_features_reduced, test_file_path)
#-------------------------------------------------------------------------------------------------------------------------------------------------------


if __name__ == '__main__':
	main()