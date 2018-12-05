from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier

#------------------------------------------------------------------------------------------------------------------------------

def compute_score(clf, X, y, scoring='accuracy'):
    xval = cross_val_score(clf, X, y, cv = 5, scoring=scoring)
    return np.mean(xval)

#-------------------------------------------------------------------------------------------------------------------------------
def feature_reduction(train_features,y_train,val_features,test_features,train_features_full,feature_list):

	clf = RandomForestClassifier(n_estimators=100, max_features='sqrt')
	clf = clf.fit(train_features, y_train)

	features = pd.DataFrame()
	features['feature'] = feature_list
	features['importance'] = clf.feature_importances_
	features.sort_values(by=['importance'], ascending=True, inplace=True)
	features.set_index('feature', inplace=True)

	features.plot(kind='barh', figsize=(25, 25),color='r')
	plt.savefig('feature_importance.png')

	model = SelectFromModel(clf, prefit=True)
	train_features_reduced = model.transform(train_features)
	val_features_reduced = model.transform(val_features)
	test_features_reduced = model.transform(test_features)
	train_features_full_reduced = model.transform(train_features_full)

	return train_features_reduced, val_features_reduced ,test_features_reduced, train_features_full_reduced

#-------------------------------------------------------------------------------------------------------------------------------

def apply_xgboost_gradient_boosting(X_train_preprocessed, X_test_preprocessed, y_train, y_test):

	print 'Applying XGBoost Gradient Boosting'

	model = XGBClassifier()
	model.fit(X_train_preprocessed, y_train)

	y_test_pred_values = model.predict(X_test_preprocessed)
	y_test_pred = [int(value) for value in y_test_pred_values]

	# Compute Accuracy Score
	acc = accuracy_score(y_test,y_test_pred,normalize=True)

	print 'The accuracy achieved by the Gradient Boosting Model from XgBoost is: ', acc

	return model, acc

#--------------------------------------------------------------------------------------------------------------------------------

def apply_adaboost_classifier(X_train_preprocessed, X_test_preprocessed, y_train, y_test):

	##TO DO : Testing Hyper Parameters and Cross Validation
	
	print 'Applying AdaBoost Classifier'

	# Training the classifier
	classifier = AdaBoostClassifier(n_estimators=100)
	classifier = classifier.fit(X_train_preprocessed,y_train)
	
	# Testing the classifier on Test Data
	y_test_pred = classifier.predict(X_test_preprocessed) 

	#Compute Accuracy Score
	acc = accuracy_score(y_test,y_test_pred,normalize=True)

	print 'The accuracy achieved by the Adaboost Classifier Model is: ', acc

	return classifier, acc

#---------------------------------------------------------------------------------------------------------------------------------

def apply_gradient_boosting(X_train_preprocessed, X_test_preprocessed, y_train, y_test):

	##TO DO : Testing Hyper Parameters and Cross Validation
	
	print 'Applying Gradient Boosting'

	# Training the classifier
	classifier = GradientBoostingClassifier(n_estimators=100)
	classifier = classifier.fit(X_train_preprocessed,y_train)
	
	# Testing the classifier on Test Data
	y_test_pred = classifier.predict(X_test_preprocessed)

	#Compute Accuracy Score
	acc = accuracy_score(y_test,y_test_pred,normalize=True)

	print 'The accuracy achieved by the Gradient Boosting Classifier Model is: ', acc

	return classifier, acc

#--------------------------------------------------------------------------------------------------------------------------------

def apply_logistic_regression(X_train_preprocessed, X_test_preprocessed, y_train, y_test):

	##TO DO : Testing Hyper Parameters and Cross Validation
	
	print 'Applying Logistic Regression'

	# Training the classifier
	classifier = LogisticRegression()
	classifier = classifier.fit(X_train_preprocessed,y_train)
	
	# Testing the classifier on Test Data
	y_test_pred = classifier.predict(X_test_preprocessed)

	#Compute Accuracy Score
	acc = accuracy_score(y_test,y_test_pred,normalize=True)

	print 'The accuracy achieved by the Logistic Regression Classifier Model is: ', acc

	return classifier, acc

#-------------------------------------------------------------------------------------------------------------------------------

def apply_data_split_preprocessing(raw_dataset,labels):

	# Perform Basic Preprocessing and Train, Validation Split
	X_train, X_val, y_train, y_val = train_test_split(raw_dataset,labels,test_size=0.30,random_state=22)
	
	# Apply Basic Preprocessing Steps

	print 'Applying Data Preprocessing'

	min_max_scaler = preprocessing.MinMaxScaler()
	X_train_minmax = min_max_scaler.fit_transform(X_train)
	X_val_minmax = min_max_scaler.transform(X_val)

	return X_train_minmax, X_val_minmax, y_train, y_val, min_max_scaler

#------------------------------------------------------------------------------------------------------------------------------------

def apply_hold_one_out_crossval(train_features,test_features,train_targets,feature_list):

	print 'Apply Hold One Out Cross Validation!'
	
	#Apply Machine Learning Techinques and Cross Validation
	X_train_preprocessed, X_val_preprocessed, y_train, y_val, min_max_scaler = apply_data_split_preprocessing(train_features,train_targets)

	#Test Feature Preprocessing
	test_features_preprocessed = min_max_scaler.fit_transform(test_features)

	#Train Feature Preprocessing
	train_features_preprocessed = min_max_scaler.fit_transform(train_features)

	#Apply Feature Reduction
	X_train_reduced, X_val_reduced, test_features_reduced, train_features_full_reduced =feature_reduction(X_train_preprocessed,y_train, X_val_preprocessed,test_features_preprocessed,train_features_preprocessed,feature_list)

	model1, score1 = apply_xgboost_gradient_boosting(X_train_reduced, X_val_reduced, y_train, y_val)
	model2, score2 = apply_random_forest(X_train_reduced, X_val_reduced, y_train, y_val)
	model3, score3 = apply_multi_class_svc(X_train_reduced, X_val_reduced, y_train, y_val)
	model4, score4 = apply_gradient_boosting(X_train_reduced, X_val_reduced, y_train, y_val)
	model5, score5 = apply_logistic_regression(X_train_reduced, X_val_reduced, y_train, y_val)
	model6, score6 = apply_adaboost_classifier(X_train_reduced, X_val_reduced, y_train, y_val)

	all_model = [model1,model2,model3,model4,model5,model6]
	all_score = [score1,score2,score3,score4,score5,score6]

	max_score = max(all_score)

	model = all_model[all_score.index(max_score)]

	return train_features_full_reduced, test_features_reduced, model

#--------------------------------------------------------------------------------------------------------------------------------------

def apply_random_forest(X_train_preprocessed, X_test_preprocessed, y_train, y_test):

	##TO DO : Testing Hyper Parameters and Cross Validation
	
	print 'Applying Random Forest'

	# Training the classifier
	classifier = RandomForestClassifier(n_estimators=100, max_depth=4,random_state=0)
	classifier = classifier.fit(X_train_preprocessed,y_train)
	
	# Testing the classifier on Test Data
	y_test_pred = classifier.predict(X_test_preprocessed) 

	#Compute Accuracy Score
	acc = accuracy_score(y_test,y_test_pred,normalize=True)

	print 'The accuracy achieved by the Random Forest Classifier Model is: ', acc

	return classifier, acc

#---------------------------------------------------------------------------------------------------------------------------------------

def apply_multi_class_svc(X_train_preprocessed, X_test_preprocessed, y_train, y_test):

	##TO DO : Testing Hyper Parameters and Cross Validation
	
	print 'Applying Multi-Class SVC'

	clf = SVC(gamma='auto')
	clf = clf.fit(X_train_preprocessed,y_train)

	# Testing the Classifier on Test Data
	y_test_pred = clf.predict(X_test_preprocessed)

	#Compute Accuracy Score
	acc = accuracy_score(y_test,y_test_pred,normalize=True)

	print 'The accuracy achieved by Support Vector Classifier is: ', acc

	return clf, acc


#---------------------------------------------------------------------------------------------------------------------------------------

def complete_training(train_features_full_reduced, train_targets):

	#Retrain the model using full training dataset

	print 'Training on the complete Dataset'
	model = XGBClassifier()
	model.fit(train_features_full_reduced, train_targets)
	
	return model

#------------------------------------------------------------------------------------------------------------------------------------------

def get_final_prediction(model,test_features_reduced,test_file_path,pred_file_name):

	y_test_pred_values = model.predict(test_features_reduced)
	y_test_pred = [int(value) for value in y_test_pred_values]

	# Computing Results for Submission
	df_output = pd.DataFrame()
	aux = pd.read_csv(test_file_path)
	df_output['PassengerId'] = aux['PassengerId']
	df_output['Survived'] = y_test_pred
	df_output[['PassengerId','Survived']].to_csv(pred_file_name, index=False)

#----------------------------------------------------------------------------------------------------------------------------------------------

def apply_stratified_kfold_crossval_xgboost(train_features_full_reduced,train_targets,nsplit):

	print 'Apply Stratified Cross Validation and Grid Search'

   	parameter_grid = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
        }

	xgb = XGBClassifier(learning_rate=0.02, n_estimators=600, objective='binary:logistic',
                    silent=True, nthread=1)

	cross_validation = StratifiedKFold(n_splits=nsplit)

   	grid_search = GridSearchCV(xgb,
                              scoring='accuracy',
                              param_grid=parameter_grid,
                              cv=cross_validation,
                              verbose=1
                             )

   	grid_search.fit(train_features_full_reduced, train_targets)
   	tuned_model = grid_search
   	#parameters = grid_search.best_params_
   	#print('Best score: {}'.format(grid_search.best_score_))
   	#print('Best parameters: {}'.format(grid_search.best_params_))

   	return tuned_model
    
#--------------------------------------------------------------------------------------------------------------------