import numpy as np 
import pandas as pd 

#-----------------------------------------------------------------------------------------------------------------------

def get_merged_data(train_file_path,test_file_path):
    # reading train data
    train = pd.read_csv(train_file_path)
    
    # reading test data
    test = pd.read_csv(test_file_path)

    # extracting and then removing the targets from the training data 
    targets = train.Survived
    train.drop(['Survived'], 1, inplace=True)
    

    # merging train data and test data for future feature engineering
    # we'll also remove the PassengerID since this is not an informative feature
    merged = train.append(test)
    merged.reset_index(inplace=True)
    merged.drop(['index', 'PassengerId'], inplace=True, axis=1)

    return merged

#-------------------------------------------------------------------------------------------------------------------------

def get_titles(data):

	Title_Dictionary = {
    "Capt": "Officer",
    "Col": "Officer",
    "Major": "Officer",
    "Jonkheer": "Royalty",
    "Don": "Royalty",
    "Sir" : "Royalty",
    "Dr": "Officer",
    "Rev": "Officer",
    "the Countess":"Royalty",
    "Mme": "Mrs",
    "Mlle": "Miss",
    "Ms": "Mrs",
    "Mr" : "Mr",
    "Mrs" : "Mrs",
    "Miss" : "Miss",
    "Master" : "Master",
    "Lady" : "Royalty"}

	# we extract the title from each name
	data['Title'] = data['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
    
	# a map of more aggregated title
	# we map each title
	data['Title'] = data.Title.map(Title_Dictionary)

	return data

#-----------------------------------------------------------------------------------------------------------------------------

def fill_age(row,grouped_median_train_data):

    condition = (
        (grouped_median_train_data['Sex'] == row['Sex']) & 
        (grouped_median_train_data['Title'] == row['Title']) & 
        (grouped_median_train_data['Pclass'] == row['Pclass'])
    ) 
    return grouped_median_train_data[condition]['Age'].values[0]

#----------------------------------------------------------------------------------------------------------------------------

def processing_age_features(data,num_train_obs):

	grouped_train_data = data.iloc[:num_train_obs].groupby(['Sex','Pclass','Title'])
	grouped_median_train_data = grouped_train_data.median()
	grouped_median_train_data = grouped_median_train_data.reset_index()[['Sex', 'Pclass', 'Title', 'Age']]
    
    # a function that fills the missing values of the Age variable
	data['Age'] = data.apply(lambda row: fill_age(row,grouped_median_train_data) if np.isnan(row['Age']) else row['Age'], axis=1)
	return data

#----------------------------------------------------------------------------------------------------------------------------

def processing_fares_features(data,num_train_obs):
    
    # there's one missing fare value - replacing it with the mean.
    data.Fare.fillna(data.iloc[:num_train_obs].Fare.mean(), inplace=True)
    return data

#----------------------------------------------------------------------------------------------------------------------------

def processing_embarked_features(data):
    
    # two missing embarked values - filling them with the most frequent one in the train  set(S)
    data.Embarked.fillna('S', inplace=True)
    # dummy encoding 
    embarked_dummies = pd.get_dummies(data['Embarked'], prefix='Embarked')
    data = pd.concat([data, embarked_dummies], axis=1)
    data.drop('Embarked', axis=1, inplace=True)
  
    return data

#----------------------------------------------------------------------------------------------------------------------------

def processing_names_features(data):
    
    # we clean the Name variable
    data.drop('Name', axis=1, inplace=True)
    
    # encoding in dummy variable
    titles_dummies = pd.get_dummies(data['Title'], prefix='Title')
    data = pd.concat([data, titles_dummies], axis=1)
    
    # removing the title variable
    data.drop('Title', axis=1, inplace=True)
    
    return data


#-----------------------------------------------------------------------------------------------------------------------------

def processing_sex_features(data):

    # mapping string values to numerical one 
    data['Sex'] = data['Sex'].map({'male':1, 'female':0})
    return data

#------------------------------------------------------------------------------------------------------------------------------

def processing_pclass_features(data):
    
    # encoding into 3 categories:
    pclass_dummies = pd.get_dummies(data['Pclass'], prefix="Pclass")
    
    # adding dummy variable
    data = pd.concat([data, pclass_dummies],axis=1)
    
    # removing "Pclass"
    data.drop('Pclass',axis=1,inplace=True)
   
    return data

#---------------------------------------------------------------------------------------------------------------------------------

# a function that extracts each prefix of the ticket, returns 'XXX' if no prefix (i.e the ticket is a digit)
def cleanTicket(ticket):
	ticket = ticket.replace('.','')
	ticket = ticket.replace('/','')
	ticket = ticket.split()
	ticket = map(lambda t : t.strip(), ticket)
	ticket = filter(lambda t : not t.isdigit(), ticket)
	if len(ticket) > 0:
		return ticket[0]
	else:
		return 'XXX'

#--------------------------------------------------------------------------------------------------------------------------------

def processing_ticket_features(data):
    
    # Extracting dummy variables from tickets:

    data['Ticket'] = data['Ticket'].apply(lambda x: cleanTicket(x))
    tickets_dummies = pd.get_dummies(data['Ticket'], prefix='Ticket')
    data = pd.concat([data, tickets_dummies], axis=1)
    data.drop('Ticket', inplace=True, axis=1)

    return data

#-------------------------------------------------------------------------------------------------------------------------------------

def processing_family_features(data):
    
    # introducing a new feature : the size of families (including the passenger)
    data['FamilySize'] = data['Parch'] + data['SibSp'] + 1
    
    # introducing other features based on the family size
    data['Singleton'] = data['FamilySize'].map(lambda s: 1 if s == 1 else 0)
    data['SmallFamily'] = data['FamilySize'].map(lambda s: 1 if 2 <= s <= 4 else 0)
    data['LargeFamily'] = data['FamilySize'].map(lambda s: 1 if 5 <= s else 0)
    
    return data

#---------------------------------------------------------------------------------------------------------------------------------------

def processing_cabin_features(data):
    
    # replacing missing cabins with U (for Uknown)
    data.Cabin.fillna('U', inplace=True)
    
    # mapping each Cabin value with the cabin letter
    data['Cabin'] = data['Cabin'].map(lambda c: c[0])
    
    # dummy encoding ...
    cabin_dummies = pd.get_dummies(data['Cabin'], prefix='Cabin')    
    data = pd.concat([data, cabin_dummies], axis=1)

    data.drop('Cabin', axis=1, inplace=True)
    
    return data

#------------------------------------------------------------------------------------------------------------------

def recover_train_test_target(merged_data,train_data,num_train_obs):
    
    targets = train_data['Survived'].values
    train = merged_data.iloc[:num_train_obs]
    test = merged_data.iloc[num_train_obs:]
    
    return train, test, targets

#--------------------------------------------------------------------------------------------------------------------