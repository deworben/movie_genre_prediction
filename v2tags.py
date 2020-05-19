import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

train_features = pd.read_csv('train_features.tsv', sep='\t')
train_labels = pd.read_csv('train_labels.tsv', sep='\t')

test_features = pd.read_csv('valid_features.tsv', sep='\t')
test_labels = pd.read_csv('valid_labels.tsv', sep='\t')

#concatenate the training and test data
concat_features = pd.concat([train_features, test_features], ignore_index=True)
concat_labels = pd.concat([train_labels, test_labels], ignore_index=True)


#apply one-hot encoding to the tags and process the labels
concat_features = pd.concat([concat_features.drop('tag', 1), concat_features['tag'].str.get_dummies(sep=",")], 1)

#turn genres into numbers for use by MLP classifier
le = preprocessing.LabelEncoder()
concat_labels = le.fit_transform(concat_labels['genres'])


#split the features and labels back into train and test sets
features_end_index = len(train_features) - 1 
train_features = concat_features.iloc[:features_end_index,135:]
train_labels = concat_labels[:features_end_index]

test_features = concat_features.iloc[features_end_index:,135:]
test_labels = concat_labels[features_end_index:]

print(train_features)
print(train_labels)


#Initialise the model
clf = MLPClassifier(solver='sgd', 
            hidden_layer_sizes=(50,50,), random_state=4, max_iter=500000, learning_rate_init = 0.1, activation = 'logistic')

#Train the model based on training features and actual classes
clf.fit(train_features, train_labels)

#score the trained model against the validation data
print(clf.score(test_features, test_labels))
