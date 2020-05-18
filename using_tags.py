import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder

def accuracy(confusion_matrix):
   diagonal_sum = confusion_matrix.trace()
   sum_of_all_elements = confusion_matrix.sum()
   return diagonal_sum / sum_of_all_elements



train_features = pd.read_csv('train_features.tsv', sep='\t')
train_labels = pd.read_csv('train_labels.tsv', sep='\t')

test_features = pd.read_csv('./data/valid_features.tsv', sep='\t')
test_labels = pd.read_csv('./data/valid_labels.tsv', sep='\t')

# print(len(train_features["tag"].unique()))
# print(len(train_features))

# enc = OneHotEncoder(handle_unknown='ignore')
# enc.fit(train_features["tag"].values.reshape(-1, 1))
# print(enc.categories_)

train_features = pd.concat([train_features.drop('tag', 1), train_features['tag'].str.get_dummies(sep=",")], 1)
test_features = pd.concat([test_features.drop('tag', 1), test_features['tag'].str.get_dummies(sep=",")], 1)

# for i, j in enumerate(new_df.columns):
#     print(i, j)

#convert the labels to numbers
le = preprocessing.LabelEncoder()
train_labels = le.fit_transform(train_labels['genres'])

#Select only some features to be used
train_features = train_features.iloc[:, 131:150]


#are these the same numbers as the train labels? thriller = 0?
#convert the labels to numbers
test_labels = le.fit_transform(test_labels['genres'])
#Select only some features to be used
test_features = test_features.iloc[:, 131:150]


#Initialise the model
clf = MLPClassifier(solver='sgd', alpha=1e-6, \
            hidden_layer_sizes=(7, 27), random_state=4, max_iter=9999)

#Train the model based on training features and actual classes
clf.fit(train_features, train_labels)
print("done")

#Use the trained model to predict some classes given the features
features_predicted = clf.predict(test_features)

#Use the corresponding classes to find an array of classes they're meant to be
# test_pred = le.fit_transform(test_labels['genres'].iloc[0:10, :])

#Match the predicted to the actual and see how well you performed
cm = confusion_matrix(features_predicted, test_labels)

#Printing the accuracy
print("i={}, j={}. Accuracy of MLPClassifier : {}".format(7, 27, accuracy(cm)))



#test for overfitting
# train_pred = clf.predict(train_features[my_feature_selection])
# cm = confusion_matrix(train_pred, train_labels)

# #Printing theoverfitting
# print("i={}, j={}. Check overfitting of MLPClassifier : {}".format(7, 27, accuracy(cm)))

