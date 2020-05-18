import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix

from my_feature_selection import my_feature_selection

def accuracy(confusion_matrix):
   diagonal_sum = confusion_matrix.trace()
   sum_of_all_elements = confusion_matrix.sum()
   return diagonal_sum / sum_of_all_elements


train_features = pd.read_csv('train_features.tsv', sep='\t')
train_labels = pd.read_csv('train_labels.tsv', sep='\t')

test_features = pd.read_csv('./data/valid_features.tsv', sep='\t')
test_labels = pd.read_csv('./data/valid_labels.tsv', sep='\t')


# Add numerical values for the categorical labels
le = preprocessing.LabelEncoder()
train_labels = le.fit_transform(train_labels['genres'])


train_features = train_features[my_feature_selection]

for i in range(1, 27, 2):
    for j in range(25, 50, 2):
        clf = MLPClassifier(solver='sgd', alpha=1e-6, \
            hidden_layer_sizes=(7, j), random_state=4, max_iter=9999)


        clf.fit(train_features, train_labels)

        y_pred = clf.predict(test_features[my_feature_selection])
        test_pred = le.fit_transform(test_labels['genres'])
        cm = confusion_matrix(y_pred, test_pred)
        #Printing the accuracy
        print("i={}, j={}. Accuracy of MLPClassifier : {}".format(i, j, accuracy(cm)))

        #test for overfitting
        train_pred = clf.predict(train_features[my_feature_selection])
        # actual_result = le.fit_transform(train_labels['genres'])
        cm = confusion_matrix(train_pred, train_labels)
        #Printing the accuracy
        print("i={}, j={}. Check overfitting of MLPClassifier : {}".format(i, j, accuracy(cm)))
        

        print()


# print(clf.coefs_[0][0])