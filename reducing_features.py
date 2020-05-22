import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn import svm

def accuracy(confusion_matrix):
   diagonal_sum = confusion_matrix.trace()
   sum_of_all_elements = confusion_matrix.sum()
   return diagonal_sum / sum_of_all_elements

#function to output the proportion of each class correctly identified
def class_accuracy(confusion):
    #possible genres
    genre_list =['Action','Adventure','Animation','Children','Comedy','Crime','Documentary','Drama', 'Fantasy','Film noir',
             'Horror','Musical','Mystery','Romance','Sci ﬁ','Thriller','War','Western']

    #for class of interest, proportion correctly identified is the entry on that row corresponding to the matrix diagonal, 
    #divided by the sum of all entries on that row
    for i in range(18):
        total = sum(confusion[i])
        correct = confusion[i][i]
        correct_proportion = correct / total
        print("correct for genre:" + genre_list[i] + ": " + str(correct_proportion) + "%")
    return 

train_features = pd.read_csv('train_features.tsv', sep='\t')
train_labels = pd.read_csv('train_labels.tsv', sep='\t')

test_features = pd.read_csv('valid_features.tsv', sep='\t')
test_labels = pd.read_csv('valid_labels.tsv', sep='\t')


# ###### Principal componenet analysis - video
def pca_cols(df, col_names, n_cols, new_name_prefix):
    pca = PCA(n_components=n_cols)
    col_features = df[col_names]
    prinComp = pca.fit_transform(col_features)
    n_cols = [new_name_prefix+str(i) for i in range(n_cols)]
    temp = pd.DataFrame(prinComp, columns=n_cols)
    # temp = pd.DataFrame(x, columns=train_features.columns[5:])
    # train_features = pd.concat([train_features.drop(train_features.columns[5:], 1), temp], 1, sort=False)
    return pd.concat([df.drop(df[col_names], 1), temp], 1, sort=False)
    # print(pca.explained_variance_ratio_)


###### Standardise all the numerical data
#Perform PCA on the train features twice, once on the video, and once on the audio
train_features = pca_cols(train_features, ["avf"+str(i) for i in range(1, 108)], 1, "new_vid_")
train_features = pca_cols(train_features, ["ivec"+str(i) for i in range(1, 21)], 2, "new_audio_")

test_features = pca_cols(test_features, ["avf"+str(i) for i in range(1, 108)], 1, "new_vid_")
test_features = pca_cols(test_features, ["ivec"+str(i) for i in range(1, 21)], 2, "new_audio_")



# #Normalise data
transform = pd.DataFrame(StandardScaler().fit_transform(train_features.iloc[:, 5:]), columns=train_features.columns[5:])
train_features = pd.concat([train_features.iloc[:,0:5], transform], 1, sort=False)

transform = pd.DataFrame(StandardScaler().fit_transform(test_features.iloc[:, 5:]), columns=test_features.columns[5:])
test_features = pd.concat([test_features.iloc[:,0:5], transform], 1, sort=False)




#concatenate the training and test data
concat_features = pd.concat([train_features, test_features], ignore_index=True)
concat_labels = pd.concat([train_labels, test_labels], ignore_index=True)

# apply one hot encoding to tags
concat_features = pd.concat([concat_features.drop('tag', 1), concat_features['tag'].str.get_dummies(sep=",")], 1)


#turn genres into numbers for use by MLP classifier
le = preprocessing.LabelEncoder()
concat_labels = le.fit_transform(concat_labels['genres'])
# print(concat_labels)

#split the features and labels back into train and test sets
features_end_index = len(train_features) - 1 
train_features = concat_features.iloc[:features_end_index,5:]
train_labels = concat_labels[:features_end_index]
# print(train_features.columns)

test_features = concat_features.iloc[features_end_index:,5:]
test_labels = concat_labels[features_end_index:]


for i in range(5, 12):
    for j in range(5, 12):

        if(j > i*1.5):
            continue
        #Initialise the model
        clf = MLPClassifier(solver='sgd', 
                    hidden_layer_sizes=(i,j), random_state=4, max_iter=500000, learning_rate_init = 0.1, activation = 'logistic')
        # clf = svm.LinearSVC()
        # clf = svm.SVC(decision_function_shape='ovo')
        # clf = svm.SVC()

        #Train the model based on training features and actual classes
        clf.fit(train_features, train_labels)



        #Use the trained model to predict some classes given the features
        features_predicted = clf.predict(test_features)
        cm = confusion_matrix(features_predicted, test_labels)
        print("i={}, j={}. Accuracy of MLPClassifier : {}".format(i, j, accuracy(cm)))

    
        # #call the function for the audio data
        # print("by-class accuracy for the audio data:\n")
        # class_accuracy(cm)


        #test for overfitting
        train_pred = clf.predict(train_features)
        cm = confusion_matrix(train_pred, train_labels)
        print("i={}, j={}. Check overfitting of MLPClassifier : {}".format(i, j, accuracy(cm)))
        print()



        

