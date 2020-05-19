import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA

def accuracy(confusion_matrix):
   diagonal_sum = confusion_matrix.trace()
   sum_of_all_elements = confusion_matrix.sum()
   return diagonal_sum / sum_of_all_elements


train_features = pd.read_csv('train_features.tsv', sep='\t')
train_labels = pd.read_csv('train_labels.tsv', sep='\t')

test_features = pd.read_csv('valid_features.tsv', sep='\t')
test_labels = pd.read_csv('valid_labels.tsv', sep='\t')


# print(train_labels['genres'].value_counts())

###### Standardisation
x = StandardScaler().fit_transform(train_features.iloc[:, 5:])
# ###### Principal componenet analysis - video
vid_n_comp = 50
pca = PCA(n_components=vid_n_comp)
train_video_features = train_features[["avf"+str(i) for i in range(1, 107)]]
prinComp = pca.fit_transform(train_video_features)
n_cols = ["new_vid_"+str(i) for i in range(vid_n_comp)]
temp = pd.DataFrame(prinComp, columns=n_cols)
# temp = pd.DataFrame(x, columns=train_features.columns[5:])
# train_features = pd.concat([train_features.drop(train_features.columns[5:], 1), temp], 1, sort=False)
train_features = pd.concat([train_features.drop(train_features[["avf"+str(i) for i in range(1, 108)]], 1), temp], 1, sort=False)
print(pca.explained_variance_ratio_)

# ###### Principal componenet analysis - audio
audio_n_comp = 20
pca = PCA(n_components=audio_n_comp)
train_audio_features = train_features[["ivec"+str(i) for i in range(1, 21)]]
prinComp = pca.fit_transform(train_audio_features)
n_cols = ["new_audio_"+str(i) for i in range(audio_n_comp)]
temp = pd.DataFrame(prinComp, columns=n_cols)
# temp = pd.DataFrame(x, columns=train_features.columns[5:])
# train_features = pd.concat([train_features.drop(train_features.columns[5:], 1), temp], 1, sort=False)
train_features = pd.concat([train_features.drop(train_features[["ivec"+str(i) for i in range(1, 21)]], 1), temp], 1, sort=False)
print(train_features.columns)
print(pca.explained_variance_ratio_)






#standardise output data
x = StandardScaler().fit_transform(test_features.iloc[:, 5:])
###### Principal componenet analysis
# prinComp = pca.fit_transform(x)
# temp = pd.DataFrame(prinComp, columns=n_cols)
# temp = pd.DataFrame(x, columns=test_features.columns[5:])
# test_features = pd.concat([test_features.drop(test_features.columns[5:], 1), temp], 1, sort=False)
pca = PCA(n_components=vid_n_comp)
test_video_features = test_features[["avf"+str(i) for i in range(1, 107)]]
prinComp = pca.fit_transform(test_video_features)
n_cols = ["new_vid_"+str(i) for i in range(vid_n_comp)]
temp = pd.DataFrame(prinComp, columns=n_cols)
# temp = pd.DataFrame(x, columns=train_features.columns[5:])
# train_features = pd.concat([train_features.drop(train_features.columns[5:], 1), temp], 1, sort=False)
test_features = pd.concat([test_features.drop(test_features[["avf"+str(i) for i in range(1, 108)]], 1), temp], 1, sort=False)
print(pca.explained_variance_ratio_)

# ###### Principal componenet analysis - audio
pca = PCA(n_components=audio_n_comp)
train_audio_features = test_features[["ivec"+str(i) for i in range(1, 21)]]
prinComp = pca.fit_transform(train_audio_features)
n_cols = ["new_audio_"+str(i) for i in range(audio_n_comp)]
temp = pd.DataFrame(prinComp, columns=n_cols)
# temp = pd.DataFrame(x, columns=train_features.columns[5:])
# train_features = pd.concat([train_features.drop(train_features.columns[5:], 1), temp], 1, sort=False)
test_features = pd.concat([test_features.drop(test_features[["ivec"+str(i) for i in range(1, 21)]], 1), temp], 1, sort=False)
print(pca.explained_variance_ratio_)








#concatenate the training and test data
concat_features = pd.concat([train_features, test_features], ignore_index=True)
concat_labels = pd.concat([train_labels, test_labels], ignore_index=True)

# #apply one-hot encoding to the tags and process the labels
concat_features = pd.concat([concat_features.drop('tag', 1), concat_features['tag'].str.get_dummies(sep=",")], 1)
# tags_n_comp = 20
# pca = PCA(n_components=tags_n_comp)
# train_tags_features = concat_features.iloc[:, 5+audio_n_comp+vid_n_comp:]
# prinComp = pca.fit_transform(train_tags_features)
# n_cols = ["new_tags_"+str(i) for i in range(tags_n_comp)]
# temp = pd.DataFrame(prinComp, columns=n_cols)
# concat_features = pd.concat([concat_features.drop(concat_features.iloc[:, 5+audio_n_comp+vid_n_comp:], 1), temp], 1, sort=False)
# print("tags")
# print(pca.explained_variance_ratio_)
# print(concat_features.columns)



#turn genres into numbers for use by MLP classifier
le = preprocessing.LabelEncoder()
concat_labels = le.fit_transform(concat_labels['genres'])


#split the features and labels back into train and test sets
features_end_index = len(train_features) - 1 
train_features = concat_features.iloc[:features_end_index,5:]
train_labels = concat_labels[:features_end_index]
print(train_features.columns)

test_features = concat_features.iloc[features_end_index:,5:]
test_labels = concat_labels[features_end_index:]




for i in range(5, 12):
    for j in range(5, 12):

        if(j > i*1.5):
            continue
        #Initialise the model
        clf = MLPClassifier(solver='sgd', 
                    hidden_layer_sizes=(i,j), random_state=4, max_iter=500000, learning_rate_init = 0.1, activation = 'logistic')

        #Train the model based on training features and actual classes
        clf.fit(train_features, train_labels)

        #Use the trained model to predict some classes given the features
        features_predicted = clf.predict(test_features)
        cm = confusion_matrix(features_predicted, test_labels)
        print("i={}, j={}. Accuracy of MLPClassifier : {}".format(i, j, accuracy(cm)))



        #test for overfitting
        train_pred = clf.predict(train_features)
        cm = confusion_matrix(train_pred, train_labels)
        print("i={}, j={}. Check overfitting of MLPClassifier : {}".format(i, j, accuracy(cm)))
        print()
