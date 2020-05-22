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

import seaborn as sn

import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib     
matplotlib.rc('xtick', labelsize=6) 

import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
#possible genres
GENRE_LIST =['Action','Adventure','Animation','Children','Comedy','Crime','Documentary','Drama', 'Fantasy','Film noir',
            'Horror','Musical','Mystery','Romance','Sci ﬁ','Thriller','War','Western']

#Filepaths of tsv files - Note, requires all of these to run
test_features_tsv = 'test_features.tsv'

train_features_tsv = 'train_features.tsv'
train_labels_tsv = 'train_labels.tsv'

initial_test_features_tsv = 'valid_features.tsv'
initial_test_labels_tsv = 'valid_labels.tsv'



#Instance of entiment analyser class for use with senitment analysis
sid = SentimentIntensityAnalyzer()

def accuracy(confusion_matrix):
   diagonal_sum = confusion_matrix.trace()
   sum_of_all_elements = confusion_matrix.sum()
   return diagonal_sum / sum_of_all_elements

#function to output the proportion of each class correctly identified
def class_accuracy(cm):
    #for class of interest, proportion correctly identified is the entry on that row corresponding to the matrix diagonal, 
    #divided by the sum of all entries on that row
    for i in range(len(GENRE_LIST)):
        total = sum(cm[i])
        correct = cm[i][i]
        correct_proportion = correct / total
        print("correct for genre:" + GENRE_LIST[i] + ": " + str(correct_proportion) + "%")
    return 

#Perform the suntiment analysis on each tag
def row_sentiment(sent):
    #Credits
    #https://medium.com/@b.terryjack/nlp-pre-trained-sentiment-analysis-1eb52a9d742c 
    sent = " ".join(sent.split(','))
    sent = " ".join(sent.split('_'))
    sentiment = sid.polarity_scores(sent)
    return pd.Series(sentiment)

#Add the sentiment values to the dows of the dataframe
def add_sentiment_to_df(df):
    df[['neg','neu', 'pos', 'compound']] = df.apply(lambda row: row_sentiment(row['tag']), axis=1)
    return df

#Clean dirty columns
def clean_rows(test_set, validation_set=None):
    first_type = None
    a = None
    bad_rows = []
    for col in test_set:
        for j, element in enumerate(test_set[col]):
            if j==0:
                first_type = type(element)
                continue
            
            if type(element) != first_type:
                bad_rows.append(j)

            if col == 'year':
                try:
                    a = int(element)
                except:
                    bad_rows.append(j)
 

    bad_rows.sort(reverse=True)
    bad_rows = np.unique(np.array(bad_rows))

    for i in bad_rows:
        test_set = test_set.drop(i, axis=0)
        validation_set = validation_set.drop(i, axis=0)

    test_set.reset_index(drop=True, inplace=True)
    validation_set.reset_index(drop=True, inplace=True)
    
 
    return (test_set, validation_set)


#Clean the final_features data independently
def clean_final_features(final_features):
    test_set = final_features
    first_type = None
    a = None
    bad_rows = []
    for col in test_set:
        for j, element in enumerate(test_set[col]):
            if j==0:
                first_type = type(element)
                continue
            
            if type(element) != first_type:
                bad_rows.append(j)

            if col == 'year':
                try:
                    a = int(element)
                except:
                    bad_rows.append(j)

    bad_rows.sort(reverse=True)
    bad_rows = np.unique(np.array(bad_rows))
    for bad_row_num in bad_rows:
        for i, col in enumerate(test_set):
            if col == 'movieId':
                continue
            else:
                test_set.iloc[bad_row_num, i] = '0'
    return test_set

#Returns the same arrays back with numbers instead of words
def column_str_to_int(train_set, validation_set, col):
    #concatenate the training and test data
    concat_labels = pd.concat([train_set, validation_set], ignore_index=True)

    #turn genres into numbers for use by MLP classifier
    le = preprocessing.LabelEncoder()

    #A 1D array of the transformed titles
    temp = le.fit_transform(concat_labels[col])
    concat_labels[col] = temp

    #split the features and labels back into train and test sets
    features_end_index = len(train_set)
    train_set = concat_labels[:features_end_index]
    validation_set = concat_labels[features_end_index:]
    return (train_set, validation_set)

# ###### Principal componenet analysis - video
def pca_cols(df, col_names, n_cols, new_name_prefix):
    pca = PCA(n_components=n_cols)
    col_features = df[col_names]
    prinComp = pca.fit_transform(col_features)
    n_cols = [new_name_prefix+str(i) for i in range(n_cols)]
    temp = pd.DataFrame(prinComp, columns=n_cols)
    returnVal = pd.concat([df.drop(df[col_names], 1), temp], 1, sort=False)
    return returnVal

#Import the train and test data
train_features = pd.read_csv(train_features_tsv, sep='\t')
train_labels = pd.read_csv(train_labels_tsv, sep='\t')

test_features = pd.read_csv(initial_test_features_tsv, sep='\t')
test_labels = pd.read_csv(initial_test_labels_tsv, sep='\t')

final_features = pd.read_csv(test_features_tsv, sep='\t')


# #Clean the data
train_features, train_labels = clean_rows(train_features, train_labels)
test_features, test_labels = clean_rows(test_features, test_labels)

final_features = clean_final_features(final_features)


#Perform PCA on the train features twice, once on the video, and once on the audio
train_features = pca_cols(train_features, ["avf"+str(i) for i in range(1, 108)], 1, "new_vid_")
train_features = pca_cols(train_features, ["ivec"+str(i) for i in range(1, 21)], 5, "new_audio_")

test_features = pca_cols(test_features, ["avf"+str(i) for i in range(1, 108)], 1, "new_vid_")
test_features = pca_cols(test_features, ["ivec"+str(i) for i in range(1, 21)], 5, "new_audio_")

final_features = pca_cols(final_features, ["avf"+str(i) for i in range(1, 108)], 1, "new_vid_")
final_features = pca_cols(final_features, ["ivec"+str(i) for i in range(1, 21)], 5, "new_audio_")


# add sentiment analysis features to featureset
train_features = add_sentiment_to_df(train_features)
test_features = add_sentiment_to_df(test_features)
final_features = add_sentiment_to_df(final_features)


#Normalise data
transform = pd.DataFrame(StandardScaler().fit_transform(train_features.iloc[:, 5:]), columns=train_features.columns[5:])
train_features = pd.concat([train_features.iloc[:,0:5], transform], 1, sort=False)

transform = pd.DataFrame(StandardScaler().fit_transform(test_features.iloc[:, 5:]), columns=test_features.columns[5:])
test_features = pd.concat([test_features.iloc[:,0:5], transform], 1, sort=False)

transform = pd.DataFrame(StandardScaler().fit_transform(final_features.iloc[:, 5:]), columns=final_features.columns[5:])
final_features = pd.concat([final_features.iloc[:,0:5], transform], 1, sort=False)



#concatenate the training and test data for one-hot encoding and genre tokenizing
concat_features = pd.concat([train_features, test_features], ignore_index=True)
concat_final_features = pd.concat([train_features, final_features], ignore_index=True)
concat_labels = pd.concat([train_labels, test_labels], ignore_index=True)


# apply one hot encoding to tags
concat_features = pd.concat([concat_features.drop('tag', 1), concat_features['tag'].str.get_dummies(sep=",")], 1)
concat_final_features = pd.concat([concat_final_features.drop('tag', 1), concat_final_features['tag'].str.get_dummies(sep=",")], 1)



#turn genres into numbers for use by MLP classifier
le = preprocessing.LabelEncoder()
concat_labels = le.fit_transform(concat_labels['genres'])

#split the features and labels back into train and test sets
features_end_index = len(train_features)
train_features = concat_features.iloc[:features_end_index,:]
train_labels_num = concat_labels[:features_end_index]

test_features = concat_features.iloc[features_end_index:,:]
test_labels = concat_labels[features_end_index:]

final_features = concat_final_features.iloc[features_end_index:, :]


#Drop any lines that don't improve classifier performance
train_features = train_features.drop('YTId', axis=1)
test_features = test_features.drop('YTId', axis=1)
final_features = final_features.drop('YTId', axis=1)

train_features = train_features.drop('title', axis=1)
test_features = test_features.drop('title', axis=1)
final_features = final_features.drop('title', axis=1)

train_features = train_features.drop('movieId', axis=1)
test_features = test_features.drop('movieId', axis=1)
final_features = final_features.drop('movieId', axis=1)

train_features = train_features.drop('year', axis=1)
test_features = test_features.drop('year', axis=1)
final_features = final_features.drop('year', axis=1)

#Create an MLP classifier
clf = MLPClassifier(solver='sgd', 
            hidden_layer_sizes=(5,7), random_state=4, max_iter=500000, learning_rate_init = 0.1, activation = 'logistic')


#Train the model based on training features and actual classes
clf.fit(train_features, train_labels_num)


#If using labelled test data 
# print(accuracy(confusion_matrix(features_predicted, test_labels)))


# Use the trained model to predict some classes given the features
features_predicted = clf.predict(final_features)

#Format into readable genres
features_predicted = [GENRE_LIST[elem] for elem in features_predicted] 
final_features = pd.read_csv(test_features_tsv, sep='\t')
outputDf = pd.concat([final_features['movieId'], pd.DataFrame(features_predicted)], axis=1)

#Export the predictions to ass_2_out.csv
outputDf.to_csv('ass_2_out.csv')