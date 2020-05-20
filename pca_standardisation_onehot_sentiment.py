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


import nltk
# nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer


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

sid = SentimentIntensityAnalyzer()
def row_sentiment(sent):
    #Credits
    #https://medium.com/@b.terryjack/nlp-pre-trained-sentiment-analysis-1eb52a9d742c 
    sent = " ".join(sent.split(','))
    sent = " ".join(sent.split('_'))
    sentiment = sid.polarity_scores(sent)
    return pd.Series(sentiment)

def add_sentiment_to_df(df):
    df[['neg','neu', 'pos', 'compound']] = df.apply(lambda row: row_sentiment(row['tag']), axis=1)
    return df


def clean_rows(test_set, validation_set):
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
                # print(col + " " + str(j) + " " + str(element))

            if col == 'year':
                try:
                    a = int(element)
                except:
                    bad_rows.append(j)
 

    bad_rows.sort(reverse=True)
    bad_rows = np.unique(np.array(bad_rows))
    print(bad_rows)
    for i in bad_rows:
        test_set = test_set.drop(i, axis=0)
        validation_set = validation_set.drop(i, axis=0)
    test_set.reset_index(drop=True, inplace=True)
    validation_set.reset_index(drop=True, inplace=True)
    return (test_set, validation_set)



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
    # print(pca.explained_variance_ratio_)


train_features = pd.read_csv('train_features.tsv', sep='\t')
train_labels = pd.read_csv('train_labels.tsv', sep='\t')

test_features = pd.read_csv('valid_features.tsv', sep='\t')
test_labels = pd.read_csv('valid_labels.tsv', sep='\t')


# #Clean the data
train_features, train_labels = clean_rows(train_features, train_labels)
test_features, test_labels = clean_rows(test_features, test_labels)


#Perform PCA on the train features twice, once on the video, and once on the audio
train_features = pca_cols(train_features, ["avf"+str(i) for i in range(1, 108)], 1, "new_vid_")
train_features = pca_cols(train_features, ["ivec"+str(i) for i in range(1, 21)], 2, "new_audio_")

test_features = pca_cols(test_features, ["avf"+str(i) for i in range(1, 108)], 1, "new_vid_")
test_features = pca_cols(test_features, ["ivec"+str(i) for i in range(1, 21)], 2, "new_audio_")

# add sentiment analysis features to featureset
train_features = add_sentiment_to_df(train_features)
test_features = add_sentiment_to_df(test_features)


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

# train_labels, test_labels = column_str_to_int(train_labels, test_labels, 'genres')
train_features,test_features = column_str_to_int(train_features, test_features, 'title')


#turn genres into numbers for use by MLP classifier
le = preprocessing.LabelEncoder()
concat_labels = le.fit_transform(concat_labels['genres'])
# print(concat_labels)

#split the features and labels back into train and test sets
features_end_index = len(train_features) - 1 
train_features = concat_features.iloc[:features_end_index,:]
train_labels = concat_labels[:features_end_index]
# print(train_features.columns)

test_features = concat_features.iloc[features_end_index:,:]
test_labels = concat_labels[features_end_index:]


#Tokenise these lines
# Y_train, Y_val = column_str_to_int(Y_train, Y_val, 'genres')
train_features = train_features.drop('YTId', axis=1)
test_features = test_features.drop('YTId', axis=1)

train_features,test_features = column_str_to_int(train_features, test_features, 'title')

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
