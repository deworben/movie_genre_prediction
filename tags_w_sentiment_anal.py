import nltk
# nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np

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


sid = SentimentIntensityAnalyzer()
def row_sentiment(sent):
    sent = " ".join(sent.split(','))
    sent = " ".join(sent.split('_'))
    sentiment = sid.polarity_scores(sent)
    return pd.Series(sentiment)

def add_sentiment_to_df(df):
    df[['neg','neu', 'pos', 'compound']] = df.apply(lambda row: row_sentiment(row['tag']), axis=1)
    return df

X_train = pd.read_csv('train_features.tsv', sep='\t')
X_train = X_train[['tag', 'title', 'movieId']]
Y_train = pd.read_csv('train_labels.tsv', sep='\t')

X_val = pd.read_csv('valid_features.tsv', sep='\t')
X_val = X_val[['tag', 'title', 'movieId']]
Y_val = pd.read_csv('valid_labels.tsv', sep='\t')

# add sentiment analysis features to featureset
X_train = add_sentiment_to_df(X_train)
print(X_train)
X_train = X_train.drop('tag', axis=1)

# add sentiment analysis features to featureset
X_val = add_sentiment_to_df(X_val)
X_val = X_val.drop('tag', axis=1)

def clean_rows(test_set, validation_set):
    first_type = None
    bad_rows = []
    for col in test_set:
        for j, element in enumerate(test_set[col]):
            if j==0:
                first_type = type(element)
                continue
            
            if type(element) != first_type:
                bad_rows.append(j)
                # print(col + " " + str(j) + " " + str(element))
    bad_rows.sort(reverse=True)
    for i in bad_rows:
        test_set = test_set.drop(i, axis=0)
        validation_set = validation_set.drop(i, axis=0)
    return (test_set, validation_set)

X_train, Y_train = clean_rows(X_train, Y_train)
X_val, Y_val = clean_rows(X_val, Y_val)


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

#Tokenise these lines
Y_train, Y_val = column_str_to_int(Y_train, Y_val, 'genres')
X_train,X_val = column_str_to_int(X_train, X_val, 'title')
# X_train,X_val = column_str_to_int(X_train, X_val, 'tag')


#Put into right shape
Y_train = np.array(Y_train['genres'])
Y_val = np.array(Y_val['genres'])

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
        clf.fit(X_train, Y_train)



        #Use the trained model to predict some classes given the features
        features_predicted = clf.predict(X_val)
        cm = confusion_matrix(features_predicted, Y_val)
        print("i={}, j={}. Accuracy of MLPClassifier : {}".format(i, j, accuracy(cm)))


        #test for overfitting
        train_pred = clf.predict(X_train)
        cm = confusion_matrix(train_pred, Y_train)
        print("i={}, j={}. Check overfitting of MLPClassifier : {}".format(i, j, accuracy(cm)))
        print()



        
