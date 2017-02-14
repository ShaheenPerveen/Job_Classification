
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from bs4 import BeautifulSoup

train = pd.read_csv("finaltrain.csv", sep=",", encoding="latin1")
test = pd.read_csv("finaltest.csv", sep=",", header=0, encoding="latin1")

def text_to_words( raw_text ):

    # Function to convert a raw text to a string of words
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_text).get_text() 
    
    # 2. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    
    #3. Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    
    # 4. convert the stop words to a set
    stops = set(stopwords.words("english"))                  
    
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]   
    
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( meaningful_words ))



# Initialize an empty list to hold the clean Title

clean_train_Title = []
num_Title = train["Title"].size



# Loop over each Title 

for i in range( 0, num_Title ):

   # Call our function for each one, and add the result to the list
    if( (i+1)%10 == 0 ):

        print ("Review %d of %d\n" % ( i+1, num_Title ))
    clean_train_Title.append( text_to_words( train["Title"].iloc[i] ) )



# Initialize an empty list to hold the clean Full Description

clean_train_FullDescription = []
num_FullDescription = train["FullDescription"].size



# Loop over each Description



for i in range( 0, num_FullDescription ):

    # Call our function for each one

    if( (i+1)%10 == 0 ):

        print ("Review %d of %d\n" % ( i+1, num_FullDescription ))

    clean_train_FullDescription.append( text_to_words( train["FullDescription"].iloc[i] ) )





## combine clean title and full description

## both are combined so that they can be converted to Bag of Words
clean_train_Title_Full = [a + " " + b for a, b in zip(clean_train_Title, clean_train_FullDescription)]

############# 	TEST DATA PREPROCESSING 
# Initialize an empty list to hold the clean title

clean_test_Title = []
num_Title = test["Title"].size

# Loop over each Title

for i in range( 0, num_Title ):

    # Call our function for each one

    if( (i+1)%10 == 0 ):

        print ("Review %d of %d\n" % ( i+1, num_Title ))

    clean_test_Title.append( text_to_words( test["Title"].iloc[i] ) )



# Initialize an empty list to hold the clean Full Description

clean_test_FullDescription = []
num_Full = test["FullDescription"].size

# Loop over each review; create an index i that goes from 0 to the length

# of the movie review list 

for i in range( 0, num_Full ):

    # Call our function for each one,

    if( (i+1)%10 == 0 ):

        print ("Review %d of %d\n" % ( i+1, num_Full ))

    clean_test_FullDescription.append( text_to_words( test["FullDescription"].iloc[i] ) )


## combine clean title and full description

## both are combined so that they can be converted to Bag of Words
clean_test_Title_Full = [a + " " + b for a, b in zip(clean_test_Title, clean_test_FullDescription)]


## vectorizing
from sklearn.feature_extraction.text import CountVectorizer

# Initialize the "CountVectorizer" object

# bag of words tool.  

vectorizer = CountVectorizer(analyzer = "word",tokenizer = None, preprocessor = None, stop_words = None, max_features = 6000) 

# fit_transform() does two functions: First, it fits the model

# and learns the vocabulary; second, it transforms our training data

# into feature vectors. The input to fit_transform should be a list of 

# strings.
clean_train_Title_Full = vectorizer.fit_transform(clean_train_Title_Full)

# Numpy arrays are easy to work with, so convert the result to an 

# array
clean_train_Title_Full = clean_train_Title_Full.toarray()

## test data
# Initialize the "CountVectorizer" object, which is scikit-learn's

# bag of words tool.  

vectorizer = CountVectorizer(analyzer = "word",Tokenizer = None, preprocessor = None, stop_words = None, max_features = 5000) 

# fit_transform() does two functions: First, it fits the model

# and learns the vocabulary; second, it transforms our training data

# into feature vectors. The input to fit_transform should be a list of 

# strings.

test_data_features = vectorizer.fit_transform(clean_test_Title_Full)

# Numpy arrays are easy to work with, so convert the result to an 

# array

test_data_features = test_data_features.toarray()

test_data_features.shape

## creating class weight dictionary

weights = train["Category1"].value_counts()

newdict = {}

for key, value in weights.items():

     newdict[key] = 1/(float(weights[key])/45058)



newdict

from sklearn.ensemble import RandomForestClassifier

# Initialize a Random Forest classifier with 50 trees

forest = RandomForestClassifier(n_estimators = 50, max_features=2000, class_weight ='balanced', verbose=2) 

# Fit the forest to the training set, using the bag of words as 

# features and the sentiment labels as the response variable

#

# This may take a few minutes to run

forest = forest.fit( train_data_features, train["Category1"] )

result = forest.predict(test_data_features)



## constructing confusion matrix to check performance

y_actu = pd.Series(test["Category1"], name='Actual')

y_pred = pd.Series(result, name='Predicted')

df_confusion = pd.crosstab(y_actu, y_pred)

df_confusion



# Initialize a Random Forest classifier with 50 trees

forest = RandomForestClassifier(n_estimators = 50, max_features=2000, class_weight =newdict, verbose=2)

# Fit the forest to the training set, using the bag of words as 

# features and the sentiment labels as the response variable

#
# This may take a few minutes to run

forest = forest.fit( train_data_features, train["Category1"] )
result = forest.predict(test_data_features)

from sklearn.feature_extraction.text import TfidfVectorizer
## Using TF_IDF to vectorize

tf = TfidfVectorizer(analyzer='word', min_df = 0, stop_words = None, max_features = 6000)

# fit_transform() does two functions: First, it fits the model

# and learns the vocabulary; second, it transforms our training data

# into feature vectors. The input to fit_transform should be a list of 

# strings.

train_data_features = tf.fit_transform(clean_train_Title_Full)

# Numpy arrays are easy to work with, so convert the result to an 

# array
train_data_features = train_data_features.toarray()
train_data_features.shape

# Initialize a Random Forest classifier with 50 trees

forest = RandomForestClassifier(n_estimators = 50, max_features=2000, class_weight ='balanced', verbose=2) 

# Fit the forest to the training set, using the bag of words as 

# features and the sentiment labels as the response variable

#

# This may take a few minutes to run

forest = forest.fit( train_data_features, train["Category1"] )
result = forest.predict(test_data_features)



## constructing confusion matrix to check performance

y_actu = pd.Series(test["Category1"], name='Actual')

y_pred = pd.Series(result, name='Predicted')

df_confusion = pd.crosstab(y_actu, y_pred)

df_confusion



# Initialize a Random Forest classifier with 50 trees
forest = RandomForestClassifier(n_estimators = 50, max_features=2000, class_weight =newdict, verbose=2)



# Fit the forest to the training set, using the bag of words as 

# features and the sentiment labels as the response variable

#

# This may take a few minutes to run

forest = forest.fit( train_data_features, train["Category1"] )
result = forest.predict(test_data_features)





# Copy the results to a pandas dataframe 

output = pd.DataFrame( data={"Actual":test["Category1"], "Predicted":result} )

output.to_csv( "EdgeA1_RF_Bigram_ClassWeight.csv", index=False, quoting=3 )



y_actu = pd.Series(test["Category1"], name='Actual')

y_pred = pd.Series(result, name='Predicted')

df_confusion = pd.crosstab(y_actu, y_pred)

df_confusion



## attempts with tf-idf vectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

## Using TF_IDF to vectorizer

tf = TfidfVectorizer(analyzer='word', min_df = 0, stop_words = None, max_features = 6000)

# fit_transform() does two functions: First, it fits the model

# and learns the vocabulary; second, it transforms our training data

# into feature vectors. The input to fit_transform should be a list of 

# strings.

train_data_features = tf.fit_transform(clean_train_Title_Full)

# Numpy arrays are easy to work with, so convert the result to an 

# array

train_data_features = train_data_features.toarray()

train_data_features.shape


## Using TF_IDF to vectorizer for test data

tf = TfidfVectorizer(analyzer='word', min_df = 0, stop_words = None, max_features = 6000)


# fit_transform() does two functions: First, it fits the model

# and learns the vocabulary; second, it transforms our training data

# into feature vectors. The input to fit_transform should be a list of 

# strings.

test_data_features = tf.fit_transform(clean_test_Title_Full)



# Numpy arrays are easy to work with, so convert the result to an 

# array

test_data_features = test_data_features.toarray()

test_data_features.shape



## attempts with Random Forest
# Initialize a Random Forest classifier with 50 trees

forest = RandomForestClassifier(n_estimators = 50, max_features=2000, class_weight ='balanced', verbose=2) 

# Fit the forest to the training set, using the bag of words as 

# features and the sentiment labels as the response variable
# This may take a few minutes to run

forest = forest.fit( train_data_features, train["Category1"] )
result = forest.predict(test_data_features)

## constructing confusion matrix to check performance

y_actu = pd.Series(test["Category1"], name='Actual')

y_pred = pd.Series(result, name='Predicted')

df_confusion = pd.crosstab(y_actu, y_pred)

df_confusion


# Initialize a Random Forest classifier with 50 trees

forest = RandomForestClassifier(n_estimators = 50, max_features=2000, class_weight =newdict, verbose=2)


# Fit the forest to the training set, using the bag of words as 
# features and the sentiment labels as the response variable

# This may take a few minutes to run

forest = forest.fit( train_data_features, train["Category1"] )
result = forest.predict(test_data_features)



## constructing confusion matrix to check performance

y_actu = pd.Series(test["Category1"], name='Actual')

y_pred = pd.Series(result, name='Predicted')

df_confusion = pd.crosstab(y_actu, y_pred)

df_confusion


#### ATTEMPTS WITH NAIVE BAYES
#Create a Naive Bayes Gaussian Classifier 
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB() 

# Train the model using the training sets 

gnb = gnb.fit(train_data_features, train["Category1"]) 
result = gnb.predict(test_data_features)



#Create a Naive Bayes Multinomial Classifier 
from sklearn.naive_bayes import MultinomialNB

mnb = MultinomialNB(alpha=0.5) 

# Train the model using the training sets 

mnb = mnb.fit(train_data_features, train["Category1"]) 
result = mnb.predict(test_data_features)






###################################################################################################################################################################
############## BIGRAM VECTORIZER



bigram_vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=1, max_features = 4000)



analyze = bigram_vectorizer.build_analyzer()



## converting train and test data to bigram model

train_data_features = bigram_vectorizer.fit_transform(clean_train_Title_Full).toarray()

train_data_features.shape



test_data_features = bigram_vectorizer.fit_transform(clean_test_Title_Full).toarray()

test_data_features.shape



## using Gaussian Naive bayes



gnb = GaussianNB() 

# Train the model using the training sets 

gnb = gnb.fit(train_data_features, train["Category1"]) 



result = gnb.predict(test_data_features)



#Create a Naive Bayes Multinomial Classifier 



## using multinomial naive bayes



mnb = MultinomialNB(alpha=0.5) 

# Train the model using the training sets 

mnb = mnb.fit(train_data_features, train["Category1"]) 



result = mnb.predict(test_data_features)





## applying Random forest on bigram



# Initialize a Random Forest classifier with 100 trees

forest = RandomForestClassifier(n_estimators = 70, max_features=2000, class_weight =newdict, verbose=2)



# Fit the forest to the training set, using the bag of words as 

# features and the sentiment labels as the response variable

#

# This may take a few minutes to run

forest = forest.fit( train_data_features, train["Category1"] )



result = forest.predict(test_data_features)



## constructing confusion matrix to check performance

y_actu = pd.Series(test["Category1"], name='Actual')

y_pred = pd.Series(result, name='Predicted')

df_confusion = pd.crosstab(y_actu, y_pred)

df_confusion




