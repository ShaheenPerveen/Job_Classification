
# coding: utf-8

# In[2]:

import numpy as np
import pandas as pd


from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution1D, MaxPooling1D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import LearningRateScheduler
from keras.layers import LSTM
from keras.layers.embeddings import Embedding

from matplotlib import pyplot as plt
get_ipython().magic('matplotlib inline')

import re

from nltk.corpus import stopwords

from bs4 import BeautifulSoup

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


# In[3]:

from sklearn.cross_validation import train_test_split


# In[11]:

JobData = pd.read_csv("/home/aiml_test_user/Shaheen/EdgeNetworksData.csv", encoding="latin1")

JobData = JobData.dropna()

labels = JobData[[4]].values.ravel()
trainTitle = JobData.iloc[:,1].values
trainDesc = JobData.iloc[:,2].values


## converting the categorical label to number
from sklearn import preprocessing 
le = preprocessing.LabelEncoder()
le.fit(labels)
labels_number = le.transform(labels)
labels_number[1]

# In[17]:


def text_to_words( raw_text ):

    # Function to convert a raw text to a string of words
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_text).get_text() 

    # 2. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 

    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()                             

    # 4. convert the stop words to a set

    stops = set(stopwords.words("english"))                  

    # 5. Remove stop words

    meaningful_words = [w for w in words if not w in stops]   

    # 6. Join the words back into one string separated by space, 
    # and return the result.

    return( " ".join( meaningful_words ))


# In[18]:


# Initialize an empty list to hold the clean Full Description

train_FullDescription = []
num_FullDescription = trainDesc.size

# Loop over each Description

for i in range( 0, num_FullDescription ):

    # Call our function for each one

    if( (i+1)%10 == 0 ):

        print ("Review %d of %d\n" % ( i+1, num_FullDescription ))

    train_FullDescription.append( text_to_words( trainDesc[i] ) )


# In[19]:

# Initialize an empty list to hold the clean Title

clean_train_Title = []
num_Title = trainTitle.size



# Loop over each Title 

for i in range( 0, num_Title ):

    # Call our function for each one, and add the result to the list

    if( (i+1)%10 == 0 ):

        print ("Review %d of %d\n" % ( i+1, num_Title ))

    clean_train_Title.append( text_to_words( trainTitle[i] ) )



# Initialize an empty list to hold the clean Full Description

clean_train_FullDescription = []
num_FullDescription = trainDesc.size



# Loop over each Description



for i in range( 0, num_FullDescription ):

    # Call our function for each one

    if( (i+1)%10 == 0 ):

        print ("Review %d of %d\n" % ( i+1, num_FullDescription ))

    clean_train_FullDescription.append( text_to_words( trainDesc[i] ) )


## combine clean title and full description

## both are combined so that they can be converted to Bag of Words



clean_train_Title_Full = [a + " " + b for a, b in zip(clean_train_Title, clean_train_FullDescription)]


# In[22]:

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

MAX_NB_WORDS = 20000
MAX_SEQUENCE_LENGTH = 5000

tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(clean_train_Title_Full)
sequences = tokenizer.texts_to_sequences(clean_train_Title_Full)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)


print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels_number.shape)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels_number[indices]
nb_validation_samples = int(0.3 * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels_number[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels_number[-nb_validation_samples:]



# In[26]:

nb_classes = 29

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_val = np_utils.to_categorical(y_val, nb_classes)


# In[32]:


embedding_vecor_length = 8

def cnn_model():
    model = Sequential()
    model.add(Embedding(len(word_index) + 1,embedding_vecor_length,input_length=MAX_SEQUENCE_LENGTH))
    model.add(Convolution1D(nb_filter=16, filter_length=3, border_mode='same', activation='relu'))
    model.add(MaxPooling1D(pool_length=2))
    model.add(Dropout(0.2))
    
    model.add(Convolution1D(nb_filter=16, filter_length=3, border_mode='same', activation='relu'))
    model.add(MaxPooling1D(pool_length=2))
    model.add(Dropout(0.2))
    
    model.add(LSTM(50))
    model.add(Dense(nb_classes, activation='softmax'))
    return model

model = cnn_model()
model1 = cnn_model()
lr = 0.001
sgd = SGD(lr=lr, decay=1e-5, momentum=0.9, nesterov=True)
model1.compile(loss='categorical_crossentropy',
          optimizer=sgd,
          metrics=['accuracy'])

model.compile(loss='categorical_crossentropy',
          optimizer='adadelta',
          metrics=['accuracy'])


#def lr_schedule(epoch):
    #return lr*(0.1**int(epoch/10))

print(model1.summary())


# In[ ]:

## fitting the model
batch_size = 128
nb_epoch = 5

model1.fit(x_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(x_val, Y_val))


# In[ ]:



