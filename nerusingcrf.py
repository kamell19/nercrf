#!/usr/bin/env python
# coding: utf-8

# # Named Entity Recognition using CRF model
# Total per tag
# 
# * B-PER 2508
# * I-PER 3111
# * B-ADJ 402 
# * I-ADJ 442
# * B-ANM 2556
# * I-ANM 2478
# * B-GODS 467
# * I-GODS 549
# * B-OBJ 1661
# * I-OBJ 986
# * O 74768

# #### Importing Libraries

# In[1]:


import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn_crfsuite import CRF
from sklearn_crfsuite.metrics import flat_f1_score
from sklearn_crfsuite.metrics import flat_classification_report


# In[2]:


# Membaca file 
df = pd.read_excel('ner_dataset.xlsx')


# In[3]:


#Display first 10 rows
df.head(10)


# In[4]:


df.describe()


# #### Observations : 
# * There are total 47959 sentences in the dataset.
# * Number unique words in the dataset are 35178.
# * Total 17 lables (Tags).

# In[5]:


#Displaying the unique Tags
df['Tag'].unique()


# In[6]:


#Checking null values, if any.
df.isnull().sum()


# There are lots of missing values in 'Sentence #' attribute. So we will use pandas fillna technique and use 'ffill' method which propagates last valid observation forward to next.

# In[7]:


df = df.fillna(method = 'ffill')


# In[8]:


# Memproses dataset untuk mengelompokkan per kalimat
class SentenceGetter:
    def __init__(self, data):
        self.data = data
        self.grouped = data.groupby("SentenceID").apply(
            lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                    s["POS"].values.tolist(),
                                                    s["Tag"].values.tolist())]
        )
        self.sentences = [sentence for sentence in self.grouped]


# In[9]:


# Inisialisasi SentenceGetter
getter = SentenceGetter(df)
sentences = getter.sentences

# Menampilkan salah satu kalimat untuk verifikasi
print("Contoh kalimat pertama:", sentences[0])


# Getting all the sentences in the dataset.

# #### Feature Preparation
# These are the default features used by the NER in nltk. We can also modify it for our customization.

# In[10]:


# Fungsi ekstraksi fitur dari setiap kata
def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    if i > 0:
        word1 = sent[i - 1][0]
        postag1 = sent[i - 1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True  # Beginning of sentence

    if i < len(sent) - 1:
        word1 = sent[i + 1][0]
        postag1 = sent[i + 1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True  # End of sentence

    return features

# Ekstraksi fitur untuk satu kalimat
def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

# Ekstraksi label dari satu kalimat
def sent2labels(sent):
    return [label for token, postag, label in sent]

# Ekstraksi token dari satu kalimat
def sent2tokens(sent):
    return [token for token, postag, label in sent]


# In[11]:


X = [sent2features(s) for s in sentences]
y = [sent2labels(s) for s in sentences]


# In[12]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# In[13]:


crf = CRF(algorithm = 'lbfgs',
         c1 = 0.1,
         c2 = 0.1,
         max_iterations = 100,
         all_possible_transitions = False)
crf.fit(X_train, y_train)


# In[14]:


#Predicting on the test set.
y_pred = crf.predict(X_test)


# #### Evaluating the model performance.
# We will use precision, recall and f1-score metrics to evaluate the performance of the model since the accuracy is not a good metric for this dataset because we have an unequal number of data points in each class.

# In[15]:


f1_score = flat_f1_score(y_test, y_pred, average = 'weighted')
print(f1_score)


# In[16]:


report = flat_classification_report(y_test, y_pred)
print(report)


# This looks quite nice.

# In[24]:


def test_model(sentence, crf_model):
    
    # Pecah kalimat menjadi kata-kata
    words = sentence.split()
    
    # Buat fitur untuk setiap kata (tanpa POS dalam contoh ini)
    sent_features = [
        {'bias': 1.0, 
         'word.lower()': word.lower(),
         'word[-3:]': word[-3:],
         'word[-2:]': word[-2:],
         'word.isupper()': word.isupper(),
         'word.istitle()': word.istitle(),
         'word.isdigit()': word.isdigit()} 
        for word in words
    ]
    
    # Tambahkan fitur konteks (kata sebelumnya dan sesudahnya)
    for i, word_features in enumerate(sent_features):
        if i > 0:  # Fitur kata sebelumnya
            word_features.update({
                '-1:word.lower()': words[i-1].lower(),
                '-1:word.istitle()': words[i-1].istitle(),
                '-1:word.isupper()': words[i-1].isupper(),
            })
        else:
            word_features['BOS'] = True  # Awal kalimat
        
        if i < len(words) - 1:  # Fitur kata sesudahnya
            word_features.update({
                '+1:word.lower()': words[i+1].lower(),
                '+1:word.istitle()': words[i+1].istitle(),
                '+1:word.isupper()': words[i+1].isupper(),
            })
        else:
            word_features['EOS'] = True  # Akhir kalimat

    # Prediksi tag menggunakan model
    predicted_tags = crf_model.predict([sent_features])[0]
    
    # Gabungkan kata dan tag untuk output
    result = list(zip(words, predicted_tags))
    return result


# Contoh penggunaan
kalimat_input = "Nuju sanja, I Rajapala ngaukin pianakné laut negak masila"
hasil = test_model(kalimat_input, crf)

# Tampilkan hasil
print("Hasil Prediksi:")
for kata, tag in hasil:
    print(f"{kata}\t{tag}")

