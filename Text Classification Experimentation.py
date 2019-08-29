# -*- coding: utf-8 -*-
"""
Created on Thu May  9 10:17:38 2019

@author: floodm
"""


import pandas as pd
import numpy as np
import re
import time
from scipy import interp
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import mlab as mlab
import pickle
from sklearn.externals import joblib
import nltk
import contractions
#from textblob import TextBlob # used for spelling corrections
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.feature_extraction.text import CountVectorizer  
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import KFold 
from sklearn.model_selection import StratifiedKFold #use k=10
from sklearn.model_selection import cross_val_score #automatically uses stratifiedKflods
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV #requires much less processing power
from nltk.tokenize import word_tokenize
from nltk.tokenize import WordPunctTokenizer
from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve, average_precision_score
from sklearn.datasets import make_classification
from imblearn import under_sampling, over_sampling
from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.under_sampling import TomekLinks
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.under_sampling import NearMiss, RandomUnderSampler, EditedNearestNeighbours
from sklearn.utils import resample
from sklearn.utils import class_weight
import random
from math import sqrt
from scipy.sparse import coo_matrix, hstack
from scipy.sparse import hstack
from scipy.sparse import csr_matrix




###########################################################################################
############# experimental framework for NLP thesis results ########################
###########################################################################################


### There are 7 Steps in this methodology
### Step 1. Importing & cleaning the dataframe

# importing the dataframe
df = pd.read_excel(r"H:\Quality and Safety\Interns\Max\Text Classification\Data.xlsx")
# Cleaning the dataframe to remove null, non-important, or falls rows and columns ###
df = df[df["EQ Review"].notnull()]
df = df[df["INCIDENT_DESCRIPTION"].notnull()]
df = df[df['GENERAL_INC_TYPE'] != 'Fall']
df = df[df['INCIDENT_SEVERITY'].notnull()]
df = df[["INCIDENT_DESCRIPTION", "INCIDENT_SEVERITY", "EQ Review"]]
df.rename(columns = {'INCIDENT_DESCRIPTION':'incident',
                     'EQ Review':'eq review',
                     'INCIDENT_SEVERITY':'severity'}, inplace=True)
df["eq review"].value_counts()
# fix the ordering of row numbers
df = df.sample(frac=1, random_state=777)
df.reset_index(drop=True, inplace=True)
dummies = pd.get_dummies(df['severity']).iloc[:, 1:]


# Creating the list of stopwords
stopset = stopwords.words('english') # I can add extra words like html jumk into the list of words
stopset.extend(["pm", "ml", "l", "mg", "g", "mcg", "oz", "lbs", "lb", "qt", "mm", 
                "wa", "ha", "aa", "pt", "patient", "rn", "nurse", "np"])


# Looking into the distribution of severity levels their respective eq reviews ###
df2 = pd.read_excel(r"H:\Quality and Safety\Interns\Max\NLP Thesis Stuff\Python Code\Data.xlsx")
df2.isnull().sum()
df2 = df2[df2["EQ Review"].notnull()]
df2.isnull().sum()
df2 = df2[df2["INCIDENT_DESCRIPTION"].notnull()]
df2.isnull().sum()
df2.rename(columns = {'INCIDENT_DESCRIPTION':'incident', 'EQ Review':'eq review'}, inplace=True)
df2['INCIDENT_SEVERITY'].value_counts()
# creates a dataframe of only eq reviews
df3 = df2[df2['eq review']==1]
df3['INCIDENT_SEVERITY'].value_counts()


# Most common features before pre-processing
tokens = df.incident.str.cat(sep=' ')
tokens = WordPunctTokenizer().tokenize(tokens) # shows there are 2,503,920 words in this corpus
# shows how many unique words there are
unique_words = nltk.FreqDist(tokens) # shows 44,200 unique words
top_words = unique_words.most_common(0)
# ploting the most common words
plt.figure(figsize=(16,5)) #increase this to (16,5) for larger display
unique_words.plot(50)
# number or characters and words per document
average_character_count = df['incident'].str.len().mean() # mean=394 
median_character_count = df['incident'].str.len().median() # median=298
max_character_count = df['incident'].str.len().max() # max=19,599
average_word_count = df['incident'].str.split(' ').str.len().mean() # mean=65
median_word_count = df['incident'].str.split(' ').str.len().median() # median=49
max_word_count = df['incident'].str.split(' ').str.len().max() # max=2,833



# Pre-processing the text with stemming
X = []
doc_length = []
for sen in range(0, len(df["incident"])):  
    # Convert everything to Lowercase
    document = (df['incident'][sen]).lower()
    # Remove all the special characters
    document = re.sub(r'\W', ' ', document)
    # Remove all numbers, dates, and times
    document = re.sub(r'[0-9]+', ' ', document)
    # Removing underscores "__"
    document = document.replace('_', ' ')
    # Reducing repeating letters
    document = re.sub(r'(\w)\1{1,7}', r'\1\1', document)
    # Remove all single character words
    document = re.sub(r'\s[a-zA-z]{1}\s', ' ', document)
    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)
    # Tokenizing
    document = WordPunctTokenizer().tokenize(document)
    # Remove Stopwords
    document = [word for word in document if word not in stopset]
    # Stemming
    document = [SnowballStemmer('english').stem(t) for t in document]
    doc_length.append(len(document))
    document = ' '.join(document)
    # Remove all single characters that could have been created due to tokenization
    document = re.sub(r'\s[a-zA-z]{1}\s', ' ', document)
    # Editing some words of intrest
    document = document.replace('bp', 'bloodpressure')
    document = document.replace('blood pressure', 'bloodpressure')
    document = document.replace('ordered', 'order')
    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)
    X.append(document)
df['incident'] = X



# Most common features after stemming pre-processing
tokens = df.incident.str.cat(sep=' ')
tokens = WordPunctTokenizer().tokenize(tokens) #shows there are 1,297,146 words in this corpus
# shows how many unique words there are
unique_words = nltk.FreqDist(tokens) # shows 21,116 unique words
top_words = unique_words.most_common(50)
# ploting the most common words
plt.figure(figsize=(16,5)) #increase this to (16,5) for larger display
unique_words.plot(50)
# number or characters and words per document
average_character_count = df['incident'].str.len().mean() # mean=226
median_character_count = df['incident'].str.len().median() # median=174
max_character_count = df['incident'].str.len().max() # max=16,936
average_word_count = df['incident'].str.split(' ').str.len().mean() # mean=37
median_word_count = df['incident'].str.split(' ').str.len().median() # median=28
max_word_count = df['incident'].str.split(' ').str.len().max() # max=2,501


# Boxplot of word counts
doc_length = [t for t in doc_length if t < 500]
fig, ax = plt.subplots(figsize=(5, 5))
plt.boxplot(doc_length)
plt.ylabel("Word Count")
plt.title("Box Plot")
plt.xticks([1], ['Incident Reports'])
plt.show()



# Pre-processing the text with lemmatizion 
X = []
doc_length = []
for sen in range(0, len(df["incident"])):  
    # Convert everything to Lowercase
    document = (df['incident'][sen]).lower()
    # Remove all the special characters
    document = re.sub(r'\W', ' ', document)
    # Remove all numbers, dates, and times
    document = re.sub(r'[0-9]+', ' ', document)
    # Removing underscores "__"
    document = document.replace('_', ' ')
    # Reducing repeating letters
    document = re.sub(r'(\w)\1{1,7}', r'\1\1', document)
    # Remove all single character words
    document = re.sub(r'\s[a-zA-z]{1}\s', ' ', document)
    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)
    # Tokenizing
    document = WordPunctTokenizer().tokenize(document)
    # Remove Stopwords
    document = [word for word in document if word not in stopset]
    # Lemmatizing
    document = [WordNetLemmatizer().lemmatize(t) for t in document]
    doc_length.append(len(document))
    document = ' '.join(document)
    # Remove all single characters that could have been created due to tokenization
    document = re.sub(r'\s[a-zA-z]{1}\s', ' ', document)
    # Editing some words of intrest
    document = document.replace('bp', 'bloodpressure')
    document = document.replace('blood pressure', 'bloodpressure')
    document = document.replace('ordered', 'order')
    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)
    X.append(document)
df['incident'] = X



# Most common features after lemmatizion pre-processing
tokens = df.incident.str.cat(sep=' ')
tokens = WordPunctTokenizer().tokenize(tokens) #shows there are 1,110,959 words in this corpus
# shows how many unique words there are
unique_words = nltk.FreqDist(tokens) # shows 25,817 unique words
top_words = unique_words.most_common(50)
# ploting the most common words
plt.figure(figsize=(16,5)) #increase this to (16,5) for larger display
unique_words.plot(50)
# number or characters and words per document
average_character_count = df['incident'].str.len().mean() # mean=252
median_character_count = df['incident'].str.len().median() # median=196
max_character_count = df['incident'].str.len().max() # max=17,054
average_word_count = df['incident'].str.split(' ').str.len().mean() # mean=35
median_word_count = df['incident'].str.split(' ').str.len().median() # median=27
max_word_count = df['incident'].str.split(' ').str.len().max() # max=2,464






### Step 2. Setting up Stratified Cross Validcation x10 folds, and the empty data sets to hold the values for confusion matrix
kf = StratifiedKFold(n_splits=10)
cross_val_accuracy = []
cross_val_precision = []
cross_val_recall = []
cross_val_f1 = []
cross_val_auc = []
pred_sum = []
actual_sum = []
start = time.time()
for train_index, test_index in kf.split(df['incident'], df['eq review']):
    X_train, X_test = df['incident'][train_index], df['incident'][test_index]
    y_train, y_test = df['eq review'][train_index], df['eq review'][test_index]
    ### Step 3. Vectorizing the text data
    vect = TfidfVectorizer(stop_words = stopset, min_df=2, ngram_range=(1,1))
    #vect = CountVectorizer(stop_words = stopset, min_df=2, ngram_range=(1,1))
    vectorizer = vect.fit(X_train)
    X_train_vectorized = vectorizer.transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)
    
    # Showing the Smallest & Largest TF-IDF scores
    #feature_names = np.array(vect.get_feature_names())
    #sorted_tfidf_index = X_train_vectorized.max(0).toarray()[0].argsort()
    #print('Smallest tfidf values: \n{}\n'.format(feature_names[sorted_tfidf_index[:20]]))
    #print('largest tfidf values: \n{}\n'.format(feature_names[sorted_tfidf_index[:-21:-1]]))

    ### Step 4. Feature Selection; Selecting a subset of the best 800 features(words) to use in the model via Chi-Squared method
    fs = SelectKBest(chi2, k=1200)
    #fs = SelectKBest(chi2, k=1000)
    #fs = SelectKBest(chi2, k=800)
    #fs = SelectKBest(chi2, k=600)
    #fs = SelectKBest(chi2, k=400)
    #fs = SelectKBest(chi2, k=200)
    fs.fit(X_train_vectorized, y_train)
    X_train_fs = fs.transform(X_train_vectorized)
    X_test_fs = fs.transform(X_test_vectorized)
    
    # Number of words with significant p-value scores
    #p_scores = chi2(X_train_vectorized, y_train)[1]
    #pvalues = p_scores < 0.05
    #important_column_indexs = np.where(pvalues == True)[0]
    #print(len(important_column_indexs))
    
    # Displaying the best 50 words
    best_feature_indices = np.argsort(fs.pvalues_)[:800]
    best_feature_names = np.array(vect.get_feature_names())[best_feature_indices]
    print('Best 50 features: \n{}\n'.format(best_feature_names[:50]))
    
    # Displaying the worst 50 words
    p_scores = chi2(X_train_vectorized, y_train)[1]
    worst_feature_indices = np.argsort(p_scores)[-51:]
    worst_feature_names = np.array(vect.get_feature_names())[worst_feature_indices]
    print('worst 50 features: \n{}\n'.format(worst_feature_names))
    
    ### Step 5. Resampling; under-sampling & over-sampling
    rus = RandomUnderSampler()
    #rus = NearMiss(version=1)
    #rus = NearMiss(version=2)
    #rus = NearMiss(version=3)
    #rus = TomekLinks()
    #rus = SMOTE()
    #rus = RandomUnderSampler(ratio=.2)
    #X_train_vectorized, y_train = rus.fit_resample(X_train_vectorized, y_train)
    #X_test_vectorized, y_test = rus.fit_resample(X_test_vectorized, y_test)
    #rus = SMOTE()
    #rus = BorderlineSMOTE(kind='borderline-1')
    #rus = BorderlineSMOTE(kind='borderline-2')
    #rus = SMOTETomek()
    #rus = SMOTEENN()
    X_train_rus, y_train_rus = rus.fit_resample(X_train_fs, y_train)
    X_test_rus, y_test_rus = rus.fit_resample(X_test_fs, y_test)
    
    ### Step 6. Building the classification model
    MNB_model = MultinomialNB()
    #MNB_model = GradientBoostingClassifier()
    #MNB_model = RandomForestClassifier()
    #MNB_model = DecisionTreeClassifier()
    #MNB_model = SVC(kernel='linear')
    #MNB_model = LogisticRegression()
    MNB_model.fit(X_train_rus, y_train_rus)
    probs = MNB_model.predict_proba(X_test_rus)[:,1]
    y_pred = MNB_model.predict(X_test_rus)
    pred_sum.extend(y_pred)
    actual_sum.extend(y_test_rus)
    cross_val_accuracy.append(accuracy_score(y_test_rus, y_pred))
    cross_val_precision.append(precision_score(y_test_rus, y_pred))
    cross_val_recall.append(recall_score(y_test_rus, y_pred))
    cross_val_f1.append(f1_score(y_test_rus, y_pred))
    cross_val_auc.append(roc_auc_score(y_test_rus, probs))
    print(confusion_matrix(y_test_rus, y_pred))
    
    
    # Shows the most predictive words in each class
    #class0_indices = np.argsort(MNB_model.feature_log_prob_[0, :])
    #class1_indices = np.argsort(MNB_model.feature_log_prob_[1, :])
    #Class0_best_model_features = np.array(best_feature_names)[class0_indices]
    #Class1_best_model_features = np.array(best_feature_names)[class1_indices]
    #print('Class 0 Best 50 features: \n{}\n'.format(Class0_best_model_features[:10]))
    #print('Class 1 Best 50 features: \n{}\n'.format(Class1_best_model_features[:10]))
    
    #print(' ')
    stop = time.time()
print('Classifier took', round(stop-start,1) ,'seconds to run') 
# averaging together the results from the 10 classification models
print ('Cross validated accuracy score: {}'.format(np.mean(cross_val_accuracy))) # ~66%
print ('Cross validated precision score: {}'.format(np.mean(cross_val_precision))) # ~63%
print ('Cross validated recall score: {}'.format(np.mean(cross_val_recall))) # ~84%
print ('Cross validated f1 score: {}'.format(np.mean(cross_val_f1))) # ~72%
print ('Cross validated auc score: {}'.format(np.mean(cross_val_auc))) # ~78%


# Total Summed Confusion Matrix
cm = confusion_matrix(actual_sum, pred_sum)
ax= plt.subplot()
cmap=plt.get_cmap('Blues')
sns.heatmap(cm, annot=True, ax = ax, cmap=cmap, fmt='g') #annot=True to annotate cells
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(['Not Reviewed', 'Reviewed']); ax.yaxis.set_ticklabels(['Not Reviewed', 'Reviewed'])

# 95% Confidence Interval for Recall
ci_recall = round(np.mean(cross_val_recall),3)
interval = round((1.96 * sqrt((ci_recall*(1-ci_recall))/542)) * 100, 1)
print('Recall Confidence Interval=', 
      str(ci_recall*100)+'% '+'+/-'+' %.1f' %interval+'% or', 
      '('+'%.1f' %(ci_recall*100-interval)+'%'+', ''%.1f' %(ci_recall*100+interval)+'%'+')')

# 95% Confidence Interval for Overall Error
error = round(1-np.mean(cross_val_accuracy), 3)
interval = round((1.96 * sqrt((error*(1-error))/812))*100, 1)
print('Confidence Interval=', 
      str(error*100)+'% '+'+/-'+' %.1f' %interval+'% or', 
      '('+'%.1f' %(error*100-interval)+'%'+', ''%.1f' %(error*100+interval)+'%'+')')


print(" ")



### Step 7. Checking for Overfitting; the below code runs the same code as above but makes predictions on the same data that is used to build the classification model
kf = StratifiedKFold(n_splits=10)
cross_val_accuracy = []
cross_val_precision = []
cross_val_recall = []
cross_val_f1 = []
cross_val_auc = []
start = time.time()
for train_index, test_index in kf.split(df['incident'], df['eq review']):
    X_train, X_test = df['incident'][train_index], df['incident'][test_index]
    y_train, y_test = df['eq review'][train_index], df['eq review'][test_index]
    ### Vectorizing the text data
    vect = TfidfVectorizer(stop_words = stopset, min_df=2, ngram_range=(1,1))
    #vect = CountVectorizer(stop_words = stopset, min_df=2, ngram_range=(1,1))
    vectorizer = vect.fit(X_train)
    X_train_vectorized = vectorizer.transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)
    
    # Showing the Smallest & Largest TF-IDF scores
    #feature_names = np.array(vect.get_feature_names())
    #sorted_tfidf_index = X_train_vectorized.max(0).toarray()[0].argsort()
    #print('Smallest tfidf values: \n{}\n'.format(feature_names[sorted_tfidf_index[:20]]))
    #print('largest tfidf values: \n{}\n'.format(feature_names[sorted_tfidf_index[:-21:-1]]))

    ### Feature Selection; Selecting a subset of the best 800 features(words) to use in the model via Chi-Squared method
    fs = SelectKBest(chi2, k=1200)
    #fs = SelectKBest(chi2, k=1000)
    #fs = SelectKBest(chi2, k=800)
    #fs = SelectKBest(chi2, k=600)
    #fs = SelectKBest(chi2, k=400)
    #fs = SelectKBest(chi2, k=200)
    fs.fit(X_train_vectorized, y_train)
    X_train_fs = fs.transform(X_train_vectorized)
    X_test_fs = fs.transform(X_test_vectorized)
    
    # Displaying the best 50 words
    #best_feature_indices = np.argsort(fs.pvalues_)[:800]
    #best_feature_names = np.array(vect.get_feature_names())[best_feature_indices]
    #print('Best 50 features: \n{}\n'.format(best_feature_names[:50]))
    
    ### Resampling; under-sampling & over-sampling
    rus = RandomUnderSampler()
    #rus = NearMiss(version=1)
    #rus = NearMiss(version=2)
    #rus = NearMiss(version=3)
    #rus = TomekLinks()
    #rus = SMOTE()
    #rus = RandomUnderSampler(ratio=.2)
    #X_train_vectorized, y_train = rus.fit_resample(X_train_vectorized, y_train)
    #X_test_vectorized, y_test = rus.fit_resample(X_test_vectorized, y_test)
    #rus = SMOTE()
    #rus = BorderlineSMOTE(kind='borderline-1')
    #rus = BorderlineSMOTE(kind='borderline-2')
    #rus = SMOTETomek()
    #rus = SMOTEENN()
    X_train_rus, y_train_rus = rus.fit_resample(X_train_fs, y_train)
    X_test_rus, y_test_rus = rus.fit_resample(X_test_fs, y_test)
    
    ### Building the classification model
    MNB_model = MultinomialNB()
    MNB_model.fit(X_train_rus, y_train_rus)
    probs = MNB_model.predict_proba(X_train_rus)[:,1]
    y_pred = MNB_model.predict(X_train_rus)
    cross_val_accuracy.append(accuracy_score(y_train_rus, y_pred))
    cross_val_precision.append(precision_score(y_train_rus, y_pred))
    cross_val_recall.append(recall_score(y_train_rus, y_pred))
    cross_val_f1.append(f1_score(y_train_rus, y_pred))
    cross_val_auc.append(roc_auc_score(y_train_rus, probs))
    print(confusion_matrix(y_train_rus, y_pred))
    stop = time.time()
print('Classifier took', round(stop-start,1) ,'seconds to run')
# averaging together the results from the 10 classification models
print ('Cross validated accuracy score: {}'.format(np.mean(cross_val_accuracy))) # ~ 72%
print ('Cross validated precision score: {}'.format(np.mean(cross_val_precision))) # ~68%
print ('Cross validated recall score: {}'.format(np.mean(cross_val_recall))) # ~86%
print ('Cross validated f1 score: {}'.format(np.mean(cross_val_f1))) # ~76%
print ('Cross validated auc score: {}'.format(np.mean(cross_val_auc))) # ~83%
 

# Confidence Interval
ci_recall = round(np.mean(cross_val_recall),3)
interval = round((1.96 * sqrt((ci_recall*(1-ci_recall))/96)) * 100, 1)
print('Recall Confidence Interval=', 
      str(ci_recall*100)+'% '+'+/-'+' %.1f' %interval+'% or', 
      '('+'%.1f' %(ci_recall*100-interval)+'%'+', ''%.1f' %(ci_recall*100+interval)+'%'+')')












##########################################################################################################
######## Does prediction scores increase when resampling is done before feature selection? ###############

# make sure to have run step 1 before running the below code

### Step 2. Setting up Stratified Cross Validcation x10 folds, and the empty data sets to hold the values for confusion matrix
kf = StratifiedKFold(n_splits=10)
cross_val_accuracy = []
cross_val_precision = []
cross_val_recall = []
cross_val_f1 = []
cross_val_auc = []
test = []
start = time.time()
for train_index, test_index in kf.split(df['incident'], df['eq review']):
    X_train, X_test = df['incident'][train_index], df['incident'][test_index]
    y_train, y_test = df['eq review'][train_index], df['eq review'][test_index]
    ### Step 3. Vectorizing the text data
    vect = TfidfVectorizer(stop_words = stopset, min_df=2, ngram_range=(1,1))
    #vect = CountVectorizer(stop_words = stopset, min_df=2, ngram_range=(1,1))
    vectorizer = vect.fit(X_train)
    X_train_vectorized = vectorizer.transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    ### Step 4. Resampling; under-sampling & over-sampling
    #rus = RandomUnderSampler()
    #rus = NearMiss(version=1)
    #rus = NearMiss(version=2)
    #rus = NearMiss(version=3)
    #rus = TomekLinks()
   # rus = SMOTE()
    rus = RandomUnderSampler(ratio=.2)
    X_train_vectorized, y_train = rus.fit_resample(X_train_vectorized, y_train)
    X_test_vectorized, y_test = rus.fit_resample(X_test_vectorized, y_test)
    #rus = SMOTE()
    #rus = BorderlineSMOTE(kind='borderline-1')
    #rus = BorderlineSMOTE(kind='borderline-2')
    #rus = SMOTETomek()
    rus = SMOTEENN()
    X_train_rus, y_train_rus = rus.fit_resample(X_train_vectorized, y_train)
    X_test_rus, y_test_rus = rus.fit_resample(X_test_vectorized, y_test)
    
    ### Step 5. Feature Selection; Selecting a subset of the best features(words) to use in the model via Chi-Squared method
    fs = SelectKBest(chi2, k=1200)
    #fs = SelectKBest(chi2, k=1000)
    #fs = SelectKBest(chi2, k=800)
    #fs = SelectKBest(chi2, k=600)
    #fs = SelectKBest(chi2, k=400)
    #fs = SelectKBest(chi2, k=200)
    fs.fit(X_train_vectorized, y_train)
    X_train_fs = fs.transform(X_train_rus)
    X_test_fs = fs.transform(X_test_rus)
    
    ### Step 6. Building the classification model
    MNB_model = MultinomialNB()
    MNB_model.fit(X_train_fs, y_train_rus)
    probs = MNB_model.predict_proba(X_test_fs)[:,1]
    y_pred = MNB_model.predict(X_test_fs)
    cross_val_accuracy.append(accuracy_score(y_test_rus, y_pred))
    cross_val_precision.append(precision_score(y_test_rus, y_pred))
    cross_val_recall.append(recall_score(y_test_rus, y_pred))
    cross_val_f1.append(f1_score(y_test_rus, y_pred))
    cross_val_auc.append(roc_auc_score(y_test_rus, probs))
    print(confusion_matrix(y_test_rus, y_pred))
# averaging together the results from the 10 classification models
print ('Cross validated accuracy score: {}'.format(np.mean(cross_val_accuracy)))
print ('Cross validated precision score: {}'.format(np.mean(cross_val_precision)))
print ('Cross validated recall score: {}'.format(np.mean(cross_val_recall)))
print ('Cross validated f1 score: {}'.format(np.mean(cross_val_f1)))
print ('Cross validated auc score: {}'.format(np.mean(cross_val_auc)))



print(" ")


### Step 7. Checking for Overfitting; the below code runs the same code as above but makes predictions on the same data that is used to build the classification model
kf = StratifiedKFold(n_splits=10)
cross_val_accuracy = []
cross_val_precision = []
cross_val_recall = []
cross_val_f1 = []
cross_val_auc = []
start = time.time()
for train_index, test_index in kf.split(df['incident'], df['eq review']):
    X_train, X_test = df['incident'][train_index], df['incident'][test_index]
    y_train, y_test = df['eq review'][train_index], df['eq review'][test_index]
    ### Vectorizing the text data
    vect = TfidfVectorizer(stop_words = stopset, min_df=2, ngram_range=(1,1))
    #vect = CountVectorizer(stop_words = stopset, min_df=2, ngram_range=(1,1))
    vectorizer = vect.fit(X_train)
    X_train_vectorized = vectorizer.transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)
    
    ### Resampling; under-sampling & over-sampling
    #rus = RandomUnderSampler()
    #rus = NearMiss(version=1)
    #rus = NearMiss(version=2)
    #rus = NearMiss(version=3)
    #rus = TomekLinks()
    #rus = SMOTE()
    rus = RandomUnderSampler(ratio=.2)
    X_train_vectorized, y_train = rus.fit_resample(X_train_vectorized, y_train)
    X_test_vectorized, y_test = rus.fit_resample(X_test_vectorized, y_test)
    #rus = SMOTE()
    #rus = BorderlineSMOTE(kind='borderline-1')
    #rus = BorderlineSMOTE(kind='borderline-2')
    #rus = SMOTETomek()
    rus = SMOTEENN()
    X_train_rus, y_train_rus = rus.fit_resample(X_train_vectorized, y_train)
    X_test_rus, y_test_rus = rus.fit_resample(X_test_vectorized, y_test)
    
    ### Feature Selection; Selecting a subset of the best features(words) to use in the model via Chi-Squared method
    fs = SelectKBest(chi2, k=1200)
    #fs = SelectKBest(chi2, k=1000)
    #fs = SelectKBest(chi2, k=800)
    #fs = SelectKBest(chi2, k=600)
    #fs = SelectKBest(chi2, k=400)
    #fs = SelectKBest(chi2, k=200)
    fs.fit(X_train_vectorized, y_train)
    X_train_rus = fs.transform(X_train_rus)
    X_test_rus = fs.transform(X_test_rus)
    
    
    ### Building the classification model
    MNB_model = MultinomialNB()
    MNB_model.fit(X_train_rus, y_train_rus)
    probs = MNB_model.predict_proba(X_train_rus)[:,1]
    y_pred = MNB_model.predict(X_train_rus)
    cross_val_accuracy.append(accuracy_score(y_train_rus, y_pred))
    cross_val_precision.append(precision_score(y_train_rus, y_pred))
    cross_val_recall.append(recall_score(y_train_rus, y_pred))
    cross_val_f1.append(f1_score(y_train_rus, y_pred))
    cross_val_auc.append(roc_auc_score(y_train_rus, probs))
    print(confusion_matrix(y_train_rus, y_pred))
    stop = time.time()
print('Classifier took', round(stop-start,1) ,'seconds to run')
# averaging together the results from the 10 classification models
print ('Cross validated accuracy score: {}'.format(np.mean(cross_val_accuracy)))
print ('Cross validated precision score: {}'.format(np.mean(cross_val_precision)))
print ('Cross validated recall score: {}'.format(np.mean(cross_val_recall)))
print ('Cross validated f1 score: {}'.format(np.mean(cross_val_f1)))
print ('Cross validated auc score: {}'.format(np.mean(cross_val_auc)))























#################################################################################################
# Does adding features like "document word length" or "severity level" boost prediction scores? #

# make sure to have run step 1 before running the below code

### Step 2. Setting up Stratified Cross Validcation x10 folds, and the empty data sets to hold the values for confusion matrix
kf = StratifiedKFold(n_splits=10)
cross_val_accuracy = []
cross_val_precision = []
cross_val_recall = []
cross_val_f1 = []
cross_val_auc = []
pred_sum = []
actual_sum = []
start = time.time()
for train_index, test_index in kf.split(df['incident'], df['eq review']):
    X_train, X_test = df['incident'][train_index], df['incident'][test_index]
    y_train, y_test = df['eq review'][train_index], df['eq review'][test_index]
    ### Step 3. Vectorizing the text data
    vect = TfidfVectorizer(stop_words = stopset, min_df=2, ngram_range=(1,1))
    #vect = CountVectorizer(stop_words = stopset, min_df=2, ngram_range=(1,1))
    vectorizer = vect.fit(X_train)
    X_train_vectorized = vectorizer.transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)
    
    # Showing the Smallest & Largest TF-IDF scores
    #feature_names = np.array(vect.get_feature_names())
    #sorted_tfidf_index = X_train_vectorized.max(0).toarray()[0].argsort()
    #print('Smallest tfidf values: \n{}\n'.format(feature_names[sorted_tfidf_index[:20]]))
    #print('largest tfidf values: \n{}\n'.format(feature_names[sorted_tfidf_index[:-21:-1]]))
    
    # Attaching severity & doc_length features to train data
    doc_length_train = [doc_length[i] for i in train_index]
    X_train_vectorized = hstack((X_train_vectorized,np.array(doc_length_train)[:,None])).tocsr()
    X_train_vectorized = hstack((X_train_vectorized,np.array(dummies['Level 1-No Harm'][train_index])[:,None])).tocsr()
    X_train_vectorized = hstack((X_train_vectorized,np.array(dummies['Level 2-Temporary Harm'][train_index])[:,None])).tocsr()
    X_train_vectorized = hstack((X_train_vectorized,np.array(dummies['Level 3-Significant or Permanent Harm'][train_index])[:,None])).tocsr()
    X_train_vectorized = hstack((X_train_vectorized,np.array(dummies['Level 4-Death'][train_index])[:,None])).tocsr()
    
    # Attaching severity & doc_length features to test data
    doc_length_test = [doc_length[i] for i in test_index]
    X_test_vectorized = hstack((X_test_vectorized,np.array(doc_length_test)[:,None])).tocsr()
    X_test_vectorized = hstack((X_test_vectorized,np.array(dummies['Level 1-No Harm'][test_index])[:,None])).tocsr()
    X_test_vectorized = hstack((X_test_vectorized,np.array(dummies['Level 2-Temporary Harm'][test_index])[:,None])).tocsr()
    X_test_vectorized = hstack((X_test_vectorized,np.array(dummies['Level 3-Significant or Permanent Harm'][test_index])[:,None])).tocsr()
    X_test_vectorized = hstack((X_test_vectorized,np.array(dummies['Level 4-Death'][test_index])[:,None])).tocsr()

    
    ### Step 4. Feature Selection; Selecting a subset of the best features(words) to use in the model via Chi-Squared method
    fs = SelectKBest(chi2, k=1200)
    #fs = SelectKBest(chi2, k=1000)
    #fs = SelectKBest(chi2, k=800)
    #fs = SelectKBest(chi2, k=600)
    #fs = SelectKBest(chi2, k=400)
    #fs = SelectKBest(chi2, k=200)
    fs.fit(X_train_vectorized, y_train)
    X_train_fs = fs.transform(X_train_vectorized)
    X_test_fs = fs.transform(X_test_vectorized)
    
    # Number of words with significant p-value scores
    #p_scores = chi2(X_train_vectorized, y_train)[1]
    #pvalues = p_scores < 0.05
    #important_column_indexs = np.where(pvalues == True)[0]
    #print(len(important_column_indexs))
    
    # Displaying the best 50 words
    best_feature_indices = np.argsort(fs.pvalues_)[:800]
    best_feature_names = np.array(vect.get_feature_names())[best_feature_indices]
    print('Best 50 features: \n{}\n'.format(best_feature_names[:50]))
    
    # Displaying the worst 50 words
    p_scores = chi2(X_train_vectorized, y_train)[1]
    worst_feature_indices = np.argsort(p_scores)[-51:]
    worst_feature_names = np.array(vect.get_feature_names())[worst_feature_indices]
    print('worst 50 features: \n{}\n'.format(worst_feature_names))
    
    ### Step 5. Resampling; under-sampling & over-sampling
    rus = RandomUnderSampler()
    #rus = NearMiss(version=1)
    #rus = NearMiss(version=2)
    #rus = NearMiss(version=3)
    #rus = TomekLinks()
    #rus = SMOTE()
    #rus = RandomUnderSampler(ratio=.2)
    #X_train_vectorized, y_train = rus.fit_resample(X_train_vectorized, y_train)
    #X_test_vectorized, y_test = rus.fit_resample(X_test_vectorized, y_test)
    #rus = SMOTE()
    #rus = BorderlineSMOTE(kind='borderline-1')
    #rus = BorderlineSMOTE(kind='borderline-2')
    #rus = SMOTETomek()
    #rus = SMOTEENN()
    X_train_rus, y_train_rus = rus.fit_resample(X_train_fs, y_train)
    X_test_rus, y_test_rus = rus.fit_resample(X_test_fs, y_test)
    
    ### Step 6. Building the classification model
    MNB_model = MultinomialNB()
    #MNB_model = GradientBoostingClassifier()
    #MNB_model = RandomForestClassifier()
    #MNB_model = DecisionTreeClassifier()
    #MNB_model = SVC(kernel='linear')
    #MNB_model = LogisticRegression()
    MNB_model.fit(X_train_rus, y_train_rus)
    probs = MNB_model.predict_proba(X_test_rus)[:,1]
    y_pred = MNB_model.predict(X_test_rus)
    pred_sum.extend(y_pred)
    actual_sum.extend(y_test_rus)
    cross_val_accuracy.append(accuracy_score(y_test_rus, y_pred))
    cross_val_precision.append(precision_score(y_test_rus, y_pred))
    cross_val_recall.append(recall_score(y_test_rus, y_pred))
    cross_val_f1.append(f1_score(y_test_rus, y_pred))
    cross_val_auc.append(roc_auc_score(y_test_rus, probs))
    print(confusion_matrix(y_test_rus, y_pred))
    
    
    # Shows the most predictive words in each class
    #class0_indices = np.argsort(MNB_model.feature_log_prob_[0, :])
    #class1_indices = np.argsort(MNB_model.feature_log_prob_[1, :])
    #Class0_best_model_features = np.array(best_feature_names)[class0_indices]
    #Class1_best_model_features = np.array(best_feature_names)[class1_indices]
    #print('Class 0 Best 50 features: \n{}\n'.format(Class0_best_model_features[:10]))
    #print('Class 1 Best 50 features: \n{}\n'.format(Class1_best_model_features[:10]))
    
    #print(' ')
    stop = time.time()
print('Classifier took', round(stop-start,1) ,'seconds to run') 
# averaging together the results from the 10 classification models
print ('Cross validated accuracy score: {}'.format(np.mean(cross_val_accuracy))) # ~66%
print ('Cross validated precision score: {}'.format(np.mean(cross_val_precision))) # ~63%
print ('Cross validated recall score: {}'.format(np.mean(cross_val_recall))) # ~84%
print ('Cross validated f1 score: {}'.format(np.mean(cross_val_f1))) # ~72%
print ('Cross validated auc score: {}'.format(np.mean(cross_val_auc))) # ~78%


# Total Summed Confusion Matrix
cm = confusion_matrix(actual_sum, pred_sum)
ax= plt.subplot()
cmap=plt.get_cmap('Blues')
sns.heatmap(cm, annot=True, ax = ax, cmap=cmap, fmt='g') #annot=True to annotate cells
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(['Not Reviewed', 'Reviewed']); ax.yaxis.set_ticklabels(['Not Reviewed', 'Reviewed'])

# 95% Confidence Interval for Recall
ci_recall = round(np.mean(cross_val_recall),3)
interval = round((1.96 * sqrt((ci_recall*(1-ci_recall))/542)) * 100, 1)
print('Recall Confidence Interval=', 
      str(ci_recall*100)+'% '+'+/-'+' %.1f' %interval+'% or', 
      '('+'%.1f' %(ci_recall*100-interval)+'%'+', ''%.1f' %(ci_recall*100+interval)+'%'+')')

# 95% Confidence Interval for Overall Error
error = round(1-np.mean(cross_val_accuracy), 3)
interval = round((1.96 * sqrt((error*(1-error))/812))*100, 1)
print('Confidence Interval=', 
      str(error*100)+'% '+'+/-'+' %.1f' %interval+'% or', 
      '('+'%.1f' %(error*100-interval)+'%'+', ''%.1f' %(error*100+interval)+'%'+')')



print(" ")


### Step 7. Checking for Overfitting; the below code runs the same code as above but makes predictions on the same data that is used to build the classification model
kf = StratifiedKFold(n_splits=10)
cross_val_accuracy = []
cross_val_precision = []
cross_val_recall = []
cross_val_f1 = []
cross_val_auc = []
start = time.time()
for train_index, test_index in kf.split(df['incident'], df['eq review']):
    X_train, X_test = df['incident'][train_index], df['incident'][test_index]
    y_train, y_test = df['eq review'][train_index], df['eq review'][test_index]
    ### Vectorizing the text data
    vect = TfidfVectorizer(stop_words = stopset, min_df=2, ngram_range=(1,1))
    #vect = CountVectorizer(stop_words = stopset, min_df=2, ngram_range=(1,1))
    vectorizer = vect.fit(X_train)
    X_train_vectorized = vectorizer.transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)
    
    # Showing the Smallest & Largest TF-IDF scores
    #feature_names = np.array(vect.get_feature_names())
    #sorted_tfidf_index = X_train_vectorized.max(0).toarray()[0].argsort()
    #print('Smallest tfidf values: \n{}\n'.format(feature_names[sorted_tfidf_index[:20]]))
    #print('largest tfidf values: \n{}\n'.format(feature_names[sorted_tfidf_index[:-21:-1]]))
    
    # Attaching severity & doc_length features to train data
    doc_length_train = [doc_length[i] for i in train_index]
    #X_train_vectorized = hstack((X_train_vectorized,np.array(doc_length_train)[:,None])).tocsr()
    X_train_vectorized = hstack((X_train_vectorized,np.array(dummies['Level 1-No Harm'][train_index])[:,None])).tocsr()
    X_train_vectorized = hstack((X_train_vectorized,np.array(dummies['Level 2-Temporary Harm'][train_index])[:,None])).tocsr()
    X_train_vectorized = hstack((X_train_vectorized,np.array(dummies['Level 3-Significant or Permanent Harm'][train_index])[:,None])).tocsr()
    X_train_vectorized = hstack((X_train_vectorized,np.array(dummies['Level 4-Death'][train_index])[:,None])).tocsr()
    
    # Attaching severity & doc_length features to test data
    doc_length_test = [doc_length[i] for i in test_index]
    #X_test_vectorized = hstack((X_test_vectorized,np.array(doc_length_test)[:,None])).tocsr()
    X_test_vectorized = hstack((X_test_vectorized,np.array(dummies['Level 1-No Harm'][test_index])[:,None])).tocsr()
    X_test_vectorized = hstack((X_test_vectorized,np.array(dummies['Level 2-Temporary Harm'][test_index])[:,None])).tocsr()
    X_test_vectorized = hstack((X_test_vectorized,np.array(dummies['Level 3-Significant or Permanent Harm'][test_index])[:,None])).tocsr()
    X_test_vectorized = hstack((X_test_vectorized,np.array(dummies['Level 4-Death'][test_index])[:,None])).tocsr()

    ### Feature Selection; Selecting a subset of the best features(words) to use in the model via Chi-Squared method
    fs = SelectKBest(chi2, k=1200)
    #fs = SelectKBest(chi2, k=1000)
    #fs = SelectKBest(chi2, k=800)
    #fs = SelectKBest(chi2, k=600)
    #fs = SelectKBest(chi2, k=400)
    #fs = SelectKBest(chi2, k=200)
    fs.fit(X_train_vectorized, y_train)
    X_train_fs = fs.transform(X_train_vectorized)
    X_test_fs = fs.transform(X_test_vectorized)
    
    # Displaying the best 50 words
    #best_feature_indices = np.argsort(fs.pvalues_)[:800]
    #best_feature_names = np.array(vect.get_feature_names())[best_feature_indices]
    #print('Best 50 features: \n{}\n'.format(best_feature_names[:50]))
    
    ### Resampling; under-sampling & over-sampling
    rus = RandomUnderSampler()
    #rus = NearMiss(version=1)
    #rus = NearMiss(version=2)
    #rus = NearMiss(version=3)
    #rus = TomekLinks()
    #rus = SMOTE()
    #rus = RandomUnderSampler(ratio=.2)
    #X_train_vectorized, y_train = rus.fit_resample(X_train_vectorized, y_train)
    #X_test_vectorized, y_test = rus.fit_resample(X_test_vectorized, y_test)
    #rus = SMOTE()
    #rus = BorderlineSMOTE(kind='borderline-1')
    #rus = BorderlineSMOTE(kind='borderline-2')
    #rus = SMOTETomek()
    #rus = SMOTEENN()
    X_train_rus, y_train_rus = rus.fit_resample(X_train_fs, y_train)
    X_test_rus, y_test_rus = rus.fit_resample(X_test_fs, y_test)
    
    
    ### Building the classification model
    MNB_model = MultinomialNB()
    MNB_model.fit(X_train_rus, y_train_rus)
    probs = MNB_model.predict_proba(X_train_rus)[:,1]
    y_pred = MNB_model.predict(X_train_rus)
    cross_val_accuracy.append(accuracy_score(y_train_rus, y_pred))
    cross_val_precision.append(precision_score(y_train_rus, y_pred))
    cross_val_recall.append(recall_score(y_train_rus, y_pred))
    cross_val_f1.append(f1_score(y_train_rus, y_pred))
    cross_val_auc.append(roc_auc_score(y_train_rus, probs))
    print(confusion_matrix(y_train_rus, y_pred))
    stop = time.time()
print('Classifier took', round(stop-start,1) ,'seconds to run')
# averaging together the results from the 10 classification models
print ('Cross validated accuracy score: {}'.format(np.mean(cross_val_accuracy))) # ~ 72%
print ('Cross validated precision score: {}'.format(np.mean(cross_val_precision))) # ~68%
print ('Cross validated recall score: {}'.format(np.mean(cross_val_recall))) # ~86%
print ('Cross validated f1 score: {}'.format(np.mean(cross_val_f1))) # ~76%
print ('Cross validated auc score: {}'.format(np.mean(cross_val_auc))) # ~83%
 

# Confidence Interval
ci_recall = round(np.mean(cross_val_recall),3)
interval = round((1.96 * sqrt((ci_recall*(1-ci_recall))/96)) * 100, 1)
print('Recall Confidence Interval=', 
      str(ci_recall*100)+'% '+'+/-'+' %.1f' %interval+'% or', 
      '('+'%.1f' %(ci_recall*100-interval)+'%'+', ''%.1f' %(ci_recall*100+interval)+'%'+')')




















################################# Creating ROC Curves ###################################
# make sure to have run step 1 before running the below code

kf = StratifiedKFold(n_splits=10)
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
i = 0
for train_index, test_index in kf.split(df['incident'], df['eq review']):
    X_train, X_test = df['incident'][train_index], df['incident'][test_index]
    y_train, y_test = df['eq review'][train_index], df['eq review'][test_index]
    vect = TfidfVectorizer(stop_words = stopset, min_df=2, ngram_range=(1,1))
    #vect = CountVectorizer(stop_words = stopset, min_df=2, ngram_range=(1,1))
    vectorizer = vect.fit(X_train)
    X_train_vectorized = vectorizer.transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)
    
    # Selecting a subset of the best 800 features to use in the model
    fs = SelectKBest(chi2, k=800)
    fs.fit(X_train_vectorized, y_train)
    X_train_fs = fs.transform(X_train_vectorized)
    X_test_fs = fs.transform(X_test_vectorized)

    # Random undersampling the majority class
    rus = RandomUnderSampler()
    X_train_rus, y_train_rus = rus.fit_resample(X_train_fs, y_train)
    X_test_rus, y_test_rus = rus.fit_resample(X_test_fs, y_test)
    
    # Building the classification model
    MNB_model = MultinomialNB()
    MNB_model.fit(X_train_rus, y_train_rus)
    probs = MNB_model.predict_proba(X_test_rus)[:,1]
    y_pred = MNB_model.predict(X_test_rus)
    print(confusion_matrix(y_test_rus, y_pred))
    
    #probas_ = classifier.fit(X_train_rus, y_train_rus).predict_proba(X_test_rus)
    fpr, tpr, thresholds = roc_curve(y_test_rus, probs)
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
    i += 1
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='navy', alpha=.8)
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='darkorange',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc))
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2)
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Curve')
plt.legend(loc="lower right")
plt.show()




############################ Creating Recall Precision Cruve ###################################
# make sure to have run step 1 before running the below code

kf = StratifiedKFold(n_splits=10)
tprs = []
aucs = []
mean_precision = np.linspace(0, 1, 100)
i = 0
for train_index, test_index in kf.split(df['incident'], df['eq review']):
    X_train, X_test = df['incident'][train_index], df['incident'][test_index]
    y_train, y_test = df['eq review'][train_index], df['eq review'][test_index]
    vect = TfidfVectorizer(stop_words = stopset, min_df=2, ngram_range=(1,1))
    #vect = CountVectorizer(stop_words = stopset, min_df=2, ngram_range=(1,1))
    vectorizer = vect.fit(X_train)
    X_train_vectorized = vectorizer.transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)
    
    # Selecting a subset of the best 800 features to use in the model
    fs = SelectKBest(chi2, k=800)
    fs.fit(X_train_vectorized, y_train)
    X_train_fs = fs.transform(X_train_vectorized)
    X_test_fs = fs.transform(X_test_vectorized)

    # Random undersampling the majority class
    rus = RandomUnderSampler()
    X_train_rus, y_train_rus = rus.fit_resample(X_train_fs, y_train)
    X_test_rus, y_test_rus = rus.fit_resample(X_test_fs, y_test)
    
    # Building the classification model
    MNB_model = MultinomialNB()
    MNB_model.fit(X_train_rus, y_train_rus)
    probs = MNB_model.predict_proba(X_test_rus)[:,1]
    y_pred = MNB_model.predict(X_test_rus)
    print(confusion_matrix(y_test_rus, y_pred))
    
    #probas_ = classifier.fit(X_train_rus, y_train_rus).predict_proba(X_test_rus)
    precision, recall, thresholds = precision_recall_curve(y_test_rus, probs)
    tprs.append(interp(mean_precision, precision, recall))
    aucscore = auc(recall, precision)
    aucs.append(aucscore)
    plt.plot(precision, recall, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, aucscore))
    i += 1
plt.plot([0, 1], [0.5,0.5], linestyle='--', lw=2, color='navy', alpha=.8)
mean_recall = np.mean(tprs, axis=0)
mean_auc = auc(mean_precision, mean_recall)
std_auc = np.std(aucs)
plt.plot(mean_precision, mean_recall, color='darkorange',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_recall + std_tpr, 1)
tprs_lower = np.maximum(mean_recall - std_tpr, 0)
plt.fill_between(mean_precision, tprs_lower, tprs_upper, color='grey', alpha=.2)
plt.xlim([-0.05, 1.03])
plt.ylim([0.45, 1.03])
plt.xlabel('Precision')
plt.ylabel('Recall')
plt.title('Precision Recall Curve')
plt.legend(loc="center left")
plt.show()



















