# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 19:21:20 2019

@author: floodm
"""

# run all these packages 
import pandas as pd
import numpy as np
import re
import time
from scipy import interp
import seaborn as sns
from matplotlib import pyplot as plt
import nltk
#from textblob import TextBlob # could use this for spelling corrections
from nltk.corpus import stopwords
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.feature_extraction.text import CountVectorizer  
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold #use k=10
from nltk.tokenize import WordPunctTokenizer
from nltk.stem import WordNetLemmatizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from imblearn.under_sampling import RandomUnderSampler
from math import sqrt
from PIL import Image
from wordcloud import WordCloud


### There are 7 Steps in this methodology

### Step 1. Importing & cleaning the dataframe
df = pd.read_excel(r"H:\Quality and Safety\Interns\Max\Text Classification\Data.xlsx")
# Cleaning the dataframe to remove null, non-important, or falls rows and columns ###
df = df[df["EQ Review"].notnull()]
df = df[df["INCIDENT_DESCRIPTION"].notnull()]
df = df[df['GENERAL_INC_TYPE'] != 'Fall']
#df = df[df['EQ Review'] == 1]
#df = df[df['EQ Review'] == 2]
df = df[df['INCIDENT_SEVERITY'].notnull()]
df = df[["INCIDENT_DESCRIPTION", "INCIDENT_SEVERITY", "EQ Review"]]
df.rename(columns = {'INCIDENT_DESCRIPTION':'incident',
                     'EQ Review':'eq review',
                     'INCIDENT_SEVERITY':'severity'}, inplace=True)
df["eq review"].value_counts()
# fix the ordering of row numbers
df = df.sample(frac=1, random_state=777)
df.reset_index(drop=True, inplace=True)


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



# Shows the Most common features before pre-processing
tokens = df.incident.str.cat(sep=' ')
tokens = WordPunctTokenizer().tokenize(tokens) # shows there are 2,517,071 words in this corpus
# shows how many unique words there are
unique_words = nltk.FreqDist(tokens) # shows 44,213 unique words
top_words = unique_words.most_common(50)
# ploting the most common words
plt.figure(figsize=(10,5)) #increase this to (16,5) for larger display
unique_words.plot(40)
# number or characters and words per document
average_character_count = df['incident'].str.len().mean() # mean=395 
median_character_count = df['incident'].str.len().median() # median=298
average_word_count = df['incident'].str.split(' ').str.len().mean() # mean=65
median_word_count = df['incident'].str.split(' ').str.len().median() # median=49



# Word cloud before pre-processing
tokens = df.incident.str.cat(sep=' ') # Uses all observations
#mask = np.array(Image.open(r'C:\Users\floodm\Downloads\mask-cloud3.png')) #cool mask
mask = np.array(Image.open(r'C:\Users\floodm\Downloads\mask-cloud5.png')) #boring mask
wordcloud = WordCloud(max_words=150, background_color="white",
                      collocations = False, stopwords=[], mask=mask)
wordcloud.generate(tokens)
plt.figure(figsize=(9,7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()



# Pre-processing the text with lemmatizion 
X = []
doc_length = []
for sen in range(0, len(df['incident'])):  
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



# SHows the Most common features after lemmatizion pre-processing
tokens = df.incident.str.cat(sep=' ')
tokens = WordPunctTokenizer().tokenize(tokens) #shows there are 1,267,090 words in this corpus
# shows how many unique words there are
unique_words = nltk.FreqDist(tokens) # shows 25,946 unique words
top_words = unique_words.most_common(50)
# ploting the most common words
plt.figure(figsize=(10,5)) #increase this to (16,5) for larger display
unique_words.plot(40)
# number or characters and words per document
average_character_count = df['incident'].str.len().mean() # mean=252
median_character_count = df['incident'].str.len().median() # median=196
max_character_count = df['incident'].str.len().max() # max=16,992
average_word_count = df['incident'].str.split(' ').str.len().mean() # mean=36
median_word_count = df['incident'].str.split(' ').str.len().median() # median=28
max_word_count = df['incident'].str.split(' ').str.len().max() # max=2,479


# Word cloud after pre-processing
cloudstop = stopwords.words('english') # I can add extra words like html jumk into the list of words
cloudstop.extend(["pm", "ml", "l", "mg", "g", "mcg", "oz", "lbs", "lb", "qt", "mm", 
                "wa", "ha", "aa", "pt", "patient", "Patient", "PATIENT", "rn", "nurse", "np", "chemo",
                "chemotherapy", "time", "chemo", 'UCC'])
tokens = df.incident.str.cat(sep=' ')
#mask = np.array(Image.open(r'C:\Users\floodm\Downloads\mask-cloud3.png')) #cool mask
mask = np.array(Image.open(r'C:\Users\floodm\Downloads\mask-cloud4.png')) #boring mask
wordcloud = WordCloud(max_words=150, background_color="white", collocations = False,
                      stopwords=cloudstop, mask=mask)
wordcloud.generate(tokens)
plt.figure(figsize=(9,7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# Boxplot of word counts
doc_length = [t for t in doc_length if t < 500] # should I include the max word count?
fig, ax = plt.subplots(figsize=(5, 5))
plt.boxplot(doc_length)
plt.ylabel("Word Count")
plt.title("Box Plot")
plt.xticks([1], ['Incident Reports'])
plt.show()


# Bar chart of event distribution 
sns.countplot(df['eq review'])


# Bar chart of event distribution(changing the value names for thesis document purposes) 
df2 = df
df2.rename(columns = {'eq review':'Incident Type'}, inplace=True)
df2['Incident Type'] = ['Not Reviewed' if x==0 else x for x in df2['Incident Type']]
df2['Incident Type'] = ['Reviewed' if x==1 else x for x in df2['Incident Type']]
sns.countplot(df2['Incident Type'])








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
    vectorizer = vect.fit(X_train)
    X_train_vectorized = vectorizer.transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)
    
    # Showing the top 20 Smallest & Largest TF-IDF scores
    feature_names = np.array(vect.get_feature_names())
    sorted_tfidf_index = X_train_vectorized.max(0).toarray()[0].argsort()
    print('Smallest tfidf values: \n{}\n'.format(feature_names[sorted_tfidf_index[:20]]))
    print('largest tfidf values: \n{}\n'.format(feature_names[sorted_tfidf_index[:-21:-1]]))

    ### Step 4. Feature Selection; Selecting a subset of the best 800 features(words) to use in the model via Chi-Squared method
    fs = SelectKBest(chi2, k=800)
    fs.fit(X_train_vectorized, y_train)
    X_train_fs = fs.transform(X_train_vectorized)
    X_test_fs = fs.transform(X_test_vectorized)
    
    # Displaying the best 50 chi squared words
    best_feature_indices = np.argsort(fs.pvalues_)[:800]
    best_feature_names = np.array(vect.get_feature_names())[best_feature_indices]
    print('Best 50 features: \n{}\n'.format(best_feature_names[:50]))
    
    # Displaying the worst 50 chi squared words
    p_scores = chi2(X_train_vectorized, y_train)[1]
    worst_feature_indices = np.argsort(p_scores)[-51:]
    worst_feature_names = np.array(vect.get_feature_names())[worst_feature_indices]
    print('worst 50 features: \n{}\n'.format(worst_feature_names))
    
    ### Step 5. Resampling; Random undersampling the majority class
    rus = RandomUnderSampler()
    X_train_rus, y_train_rus = rus.fit_resample(X_train_fs, y_train)
    X_test_rus, y_test_rus = rus.fit_resample(X_test_fs, y_test)
    
    ### Step 6. Building the classification model
    MNB_model = MultinomialNB()
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
    class0_indices = np.argsort(MNB_model.feature_log_prob_[0, :])
    class1_indices = np.argsort(MNB_model.feature_log_prob_[1, :])
    Class0_best_model_features = np.array(best_feature_names)[class0_indices]
    Class1_best_model_features = np.array(best_feature_names)[class1_indices]
    print('Class 0 Best 50 features: \n{}\n'.format(Class0_best_model_features[:10]))
    print('Class 1 Best 50 features: \n{}\n'.format(Class1_best_model_features[:10]))
    
    print(' ')
    stop = time.time()
print('Classifier took', round(stop-start,1) ,'seconds to run') 
# averaging together the results from the 10 classification models
print ('Cross validated accuracy score: {}'.format(np.mean(cross_val_accuracy))) # ~66%
print ('Cross validated precision score: {}'.format(np.mean(cross_val_precision))) # ~63%
print ('Cross validated recall score: {}'.format(np.mean(cross_val_recall))) # ~84%
print ('Cross validated f1 score: {}'.format(np.mean(cross_val_f1))) # ~72%
print ('Cross validated auc score: {}'.format(np.mean(cross_val_auc))) # ~78%


# 95% Confidence Interval for Recall
ci_recall = round(np.mean(cross_val_recall),3)
interval = round((1.96 * sqrt((ci_recall*(1-ci_recall))/96)) * 100, 1)
print('Recall Confidence Interval=', 
      str(ci_recall*100)+'% '+'+/-'+' %.1f' %interval+'% or', 
      '('+'%.1f' %(ci_recall*100-interval)+'%'+', ''%.1f' %(ci_recall*100+interval)+'%'+')')

# 95% Confidence Interval for Overall Error
error = round(1-np.mean(cross_val_accuracy), 3)
interval = round((1.96 * sqrt((error*(1-error))/812))*100, 1)
print('Confidence Interval=', 
      str(error*100)+'% '+'+/-'+' %.1f' %interval+'% or', 
      '('+'%.1f' %(error*100-interval)+'%'+', ''%.1f' %(error*100+interval)+'%'+')')

# Total Summed Confusion Matrix
cm = confusion_matrix(actual_sum, pred_sum)
ax= plt.subplot()
cmap=plt.get_cmap('Blues')
sns.heatmap(cm, annot=True, ax = ax, cmap=cmap, fmt='g') #annot=True to annotate cells
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(['Not Reviewed', 'Reviewed']); ax.yaxis.set_ticklabels(['Not Reviewed', 'Reviewed'])




print(" ")


### Step 7. Checking for Overfitting; the below code runs the same code as above but makes predictions on the same data that is used to build the classification model
kf = StratifiedKFold(n_splits=10)
cross_val_accuracy = []
cross_val_precision = []
cross_val_recall = []
cross_val_f1 = []
cross_val_auc = []
for train_index, test_index in kf.split(df['incident'], df['eq review']):
    X_train, X_test = df['incident'][train_index], df['incident'][test_index]
    y_train, y_test = df['eq review'][train_index], df['eq review'][test_index]
    vect = TfidfVectorizer(stop_words = stopset, min_df=2, ngram_range=(1,1))
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
    # And makes predictions on the training data
    # these scores are compared to the prevprevious scores to test for any overfitting
    MNB_model = MultinomialNB()
    MNB_model.fit(X_train_rus, y_train_rus)
    probs = MNB_model.predict_proba(X_train_rus)[:,1]
    y_pred = MNB_model.predict(X_train_rus)
    cross_val_accuracy.append(accuracy_score(y_train_rus, y_pred))
    cross_val_precision.append(precision_score(y_train_rus, y_pred))
    cross_val_recall.append(recall_score(y_train_rus, y_pred))
    cross_val_f1.append(f1_score(y_train_rus, y_pred))
    cross_val_auc.append(roc_auc_score(y_train_rus, probs))
    #print(confusion_matrix(y_train_rus, y_pred))
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



########################## END of methodology & the 7 steps ###################################





### Extra: Creating an ROC Curves plot
kf = StratifiedKFold(n_splits=10)
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
i = 0
for train_index, test_index in kf.split(df['incident'], df['eq review']):
    X_train, X_test = df['incident'][train_index], df['incident'][test_index]
    y_train, y_test = df['eq review'][train_index], df['eq review'][test_index]
    vect = TfidfVectorizer(stop_words = stopset, min_df=2, ngram_range=(1,1))
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




































