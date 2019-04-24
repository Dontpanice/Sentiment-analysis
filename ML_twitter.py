import pandas as pd
import re
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.cross_validation import train_test_split
from sklearn import naive_bayes
from sklearn.metrics import roc_auc_score

from sklearn.tree import DecisionTreeClassifier
from sklearn import ensemble
from sklearn import linear_model
from sklearn import svm
from sklearn import neural_network

#%%   #read the data   
traindata = pd.read_csv('train.tsv', header=None, sep='\t',names=['label','text'])
devdata = pd.read_csv('dev.tsv', header=None, sep='\t',names=['label','text'])

#Separate traindata
Xtrain = traindata['text']
Ytrain = traindata['label']

Xdev = devdata['text']
Ydev = devdata['label']

#%%    #Tokenize words in text / reading the tweets

tokenize_re = re.compile(r'''
                         \d+[:\.]\d+
                         |(https?://)?(\w+\.)(\w{2,})+([\w/]+)
                         |[@\#]?\w+(?:[-']\w+)*
                         |[^a-zA-Z0-9 ]+''',
                         re.VERBOSE)

def tokenize(text):
    return [ m.group() for m in tokenize_re.finditer(text) ]

Xtrain_toc = []
for i in range (0,len(Xtrain)):
    Xtrain_toc.append(tokenize(Xtrain[i]))
    
    
    
    
#%% Baseline
count = Ytrain.value_counts()
probs = {}
probs['neutral'] = count['neutral']/len(Ytrain)
probs['positive'] = count['positive']/len(Ytrain)
probs['negative'] = count['negative']/len(Ytrain)
print(probs)

dumbGuesses = []
for i in range(len(Ydev)):
    dumbGuesses.append('neutral') 
print(accuracy_score(Ydev, dumbGuesses)) #have an accuracy of 0.4480 on development set


#%% CLEAN DATA

import re

#Remove URLS
for i in range (0,len(Xtrain)):
    Xtrain[i] = re.sub(r'http\S+',"",Xtrain[i])


#Standardize words to corrected spelling. 


cluster_data = pd.read_csv('50mpaths2.txt', header=None, sep='\t',names=['cluster','word','nr'])

cl_word = cluster_data['word']
cl_nr = cluster_data['cluster']


for i in range(0,len(Xtrain_toc)):
    for j in range(0, len(Xtrain_toc[i])):
#        print(Xtrain_toc[i][j])
        for cl in range(0, len(cluster_data)):
            if Xtrain_toc[i][j] == cl_word[cl]:
                Xtrain_toc[i][j] = cl_nr[cl]
            else:
                Xtrain_toc[i][j] = ""
        



#%% #vectorize trainingset and apply stopwords (throws out words like "the", "and" etc...)
Xtogether = Xtrain.append(Xdev)

stopset = set(stopwords.words('english'))

vectorizer = TfidfVectorizer(use_idf=True, lowercase=True, strip_accents='unicode', stop_words=stopset, max_features = 5000)

Xtrain_vec_together = vectorizer.fit(Xtogether)

Xtrain_vec = vectorizer.transform(Xtrain)

#%%   #classifier Naive Bayers

#methods = []
#M = [
#     svm.LinearSVC(),
#     linear_model.LogisticRegression(),
#     linear_model.Perceptron(),
#     ensemble.GradientBoostingClassifier(),
#     ensemble.RandomForestClassifier(),
#     DecisionTreeClassifier(),
#     naive_bayes.MultinomialNB()]
#
#def comparisons(M):
#    """
#    M: a list of classifiers
#    
#    """
#    results = []
#    Xdev_vec = vectorizer.transform(Xdev)
#    
#    for i in range(0,len(M)):
#        clf = M[i]       
#        clf.fit(Xtrain_vec, Ytrain)
#        
#        
#        predicted_Ydev = clf.predict(Xdev_vec.toarray())
#        
#        results.append(accuracy_score(Ydev, predicted_Ydev))
#        
#    return results;
#        
#        
#        
#results = comparisons(M);
      
        

#clf = neural_network.MLPClassifier() #0.5725
#clf = svm.LinearSVC() #0.6142
#clf = linear_model.LogisticRegression() #0.6228
#clf = linear_model.Perceptron() #0.6042
#clf = ensemble.GradientBoostingClassifier() #0.6221
#clf = ensemble.RandomForestClassifier() #0.6188
#clf = DecisionTreeClassifier() #0.5619
#clf = naive_bayes.MultinomialNB() #0.5863
        
#clf.fit(Xtrain_vec, Ytrain)

#%% # Test accuracy
#Xdev_vec = vectorizer.transform(Xdev)
#predicted_Ydev = clf.predict(Xdev_vec.toarray())
##
#print(accuracy_score(Ydev,predicted_Ydev))
##acc = 
#roc_auc_score(Ydev, predicted_Ydev)




#%%


clf = ensemble.GradientBoostingClassifier(learning_rate = 0.2, n_estimators = 100)
clf.fit(Xtrain_vec, Ytrain)


Xdev_vec = vectorizer.transform(Xdev)
predicted_Ydev = clf.predict(Xdev_vec.toarray())



print(accuracy_score(Ydev,predicted_Ydev))









#%%

feature_plot = pd.Series(clf.feature_importances_, vectorizer.get_feature_names()).sort_values(ascending=False)

feature_plot[:100].plot(kind='bar', title='Feature Importances')





#%%
from sklearn.metrics import confusion_matrix
import matplotlib.pylab as plt

error_D = confusion_matrix(Ydev, predicted_Ydev)

print(Ydev.value_counts())


print("predicted neutral: " , np.count_nonzero(predicted_Ydev=="neutral"))
print("predicted postitive: " , np.count_nonzero(predicted_Ydev=="positive"))
print("predicted negative: " , np.count_nonzero(predicted_Ydev=="negative"))


plt.figure();
plt.imshow(error_D, cmap='hot')
plt.show()

#%%













#