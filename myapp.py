#required libraries
import sklearn
import os
import glob
import streamlit as st
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier



#methods
cs = ["Naive Bayes","SVM","Random Forest","Decision Tree"]
classification_space = st.sidebar.selectbox("Pick a classification method:", cs)

st.write("Results")
st.write('Dataset details here:') 
st.write("Twenty Newsgroup dataset chosen. It contains about 18000 different posts from newspapers and 20 different topics")
st.write("This app uses these models for classification: Naïve Bayes, SVM, Random Forest, Decision Tree")        
   
if st.sidebar.button('Classify'):
    
    if classification_space == "Naive Bayes":
        trainData = fetch_20newsgroups(subset='train', shuffle=True)
        st.write("Naive Bayes selected")
        classificationPipeline = Pipeline([('bow', CountVectorizer()), ('vector', TfidfTransformer()), ('classifier', MultinomialNB())])
        classificationPipeline = classificationPipeline.fit(trainData.data, trainData.target)
        test_set = fetch_20newsgroups(subset='test', shuffle=True)
        dataPrediction = classificationPipeline.predict(test_set.data)
        st.write("Accuracy of Naive Bayes:")
        st.write(np.mean(dataPrediction == test_set.target))
            
    if classification_space == "SVM":
        trainData = fetch_20newsgroups(subset='train', shuffle=True)
        st.write("SVM selected")
        classificationPipeline = Pipeline([('bow', CountVectorizer()), ('vector', TfidfTransformer()), ('classifier', SGDClassifier(loss='hinge', penalty='l1', alpha=0.0005, l1_ratio=0.17))])
        #https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html
        classificationPipeline = classificationPipeline.fit(trainData.data, trainData.target)
        test_set = fetch_20newsgroups(subset='test', shuffle=True)
        dataPrediction = classificationPipeline.predict(test_set.data)
        st.write("SVM:")    
        st.write(np.mean(dataPrediction == test_set.target))
        
    if classification_space == "Random Forest":
        trainData = fetch_20newsgroups(subset='train', shuffle=True)
        st.write("Random Forest selected")
        classificationPipeline = Pipeline([('bow', CountVectorizer()), ('vector', TfidfTransformer()), ('classifier', RandomForestClassifier(max_depth=2, random_state=0))])
        #https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html
        classificationPipeline = classificationPipeline.fit(trainData.data, trainData.target)
        test_set = fetch_20newsgroups(subset='test', shuffle=True)
        dataPrediction = classificationPipeline.predict(test_set.data)
        st.write("Random Forest:")    
        st.write(np.mean(dataPrediction == test_set.target))
     
    if classification_space == "Decision Tree":
        trainData = fetch_20newsgroups(subset='train', shuffle=True)
        st.write("Decision Tree selected")
        classificationPipeline = Pipeline([('bow', CountVectorizer()), ('vector', TfidfTransformer()), ('classifier', DecisionTreeClassifier(random_state=0))])
        #https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        classificationPipeline = classificationPipeline.fit(trainData.data, trainData.target)
        test_set = fetch_20newsgroups(subset='test', shuffle=True)
        dataPrediction = classificationPipeline.predict(test_set.data)
        st.write("Decision Tree:")    
        st.write(np.mean(dataPrediction == test_set.target))
