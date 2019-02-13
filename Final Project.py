#Author: Sean Chen
#CSS 490 Machine Learning Final Project 
import pandas as pd
import numpy as num
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import time as t
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.neighbors import KNeighborsClassifier as KNN

accuracy = []
def _init_main():
    my_data = pd.read_csv('css490data.csv')
    print("The dataset is by {0[0]} rows and {0[1]} columns.".format(my_data.shape))
    #clean up the data set by deleting the empty column
    my_data.drop(my_data.columns[[-1, 0]], axis=1, inplace=True)
    #check any missing values 
    print('Missing values:\n{}'.format(my_data.isnull().sum()))
    
    #diagnosis_all = list(data.shape)[0]
    category = list(my_data['diagnosis'].value_counts())

    print("\n\nThe number of Benign ", category[0], "\nThe number of Malignant ", category[1])
    sns.countplot(my_data.diagnosis, label = "Number of Count", palette = "Set2")
    plt.title("Count of Malignant and Benign")
    
    
    
    #correlation of each variable investigated by heatmap 
    col= my_data.iloc[:,1:11]
    correlation = col.corr()
    plt.figure(figsize=(11,11))
    sns.heatmap(correlation, annot=True, square=True, cmap='coolwarm')
    plt.title("Variable Correlation HeatMap")
    plt.show()
    #color_dic = {'M':'red', 'B':'blue'}
    plt.savefig("heatmap.png")
    

    #maligant and begnin cell feature distribution 
    mean= list(my_data.columns[1:11])
    plt.figure(figsize=(10,10))
    for i, feature in enumerate(mean):
        r = int(len(mean)/2)
    
        plt.subplot(r, 2, i+1)
        sns.distplot(my_data[my_data['diagnosis']=='B'][feature], bins= 10, color='gray', label='Benign');
        sns.distplot(my_data[my_data['diagnosis']=='M'][feature], bins= 10, color='black', label='Malignant');
        
    
        plt.legend(loc='upper right')

    plt.title("Maligant and Begnin Cell Feature Variable Distribution")
    plt.show()
    plt.savefig("Maligant and Begnin Cell Feature Variable Distribution.png")

    #change the type from B/M to numerical values for analysis 
    my_data['diagnosis'] = my_data['diagnosis'].map({"B": 0, "M": 1})

    diag = my_data.loc[:, 'diagnosis']
    feature_mean = my_data.loc[:,mean]

    #M is feature_mean in training, m is feature_mean in testing 
    #Y is diagnosis in training, y is diagnosis in testing
    M, m, D, d = train_test_split(feature_mean, diag, test_size = 0.3, random_state = 60)
    
 
    
    #conduct analysis of four ML models and feature selection 
    neural_networks(M, m, D, d, feature_mean, diag,accuracy)
    k_nearest_neighbors(M, m, D, d, feature_mean, diag, accuracy)
    support_vector_machine(M, m, D, d, feature_mean, diag, accuracy)
    random_forest_tree(M, m, D, d, feature_mean, diag, accuracy)
    feature_importance_rank(M,D)
    print("The Top 4 Variable Features are concave points, concavity, radius, and perimeter.")
   
    
    

def k_nearest_neighbors(M, m, D, d, feature_mean, diag, accuracy):
     
    #k-near neighbor
    training_start = t.time()
    knn = KNN()
    knn.fit(M, D)
    training_end = t.time()
    print("\nKNN\nTraining time: {0:.0000001} sec".format(training_end - training_start))
        
    testing_start = t.time()
    p = knn.predict(m)
    testing_end = t.time()
    print("Testing/Predict time: {0:.0000001} sec".format(testing_end - testing_start))
    
    validation = []
    validation = cross_val_score(knn, feature_mean, diag, cv=5)
    accuracy.append(accuracy_score(p, d))

    print("Accuracy: {0:.01%}".format(accuracy_score(p, d)))
    print("Cross validation result: {0:.01%} (+/- {1:.01%})".format(num.mean(validation), num.std(validation)*2))
    print(classification_report(d, p))
    


#Support Vector Machine Classifier
def support_vector_machine(M, m, D, d, feature_mean, diag, accuracy):
    training_start = t.time()
    svc = SVC()
    svc.fit(M, D)
    training_end = t.time()
    print("\nSVM\nTraining time: {0:.0000001} sec".format(training_end - training_start))
    
    testing_start = t.time()
    p = svc.predict(m)
    testing_end = t.time()
    print("Testing/Prediction time: {0:.0000001} sec".format(testing_end - testing_start))
    
    validation = []
    validation = cross_val_score(svc, feature_mean, diag, cv=5)
    accuracy.append(accuracy_score(p, d))

    print("Accuracy: {0:.01%}".format(accuracy_score(p, d)))
    print("Cross validation result: {0:.01%} (+/- {1:.01%})".format(num.mean(validation), num.std(validation)*2))
    print(classification_report(d, p))


#Random Forest Tree 
def random_forest_tree( M, m, D, d, feature_mean, diag,accuracy):

    training_start = t.time()
    rf = RF()
    rf.fit(M, D)
    training_end = t.time()
    print("\nRandom Forest\nTraining time: {0:.0000001} sec".format(training_end - training_start))
    
    testing_start = t.time()
    p = rf.predict(m)
    testing_end = t.time()
    print("Testing/Prediction time: {0:.0000001} sec".format(testing_end - testing_start))
    
    validation = []
    validation = cross_val_score(rf, feature_mean, diag, cv=5)
    accuracy.append(accuracy_score(p, d))


    print("Accuracy: {0:.01%}".format(accuracy_score(p, d)))
    print("Cross validation result: {0:.01%} (+/- {1:.01%})".format(num.mean(validation), num.std(validation) * 2))
    print(classification_report(d, p))
    
def neural_networks(M, m, D, d, feature_mean, diag,accuracy):
    from sklearn.neural_network import MLPClassifier as mlp
    
    training_start = t.time()
    nn = mlp()
    nn.fit(M, D)
    training_end = t.time()
    print("\nNeural Networks\nTraining time: {0:.0000001} sec".format(training_end - training_start))
    
   
    
    testing_start = t.time()
    p = nn.predict(m)
    testing_end = t.time()
    print("Testing/Prediction time: {0:.0000001} sec".format(testing_end - testing_start))
    
    validation = []
    validation = cross_val_score(nn, feature_mean, diag, cv=5)
    accuracy.append(accuracy_score(p, d))
   

    print("Accuracy: {0:.01%}".format(accuracy_score(p, d)))
    print("Cross validation result: {0:.01%} (+/- {1:.01%})".format(num.mean(validation), num.std(validation) * 2))
    print(classification_report(d, p))
    

def feature_importance_rank(M,D):
    from sklearn.ensemble import ExtraTreesClassifier as etc
    
    f = etc()
    f.fit(M, D)
    variable_importance = f.feature_importances_std = num.std([tree.feature_importances_ for tree in f.estimators_],
             axis=0)
    iterator = num.argsort(variable_importance)[::-1]

    print("The Ranking of Variable Feature Importance:")
    for f in range(M.shape[1]):
        print("%d. Variable Feature Column Number %d (%f)" % (f + 1, iterator[f], variable_importance[iterator[f]]))   
_init_main()