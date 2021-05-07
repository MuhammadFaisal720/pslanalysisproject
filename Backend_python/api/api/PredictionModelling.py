
from collections import Counter

import numpy as np
import pandas as pd
import os, sys
import matplotlib.pyplot as plt
#import scikitplot as skplt

from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import warnings
import joblib
import seaborn as sns
from sklearn.metrics import r2_score
from scipy import stats
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report

def results(target_test, predicted_test,ModelName,labels):
    target_names = labels
    print(classification_report(target_test, predicted_test, target_names=target_names))
    y_test = target_test
    preds = predicted_test
    rms = np.sqrt(np.mean(np.power((np.array(y_test) - np.array(preds)), 2)))
    score = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    pearson_coef, p_value = stats.pearsonr(y_test, preds)

    print("root mean square:", rms)
    print("score:", score)
    print("mean absolute error:", mae)
    print("mean squared error:", mse)
    print("pearson_coef:", pearson_coef)
    print("p_value:", p_value)
    print("=======================================================================\n\n")
    #skplt.metrics.plot_confusion_matrix(
    #    y_test,
    #    preds,
    # 3    figsize=(10, 6), title="Confusion matrix\n Deposite Category "+ModelName)
    #plt.xlim(-0.5, len(np.unique(y_test)) - 0.5)
    #plt.ylim(len(np.unique(y_test)) - 0.5, -0.5)
    #plt.savefig('cvroc.png')
    #plt.show()
def PMF():
    warnings.filterwarnings('ignore')


    # Loading dataset
    data = pd.read_csv("matches.csv")

    data.head()

    # Removing unwanted columns
    data.drop(columns=['venue', 'player_of_match', 'dl_applied','umpire1','umpire2','umpire3','date','city','season','id'], inplace=True)

    # Label encoding toss_decision
    data.toss_decision = data.toss_decision.map({'bat':1, 'field':0})

    # Encoding result
    data.result = data.result.map({'normal':1, 'tie':2, 'no result':0})

    r = len(data.team2.unique())
    teams = data.team1.unique()
    mapping = {}

    for i in range(5): # There are 14 teams.
        mapping[teams[i]] = i

    data.toss_winner = data.toss_winner.map(mapping)

    # Encoding team data in numeric form
    data.team1 = data.team1.map(mapping)
    data.team2 = data.team2.map(mapping)
    mapping # A value is repeated

    label=list(mapping)

    data.winner = data.winner.map(mapping)

    # Removing NA Fields
    data.dropna(axis=0,inplace=True)

    data.winner = data.winner.astype(int)

    data.head()

    len(data)

    data.drop(columns=["win_by_runs", "win_by_wickets"], axis=1, inplace=True)

    data.head()

    data.drop(columns=["toss_decision", "result"], inplace=True)

    data.head()

    labels = data.winner.values
    features = data.drop(columns=["winner"], axis=1).values

    labels_copy = data.winner.values
    features_copy = data.drop(columns=["winner"], axis=1).values

    features.shape

    # We have three input dim
    labels.shape

    # As there is no activaton function that can predict 'winner', we are one hot encoding it.
    #labels = to_categorical(labels)

    labels

    labels.shape

    # Now we will use a softmax which will provide probs for 14 different classes aka teams.

    #features_train, features_test, labels_train,labels_test = train_test_split(features, labels, shuffle=True, random_state=42)
    #features_copy_train, features_copy_test, labels_copy_train,labels_copy_test = train_test_split(features, labels, shuffle=True, random_state=42)

    #len(features_train)

    #len(features_test)

    # converting labels into numeric
    le = preprocessing.LabelEncoder()
    target=le.fit_transform(labels)
    # features = preprocessing.MinMaxScaler().fit_transform(features)
    feature_train, feature_test, target_train, target_test = train_test_split(features, target)
    #Create a Gaussian Classifier
    clff=RandomForestClassifier(n_estimators=800)
    #Train the model using the training sets y_pred=clf.predict(X_test)
    clff = clff.fit(feature_train,target_train)
    y_predd1=clff.predict(feature_test)
    # Model Accuracy, how often is the classifier correct?
    print("Random Forest Accuracy:",metrics.accuracy_score(target_test, y_predd1))
    results(target_test, y_predd1,"Random Forest",label)

    target_names = labels

    #sns.heatmap(confusion_matrix(target_test,y_predd1), annot=True, cmap='Blues')
    model_path = 'Models\\'
    joblib.dump(clff, model_path+"model_3000.sav")
    #Create a KNN Classifier
    knn=KNeighborsClassifier()
    #Train the model using the training sets y_pred=clf.predict(X_test)
    knn = knn.fit(feature_train,target_train)
    y_predd2=knn.predict(feature_test)
    # Model Accuracy, how often is the classifier correct?
    print("Accuracy KNN:",metrics.accuracy_score(target_test, y_predd2))
    results(target_test, y_predd2,"KNN",label)
    model_path = 'Models\\'
    joblib.dump(knn, model_path+"model_knn.sav")
    # training a linear SVM classifier
    svm_model_linear = SVC(kernel = 'linear', C = 1).fit(feature_train,target_train)
    y_predd3 = svm_model_linear.predict(feature_test)
    # model accuracy for X_test
    accuracy = svm_model_linear.score(feature_test, target_test)
    print("Accuracy SVM:",metrics.accuracy_score(target_test, y_predd3))
    results(target_test, y_predd3,"SVM",label)
    model_path = 'Models\\'
    joblib.dump(svm_model_linear, model_path+"model_svm.sav")
    model1 = RandomForestClassifier()
    model2 = KNeighborsClassifier()
    model3 = LogisticRegression()
    Voting = VotingClassifier(estimators=[('RF', model1 ), ('knn', model2),('lr',model3)], voting='hard')
    Voting.fit(feature_train,target_train)
    vpredictions = Voting.predict(feature_test)
    vscore = Voting.score(feature_test, target_test)
    print("Voting Score", vscore)
    results(target_test, vpredictions,"Voting Classifier",label)
    model_path = 'Models\\'
    joblib.dump(Voting, model_path+"model_voting.sav")
    print(y_predd1)
    Result=[]
    for p in range(len(y_predd1)):
        Results=[label[np.int(y_predd1[p])], label[np.int(y_predd2[p])], label[np.int(y_predd3[p])],
                      label[np.int(vpredictions[p])]]
        counter = Counter(Results)
        most_occur = counter.most_common(1)
        #print(most_occur)
        Result.append(most_occur[0][0])
        #print(Result)
    #for p in range(len(feature_test)):
    counter = Counter(Result)
    most_occur = counter.most_common(5)
    print(most_occur)
    print(most_occur)
    return most_occur

print(PMF())