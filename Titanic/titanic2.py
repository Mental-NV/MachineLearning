# Titanic

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
combine = pd.concat([train.drop("Survived", 1), test], ignore_index = True)

# Sex
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
combine["Sex"] = labelencoder.fit_transform(combine["Sex"])

# Fare
combine.loc[combine["Fare"].isnull(), "Fare"] = combine.loc[combine["Pclass"] == 3, "Fare"].median()

# Title
combine['Title'] = combine['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
# combine.groupby("Title")["Name"].count().sort_values(ascending = False)
combine['IsMister'] = combine['Title'] == "Mr"
combine['IsMiss'] = combine['Title'] == "Miss"
combine['IsMrs'] = combine['Title'] == "Mrs"
combine['IsMaster'] = combine['Title'] == "Master"

# Age
combine["Age_known"] = combine["Age"].isnull() == False
combine.loc[combine["Age"].isnull(), "Age"] = combine["Age"].dropna().median()
combine["IsChild"] = combine["Age"] <= 10
combine["IsYoung"] = (combine["Age"] >= 18) & (combine["Age"] <= 40)
combine["IsYoung_M"] = combine["IsYoung"] & (combine["Sex"] == 1)
combine["IsYoung_F"] = combine["IsYoung"] & (combine["Sex"] == 0)

# Family
combine["Family"] = combine["SibSp"] + combine["Parch"]
combine["IsAlone"] = combine["SibSp"] + combine["Parch"] == 0
combine["IsBigFamily"] = (combine["SibSp"] > 2) | (combine["Parch"] > 3)

# Embarked
combine.loc[combine["Embarked"].isnull(), "Embarked"] = "C"
combine["Embarked"] = combine["Embarked"].astype("category")
combine["Embarked"].cat.categories = [0,1,2]
combine["Embarked"] = combine["Embarked"].astype("int")
combine["Embarked_C"] = combine["Embarked"] == 0
combine["Embarked_Q"] = combine["Embarked"] == 1
combine["Embarked_S"] = combine["Embarked"] == 2

# Ticket
combine['SharedTicket'] = 0
groupTicket = combine.groupby('Ticket')
for ticket, group in groupTicket["Ticket"]:
    if group.count() > 1:
        combine.loc[combine["Ticket"] == ticket, "SharedTicket"] = 1

combine["Ttype"] = combine["Ticket"].str[0]
combine['Bad_ticket'] = combine['Ttype'].isin(['3','4','5','6','7','8','A','L','W'])

combine["Ttype"] = combine["Ttype"].astype("category")
combine["Ttype"].cat.categories = [0, 1, 2, 3, 4, 5, 6, 7, 8 , 9, 10, 11, 12, 13, 14, 15]
combine["Ttype"] = combine["Ttype"].astype("int")
for t in range(0, 16):
    combine["Ttype_" + str(t)] = combine["Ttype"] == t



# Cabin
combine['Cabin_known'] = combine["Cabin"].isnull() != True
combine['Deck'] = combine['Cabin'].str[0]
combine['Deck'] = combine['Deck'].fillna(value='U')
combine["Deck"] = combine["Deck"].astype("category")
combine["Deck"].cat.categories = [0,1,2,3,4,5,6,7,8]
combine["Deck"] = combine["Deck"].astype("int")
for d in range(0, 9):
    combine["Deck_" + str(d)] = combine["Deck"] == d
       
cols = ['Pclass', 'IsYoung_M', 'Family', 'Ttype_13', 'Ttype_6', 'Ttype_14', 'Parch', 'SharedTicket', 'Fare', 'IsMaster', 'Ttype_0', 'Sex', 'Deck_2']
X = combine.loc[combine["PassengerId"] <= 891, cols]
X_submission = combine.loc[combine["PassengerId"] > 891, cols]
y = train.loc[:, "Survived"]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

scores = []

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0, verbose = 1)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Accuracy
from sklearn.metrics import accuracy_score
s = accuracy_score(y_test, y_pred)

scores.append(tuple(("Logistic Regression", s)))


# Fitting GaussianNB classifier to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Accuracy
from sklearn.metrics import accuracy_score
s = accuracy_score(y_test, y_pred)

scores.append(tuple(("GaussianNB", s)))


# Fitting Random Forest Classifier to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)


# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Accuracy
from sklearn.metrics import accuracy_score
s = accuracy_score(y_test, y_pred)

scores.append(tuple(("Random Forest", s)))



# Displayng scores
def getKey(custom):
    return custom[1]

scores.sort(key = getKey)

for val in scores:
    print("%s: %.5f" %(val[0], val[1]))
    
'''hist_data = []
hist_labels = []
for val in scores:
    hist_data.append(val[1])
    hist_labels.append(val[0])

# Horizontal bar chart with scores
val = hist_data
pos = np.arange(len(val)) + .5
plt.barh(pos, val, align = 'center')
plt.yticks(pos, hist_labels)
plt.xlabel("Regressors scores")
axes = plt.gca()
xdiff = max(val) - min(val)
axes.set_xlim([min(val) - xdiff*.1, max(val) + xdiff*.1])
plt.show()'''

# Write scores to the file
from datetime import datetime
newScore = pd.DataFrame()
newScore.loc[0, "Time"] = str(datetime.now())
newScore.loc[0, "Score"] = scores[-1][1]
sc = pd.read_csv("scores.csv")
sc = sc.append(newScore)
sc.to_csv("scores.csv", index = False)

# Prepare the submission
submission = pd.DataFrame()
submission["PassengerId"] = test.loc[:, "PassengerId"]
submission["Survived"] = classifier.predict(X_submission)
submission.to_csv("submission.csv", index = False)

def Solve(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)   
    return accuracy_score(y_test, y_pred)

# Solve(X, y)

def FindColPosition(combine, train, cols, newCol):
    scores = []
    for pos in range(0, len(cols)):
        newCols = cols[:]
        newCols.insert(pos, newCol)
        X = combine.loc[combine["PassengerId"] <= 891, newCols]
        y = train.loc[:, "Survived"]
        s = Solve(X, y)
        scores.append(tuple((pos, s)))
    scores.sort(key = getKey, reverse = True)
    #for val in scores[:3]:
    #    print("%s: %.5f, %s" %(val[0], val[1], newCol))
    return scores[0]

# FindColPosition(combine, train, cols, 'Sex')


def FindColumns():
    best = 0
    goodColumns = combine.columns.values.tolist()
    goodColumns.remove("PassengerId")
    goodColumns.remove("Name")
    goodColumns.remove("Ticket")
    goodColumns.remove("Cabin")
    goodColumns.remove("Deck")
    goodColumns.remove("Title")
    goodColumns.remove("Embarked")
    goodColumns.remove("Ttype")
    cols2 = goodColumns[:]
    for startC in cols2:
        cols = goodColumns[:]
        startCols = [startC]
        cols.remove(startC)
        better = 0
        while len(cols) > 0:
            max_score = 0
            max_pos = -1
            max_cols = []
            max_c = ""
            for c in cols:
                s = FindColPosition(combine, train, startCols, c)
                score = s[1]
                if (score > max_score):
                    max_score = score
                    max_pos = s[0]
                    max_cols = startCols[:]
                    max_cols.insert(max_pos, c)
                    max_c = c
            if max_score > better:
                better = max_score
                cols.remove(max_c)
                startCols = max_cols
                if better > best:
                    best = better
                    print("%s, %s" %(max_score, max_cols))
                    bestTable = pd.read_csv("cols.csv")
                    btLen = len(bestTable.values)
                    bestTable.loc[btLen, "Score"] = max_score
                    bestTable.loc[btLen, "Cols"] = str(max_cols)
                    bestTable.to_csv("cols.csv", index = False)
            else:
                break
            print("%s, Cols len: %s" %(startC, len(cols)))
            
# FindColumns()