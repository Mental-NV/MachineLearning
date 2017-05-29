# Titanic

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
combine = pd.concat([train.drop("Survived", 1), test])

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
train["Sex"] = labelencoder.fit_transform(train["Sex"])
test["Sex"] = labelencoder.transform(test["Sex"])

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(train[["Age"]])
train["Age"] = imputer.transform(train[["Age"]]).ravel()
test["Age"]  = imputer.transform(test[["Age"]]).ravel()
test.loc[test["Fare"].isnull(), "Fare"] = combine.loc[combine["Pclass"] == 3, "Fare"].median()

X = train.loc[:, ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]]
X_submission  =  test.loc[:, ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]]
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

# Fitting KNN classifier to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Accuracy
from sklearn.metrics import accuracy_score
s = accuracy_score(y_test, y_pred)

scores.append(tuple(("K-Nearest Neighbors", s)))


# Fitting SVM classifier to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Accuracy
from sklearn.metrics import accuracy_score
s = accuracy_score(y_test, y_pred)

scores.append(tuple(("SVM linear", s)))



# Fitting classifier to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Accuracy
from sklearn.metrics import accuracy_score
s = accuracy_score(y_test, y_pred)

scores.append(tuple(("SVM rbf", s)))



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



# Fitting Decision Tree classifier to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Accuracy
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)
s = accuracy_score(y_test, y_pred)

scores.append(tuple(("Decision Tree", s)))


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
    
hist_data = []
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
plt.show()

# Prepare the submission
submission = pd.DataFrame()
submission["PassengerId"] = test.loc[:, "PassengerId"]
submission["Survived"] = classifier.predict(X_submission)
submission.to_csv("submission.csv", index = False)