from sklearn.tree import DecisionTreeClassifier, plot_tree
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


df=pd.read_csv('reduced_dataset.csv')
X = df.drop('Severity',axis=1)
Y = df['Severity']
x_train, x_test, y_train, y_test = train_test_split(\
              X, Y, test_size=0.20, random_state=42, stratify=Y)

gini_model = DecisionTreeClassifier(criterion='gini')

gini_model.fit(x_train, y_train)

y_pred= gini_model.predict(x_test)

accuracy= accuracy_score(y_test, y_pred)

f1=f1_score(y_test, y_pred, average='weighted')

print('gini model accuracy : {:.3f}.'.format(accuracy))
print('gini model f1_score : {:.3f}.'.format(f1))



entropy_model = DecisionTreeClassifier(criterion='entropy',max_depth=11)

entropy_model.fit(x_train, y_train)

y_pred= entropy_model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)

f1=f1_score(y_test, y_pred, average='weighted')

print('entropy model accuracy : {:.3f}.'.format(accuracy))
print('entropy model f1_score : {:.3f}.'.format(f1))


print(classification_report(y_test, y_pred))
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)

index = ["Actual Severity 1", "Actual Severity 2", "Actual Severity 3", "Actual Severity 4"]
columns = ["Predicted Severity 1", "Predicted Severity 2", "Predicted Severity 3", "Predicted Severity 4"]
conf_matrix = pd.DataFrame(data=confmat, columns=columns, index=index)
plt.figure(figsize=(8, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="YlGnBu")
plt.title("Confusion Matrix - Decision Tree")
plt.show()
importances = pd.DataFrame(np.zeros((x_train.shape[1], 1)), columns=["importance"], index=x_train.columns)

importances.iloc[:,0] = entropy_model.feature_importances_

importances = importances.sort_values(by="importance", ascending=False)[1:30]

plt.figure(figsize=(15, 10))
sns.barplot(x="importance", y=importances.index, data=importances)
plt.show()
fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(entropy_model, max_depth=4, fontsize=10, feature_names=x_train.columns.to_list(), class_names = True, filled=True)
plt.show()
