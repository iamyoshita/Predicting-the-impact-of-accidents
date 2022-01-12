import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm

df=pd.read_csv('reduced_dataset.csv')
parameters = [ {"kernel": ["poly"], "C": [1.], "degree": [2]},{"kernel": ["poly"], "C": [1.], "degree": [3]}]
svc = svm.SVC(verbose=5, random_state=42) #random_state is used to produce same results accross different runs
grid = GridSearchCV(svc, parameters, verbose=5, n_jobs=-1)

sample = df.sample(5_000, random_state=42)
Y = sample["Severity"]
X = sample.drop("Severity", axis=1)
grid.fit(X, Y)

print("Scores:")
svc.fit(X, Y)
sample = df.sample(20_000, random_state=42)
Y = sample["Severity"]
print(sample["Severity"].value_counts())
X = sample.drop("Severity", axis=1)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.2, random_state=42)
svc = svm.SVC(**grid.best_params_, random_state=42)
svc.fit(x_train, y_train)

y_pred = svc.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="macro")
print("accuracy",accuracy)
print("f1",f1)
print(classification_report(y_test, y_pred))
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)

yaxisnames = ["Actual Severity 1", "Actual Severity 2", "Actual Severity 3", "Actual Severity 4"]
xaxisnames = ["Predicted Severity 1", "Predicted Severity 2", "Predicted Severity 3", "Predicted Severity 4"]
conf_matrix = pd.DataFrame(data=confmat, columns=xaxisnames, index=yaxisnames)
plt.figure(figsize=(8, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="YlGnBu")
plt.title("Confusion Matrix for Support Vector Machine")
plt.show()
