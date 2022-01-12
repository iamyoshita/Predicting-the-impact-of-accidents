from sklearn.ensemble import RandomForestClassifier
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv('reduced_dataset.csv')
Y = df["Severity"]
X = df.drop("Severity", axis=1)

x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=42)

rfc = RandomForestClassifier(n_jobs=-1, random_state=42)
# n_estimators= number of trees
parameters = [{"n_estimators": [50, 250, 500], "max_depth": [5, 15, 35]}]
grid = GridSearchCV(rfc, parameters, verbose=5, n_jobs=-1)
grid.fit(x_train, y_train)

rfc.fit(x_train, y_train)

y_pred = rfc.predict(x_test)

accuracy= accuracy_score(y_test, y_pred)
f1= f1_score(y_test, y_pred, average="macro")

print("accuracy",accuracy);
print("f1 score",f1);
print(classification_report(y_test, y_pred))

confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)

index = ["Actual Severity 1", "Actual Severity 2", "Actual Severity 3", "Actual Severity 4"]
columns = ["Predicted Severity 1", "Predicted Severity 2", "Predicted Severity 3", "Predicted Severity 4"]
conf_matrix = pd.DataFrame(data=confmat, columns=columns, index=index)
plt.figure(figsize=(8, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="YlGnBu")
plt.title("Confusion Matrix - Random Forest")
plt.show()

importances = pd.DataFrame(np.zeros((x_train.shape[1], 1)), columns=["importance"], index=x_train.columns)

importances.iloc[:,0] = rfc.feature_importances_

importances = importances.sort_values(by="importance", ascending=False)[:30]

plt.figure(figsize=(15, 10))
sns.barplot(x="importance", y=importances.index, data=importances)
plt.show()

