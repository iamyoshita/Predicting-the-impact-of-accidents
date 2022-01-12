import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv('reduced_dataset.csv')
Y = df["Severity"]
X = df.drop("Severity", axis=1)

x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=42)

mnb = MultinomialNB()
mnb.fit(x_train, y_train)

y_pred = mnb.predict(x_test)

accuracy= accuracy_score(y_test, y_pred)
f1= f1_score(y_test, y_pred, average="macro")
print("accuracy",accuracy)
print("f1 score",f1)
print(classification_report(y_test, y_pred))

confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)

index = ["Actual Severity 1", "Actual Severity 2", "Actual Severity 3", "Actual Severity 4"]
columns = ["Predicted Severity 1", "Predicted Severity 2", "Predicted Severity 3", "Predicted Severity 4"]
conf_matrix = pd.DataFrame(data=confmat, columns=columns, index=index)
plt.figure(figsize=(8, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="YlGnBu")
plt.title("Confusion Matrix for Multinomial Naive Bayes")
plt.show()
