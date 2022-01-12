import matplotlib.pyplot as plt
import seaborn as sns

accuracy=dict()
f1=dict()
accuracy["decision tree"]=0.703
accuracy["random forest"]=0.764
accuracy["multinomial naive bayes"]=0.561
accuracy["svm"]=0.484
f1["decision tree"]=0.703
f1["random forest"]=0.782
f1["multinomial naive bayes"]=0.552
f1["svm"]=0.351
plt.figure(figsize=(5, 3))
plt.title("Accuracy on Validation set for each model")
sns.barplot(list(range(len(accuracy))), list(accuracy.values()))
plt.xticks(range(len(accuracy)), labels=accuracy.keys())
plt.show()

plt.figure(figsize=(5, 8))
plt.title("F1 Score on Validation set for each model")
sns.barplot(list(range(len(f1))), list(f1.values()))
plt.xticks(range(len(f1)), labels=f1.keys())
plt.show()
