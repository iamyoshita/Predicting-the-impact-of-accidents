\author{Yoshita Buthalapalli}

Packages downloaded via pip install
* numpy

* pandas

* matplotlib

* datetime

* seaborn

* sklearn


### data_preprocess.py

This code is reads the dataset in US_Accidents_Dec20.csv and stores the preprocessed data in reduced\_dataset.csv. This code drops a few features, removes duplicate rows, either remove rows with missing values/Nan or fills missing values with mean of the feature. Continuous values are normalized and categorical values are represented as one hot representation. As the dataset is huge, it has been rescaled.

### {Methods}
all the following files contain the same initial code,

Dataframe is read from reduced_dataset.csv. 
All the rows with all features except the class is stored in X.

Y contains the severity class column.

X and Y are split to training and testing dataset using predefined function train_test_split. train_test_split is given a parameter (random_state=42) which ensures that same results are produced across different runs.
### {svm.py}
SVC function is used from sklearn package to get the support vector classifier. Next, GridSearchCV function is used to check which parameter fits the model better.
As svm has high computations and takes more time, the dataset has been further reduced to just 20000 rows. It is trained on training data using fit function and predictions are done on test data. The predictions are stored in y_pred.
Followed by calculating accuracy, f1 score using accuracy_score and f1_score from sklearn.metrics library. Finally we display confusion matrix using seaborn.heatmap.

### {naive_bayes.py}

MultinomialNB function is used from sklearn.naive_bayes package to get the model. It is trained on training data using fit function and predictions are done on test data. The predictions are stored in y_pred.
Followed by calculating accuracy, f1 score using accuracy_score and f1_score from sklearn.metrics library. Finally we display confusion matrix.

### {decision_Tree.py}
 DecisionTreeClassifier function is used from sklearn.tree package to get the model.
 First, we try gini criterion as parameter to measure the impurity
 It is trained on training data using fit function and predictions are done on test data. The predictions are stored in y_pred.Followed by calculating accuracy, f1 score using accuracy_score and f1_score from sklearn.metrics library.

Next, we try using entropy criterion and repeat the above process.
Finally we display confusion matrix for entropy criterion predictions and also plot the features based on their importance using classifier.feature_importances attribute.

### {RandomForest.py}

RandomForestClassifier function is used from sklearn.ensemble package to create a random forest classifier. Next, we use GridSearchCV for exhaustively searching the parameter that fits the model best. Once this is done, the model is trained on training data and predictions are done on testing data. Followed by calculating accuracy, f1 score using accuracy_score and f1_score from sklearn.metrics library. Finally we display confusion matrix and also plot the features based on their importance using classifier.feature_importances attribute.

### {graph_compare.py}
This code is for visualising the accuracy and f1 scored obtained by using the above four classification algorithms. In the code, theaccuracy and f1 scores are stored in a dictionary and a barplot is drawn using them.
