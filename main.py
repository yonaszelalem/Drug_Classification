'''  During medical researcher of treatment, each patient responded to one of two
medications. We call them drug A and drug B. Aim of this project is to build a model
to find out which drug might be appropriate for a future patient with the same illness.
The feature sets of this dataset are age, gender, blood pressure, and cholesterol of
our group of patients and the target is the drug that each patient responded to.
 here decision tree model is used. '''

import numpy as np
import sklearn.tree as tree
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Downloading the file
path='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/drug200.csv'

import requests
def download(url, filename):
    response = requests.get(url, allow_redirects=True)
    open(filename, 'wb').write(response.content)

download(path,"drug200.csv")
path="drug200.csv"

data = pd.read_csv(path)

# Pre Processing

x = data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values

# Changing Categorical Data to Numerical Using LabelEncoder

from sklearn import preprocessing

le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
x[:,1] = le_sex.transform(x[:,1])

le_BP = preprocessing.LabelEncoder()
le_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])
x[:,2] = le_BP.transform(x[:,2])


le_Chol = preprocessing.LabelEncoder()
le_Chol.fit([ 'NORMAL', 'HIGH'])
x[:,3] = le_Chol.transform(x[:,3])

# Now we can fill the target variable
y = data['Drug']

# Setting up the Decision Tree

from sklearn.model_selection import train_test_split

#The train_test_split will need the parameters:
#x, y, test_size=0.3, and random_state=3.
# The X and y are the arrays required before the split,
# the test_size represents the ratio of the testing dataset,
# and the random_state ensures that we obtain the same splits.

x_trainset, x_testset, y_trainset, y_testset = train_test_split(x, y, test_size=0.3, random_state=3)

# Modeling
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth=4)
drugTree.fit(x_trainset, y_trainset)

# Prediction
predTree = drugTree.predict(x_testset)

# Evaluation
from sklearn import metrics
import matplotlib.pyplot as plt

print("Decision Tree's Accuracy: ", metrics.accuracy_score(y_testset, predTree))

# Visualization
tree.plot_tree(drugTree)
plt.show()