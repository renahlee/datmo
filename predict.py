import os
import pickle
import pandas as pd
import random
random.seed(3)

# loading model file
model_filename = os.path.join('model.dat')
model = pickle.load(open(model_filename, 'rb'))
Species_class_map = {0:'Iris-setosa', 1:'Iris-versicolor', 2:'Iris-virginica'}

# Test feature
X_test = [[ 6.9, 3.2, 5.7, 2.3]]
y_pred = model.predict(X_test)
y_pred = [round(value) for value in y_pred]
print({'Species': Species_class_map[y_pred[0]]})
