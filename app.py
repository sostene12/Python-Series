# macine learning
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import sklearn.externals
import joblib

from sklearn import tree

music_data = pd.read_csv('music.csv')
X = music_data.drop(columns=['genre'])
Y = music_data['genre']

model = DecisionTreeClassifier()
model.fit(X,Y)

tree.export_graphviz(
    model,out_file='music_recommender.dot',
    feature_names=['age','gender'],
    class_names=sorted(Y.unique()),
    label='all',
    rounded=True,
    filled=True
)

# model = joblib.load('music-recommender.joblib')

# predictions = model.predict([[21,1]])
# predictions


