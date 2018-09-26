import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from utils.feature_extractor import get_mfcc
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import recall_score
import pickle
import time

input_data = pd.read_csv('input_data.csv')
start = time.time()
test_data = pd.read_csv('test_data.csv')


# -----> Male 1
# -----> Female 0

train_data = pd.DataFrame()
train_data['fname'] = input_data['filename']
input_data = train_data['fname'].apply(get_mfcc)
input_data.to_csv('input_features.csv')
end = time.time()
print('done loading train mfcc')

print('Feature extraction time = ' + str(end - start))
test_df = pd.DataFrame()
test_data['fname'] = test_data['filename']
test_df = test_data['fname'].apply(get_mfcc)
test_df.to_csv('test_features.csv')
print('done loading test mfcc')
test_data = pd.read_csv('test_features.csv')


#print(test_data.head())
train_data['label'] = input_data['Gender']



X, y = train_data.iloc[:, :-1].values, train_data.iloc[:, -1].values
gender_encoder = LabelEncoder()
y = gender_encoder.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

pipe_svc = Pipeline([
    ('stdsc', StandardScaler()),
    ('svc', SVC(random_state=1, kernel='rbf', gamma=0.01, C=10.0))
])

pipe_svc.fit(X_train, y_train)

FILE_NAME = 'gender_classification_model.sav'
pickle.dump(pipe_svc, open(FILE_NAME, 'wb'))

FILE_NAME = 'gender_classification_model.sav'
start = time.time()
pipe_svc = pickle.load(open(FILE_NAME, 'rb'))
test_data['label'] = pipe_svc.predict(test_data.drop(['fname'], axis=1))
end = time.time()
print('classification time + Feature extraction time = ' + str(end - start))

#print(test_data.head())
#test_data = test_data.iloc[:, -2:].values
#test_data_df = pd.DataFrame(test_data, columns=['filename', 'gender'])
#test_data_df.to_csv('test_predictions.csv', index=False)

print('Test Accuracy: %.3f' % pipe_svc.score(X_test, y_test))
print('Training Accuracy: %.3f' % pipe_svc.score(X_train, y_train))
print('Test Recall: %.3f' % recall_score(y_test, pipe_svc.predict(X_test)) )

scores = cross_val_score(estimator=pipe_svc,
                        X=X_train,
                        y=y_train,
                        cv=10,
                        n_jobs=1)
print('Cross validation scores: %s' % scores)


param_range = [0.001, 0.01, 0.1, 1.0, 10.0]
param_grid = [{'svc__C': param_range, 'svc__kernel': ['linear']},
              {'svc__C': param_range, 'svc__gamma': param_range,
               'svc__kernel': ['rbf']}]
gs = GridSearchCV(estimator=pipe_svc,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=10)
gs = gs.fit(X_train, y_train)

print(gs.best_score_)
print(gs.best_params_)



