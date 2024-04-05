import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC  # Support Vector Classifier (SVM)
from sklearn.neighbors import KNeighborsClassifier  # K-Nearest Neighbors (KNN)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load data and labels from the pickle file
data_dict=pickle.load(open('./data.pickle','rb'))
data=np.asarray(data_dict['data'])
labels=np.asarray(data_dict['labels'])

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

#Random Forest Classifier
RF_model=RandomForestClassifier()
RF_model.fit(x_train,y_train)
y_predict=RF_model.predict(x_test)
score_rf = accuracy_score(y_predict, y_test)

print('Random Forest Classifier accuracy: {}% '.format(score_rf * 100))
f = open('RF_model.p', 'wb')
pickle.dump({'RF_model': RF_model}, f)
f.close()

#SVM classifier
SVM_model=SVC()
SVM_model.fit(x_train, y_train)
y_predict_svm = SVM_model.predict(x_test)
score_svm = accuracy_score(y_predict_svm, y_test)

print('Support Vector Machine Classifier accuracy: {}%'.format(score_svm * 100))
svm = open('SVM_model.p', 'wb')
pickle.dump({'SVM_model': SVM_model}, svm)
svm.close()

#KNN Classifier
knn_model = KNeighborsClassifier()
knn_model.fit(x_train, y_train)
y_predict_knn = knn_model.predict(x_test)
score_knn = accuracy_score(y_predict_knn, y_test)

print('K-Nearest Neighbors Classifier accuracy: {}%'.format(score_knn * 100))
Knn = open('knn_model.p', 'wb')
pickle.dump({'knn_model': knn_model}, Knn)
Knn.close()