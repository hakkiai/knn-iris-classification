import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
k = 3

knn_classifier = KNeighborsClassifier(n_neighbors=k)

knn_classifier.fit(X_train, y_train)  # Corrected method name: fit()

y_pred = knn_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy : {accuracy * 100:.2f}%")

for i in range(len(y_test)):
    if y_pred[i] == y_test[i]:
        print(f"Sample {i+1}: Predicted={iris.target_names[y_pred[i]]}, Actual = {iris.target_names[y_test[i]]} (Correct)")
    else:
        print(f"Sample {i+1}: Predicted = {iris.target_names[y_pred[i]]}, Actual = {iris.target_names[y_test[i]]} (Wrong)")

