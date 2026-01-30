# Write a program to implement k-Nearest Neighbour algorithm to classify the iris data set. Print both correct and wrong predictions

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_tes_split as tts
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = tts(X, y, test_size = 0.3, random_state = 42)

# Initialize the k-Nearest Neighbors Classifiers
k = 5
knn_classifier = KNeighborsClassifier(n_neighbors= k)

#Train the classifier
knn_classifier.fit(X_train, y_train)

y_pred = knn_classifier.predict(X_test)

#Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of k-NN (k={k}): {accuracy:.2f}")

#Print correct and wring predictions
print("\n---- Predictions ------")
correct_predictions = []
wrong_predictions = []

for i in range(len(y_test)):
    if y_pred[i] == y_test[i]:
        correct_predictions.append({
            "index": i,
            "features": X_test[i],
            "actual": iris.target_names[y_test[i]],
            "predicted": iris.target_names[y_pred[i]]
        })
    else:
        wrong_predictions.append({
            "index": i,
            "features": X_test[i],
            "actual": iris.target_names[y_test[i]],
            "predicted": iris.target_names[y_pred[i]]
        })


print(f"\nTotal predictions: {len(y_test)}")

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the k-Nearest Neighbors classifier
# You can experiment with different values of k (n_neighbors)
k = 5  # Example: using k=5
knn_classifier = KNeighborsClassifier(n_neighbors=k)

# Train the classifier
knn_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = knn_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of k-NN (k={k}): {accuracy:.2f}")

# Print correct and wrong predictions
print("\n--- Predictions ---")
correct_predictions = []
wrong_predictions = []

for i in range(len(y_test)):
    if y_pred[i] == y_test[i]:
        correct_predictions.append({
            "index": i,
            "features": X_test[i],
            "actual": iris.target_names[y_test[i]],
            "predicted": iris.target_names[y_pred[i]]
        })
    else:
        wrong_predictions.append({
            "index": i,
            "features": X_test[i],
            "actual": iris.target_names[y_test[i]],
            "predicted": iris.target_names[y_pred[i]]
        })

print(f"\nTotal predictions: {len(y_test)}")
print(f"Correct predictions: {len(correct_predictions)}")
print(f"Wrong predictions: {len(wrong_predictions)}")

print("\n--- Correct Predictions ---")
if correct_predictions:
    for pred in correct_predictions:
        print(f"Index: {pred['index']}, Features: {pred['features']}, Actual: {pred['actual']}, Predicted: {pred['predicted']}")
else:
    print("No correct predictions found (this is highly unlikely with a good model).")

print("\n--- Wrong Predictions ---")
if wrong_predictions:
    for pred in wrong_predictions:
        print(f"Index: {pred['index']}, Features: {pred['features']}, Actual: {pred['actual']}, Predicted: {pred['predicted']}")
else:
    print("No wrong predictions found!")