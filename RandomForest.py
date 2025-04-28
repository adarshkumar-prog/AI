# Importing necessary libraries
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Loading the Iris dataset
iris = load_iris()
X = iris.data          # Features
y = iris.target        # Labels

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initializing the Random Forest values
rf_clf = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, random_state=42)

# Training the model
rf_clf.fit(X_train, y_train)

# Making predictions
y_pred = rf_clf.predict(X_test)

# Evaluating the models accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
