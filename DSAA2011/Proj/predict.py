import data
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def evaluate(name, y_test, y_pred) :
	print(f"metric of {name} :")

	print(f"accuracy : {accuracy_score(y_test, y_pred)}")
	print(classification_report(y_test, y_pred))

	print(f"confusion matrix :\n{confusion_matrix(y_test, y_pred)}")
	print()
def pred_lr(X_train, y_train, X_test, y_test):
	# Logistic Regression
	lr = LogisticRegression(C=1e9, max_iter=1000)
	lr.fit(X_train, y_train)
	y_pred = lr.predict(X_test)
	
	evaluate("Logistic Regression", y_test, y_pred)

def pred_dt(X_train, y_train, X_test, y_test):

	scaler = StandardScaler()
	X_train = scaler.fit_transform(X_train.data)
	X_test = scaler.transform(X_test.data)


	dt = DecisionTreeClassifier()
	dt.fit(X_train, y_train)
	y_pred = dt.predict(X_test)
	
	print("Decision Tree", accuracy_score(y_test, y_pred))

def pred_rf(X_train, y_train, X_test, y_test):

	scaler = StandardScaler()
	X_train = scaler.fit_transform(X_train.data)
	X_test = scaler.transform(X_test.data)

	# Random Forest
	rf = RandomForestClassifier(n_estimators=100)
	rf.fit(X_train, y_train)
	y_pred = rf.predict(X_test)
	
	evaluate("Random Forest", y_test, y_pred)

def pred_ab(X_train, y_train, X_test, y_test):
	scaler = StandardScaler()
	X_train = scaler.fit_transform(X_train.data)
	X_test = scaler.transform(X_test.data)

	# AdaBoost
	ab = AdaBoostClassifier(n_estimators=100, estimator=DecisionTreeClassifier(max_depth=5))
	ab.fit(X_train, y_train)
	y_pred = ab.predict(X_test)

	evaluate("AdaBoost", y_test, y_pred)


if __name__ == "__main__":
	# Load the data
	X_train, y_train = data.load_data("./Data/train", "train")

	X_test, y_test = data.load_data("./Data/test", "test")

	# Standardize the data
	
	# Predict using different classifiers
	pred_lr(X_train, y_train, X_test, y_test)
	pred_dt(X_train, y_train, X_test, y_test)
	pred_rf(X_train, y_train, X_test, y_test)
	pred_ab(X_train, y_train, X_test, y_test)
