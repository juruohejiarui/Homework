import data
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

if __name__ == "__main__":
	# Load the data
	X_train, y_train = data.load_data("./Data/train", "train")

	X_test, y_test = data.load_data("./Data/test", "test")

	lr = LogisticRegression(C=1e9, max_iter=1000)

	scaler = StandardScaler()
	# X_train = scaler.fit_transform(X_train)
	# X_test = scaler.transform(X_test)

	lr.fit(X_train, y_train)

	y_pred = lr.predict(X_test)

	print("Accuracy:", accuracy_score(y_test, y_pred))
	print(classification_report(y_test, y_pred))

	

	
