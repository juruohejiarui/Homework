from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import RidgeClassifier, LogisticRegression

from predict import evaluate
from time import time
import matplotlib.pyplot as plt
import data

fig = plt.figure(figsize=(10, 10))

def logistic(X_train, y_train, X_test, y_test) -> tuple[LogisticRegression, float] :
	search = GridSearchCV(
		LogisticRegression(random_state=114),
		param_grid={
			'C': [11 + 0.1 * i for i in range(-10, 10)],
			'solver': ['liblinear'],
			'max_iter': [1000]
		},
		cv=5,
		n_jobs=-1,
		scoring='f1_macro'
	)
	
	start_time = time()
	search.fit(X_train, y_train)
	end_time = time()
	time_cost = end_time - start_time
	
	bst = search.best_estimator_
	print("Best parameters:", search.best_params_)
	evaluate("Logistic Regression", bst, X_test, y_test, time_cost, eval_roc=False)
	
	return bst, time_cost

if __name__ == "__main__":
	X_train, y_train = data.load_data("./data/train", "train")
	X_test, y_test = data.load_data("./data/test", "test")
	# ridge(X_train, y_train, X_test, y_test)
	logistic(X_train, y_train, X_test, y_test)
	
	plt.show()
	# plt.savefig("grid_search.png")