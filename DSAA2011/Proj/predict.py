import data
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from time import time

import matplotlib.pyplot as plt
import visualize

best_model = None
best_accuracy = 0

fig = plt.figure(figsize=(10, 10))
nrow, ncol = 3, 2
idx = 0

def evaluate(name, 
			 model : LogisticRegression | DecisionTreeClassifier | RandomForestClassifier | AdaBoostClassifier | GradientBoostingClassifier,
			 X_test, y_test, time_cost=None, eval_roc : bool = False) :
	global best_model, best_accuracy, ncol, nrow, idx
	print(f"metric of {name} :")

	y_pred = model.predict(X_test)

	print(f"time cost : {time_cost:.2f} s")
	acc = accuracy_score(y_test, y_pred)
	print(f"accuracy : {acc}")
	# update best model
	if best_model is None or best_accuracy < acc :
		best_model = model
		best_accuracy = acc

	print(classification_report(y_test, y_pred))

	print(f"confusion matrix :\n{confusion_matrix(y_test, y_pred)}")

	# draw AUC and ROC curve
	if eval_roc :
		classes_dict = data.load_label_name("./Data")
		classes = list(classes_dict.keys())

		y_test_bin = data.label_binarize(y_test)
		y_score = model.predict_proba(X_test)

		fpr, tpr, roc_auc = {}, {}, {}
		for i, cls in enumerate(classes) :
			fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
			roc_auc[i] = auc(fpr[i], tpr[i])

		# draw ROC curve
		ax = fig.add_subplot(nrow, ncol, idx + 1)
		ax.plot([0, 1], [0, 1], 'k--')
		ax.set_xlim([0.0, 1.0])
		ax.set_ylim([0.0, 1.05])
		ax.set_xlabel('False Positive Rate')
		ax.set_ylabel('True Positive Rate')
		ax.set_title(name)
		for i, cls in enumerate(classes) :
			ax.plot(fpr[i], tpr[i], label=f"{classes_dict[cls]} (AUC {roc_auc[i]:.2f})", color=visualize.get_label_color(cls))
		ax.legend(loc='lower right')

		idx += 1

	print()
def pred_logistic(X_train, y_train, X_test, y_test):
	st_time = time()
	# Logistic Regression
	lr = LogisticRegression(C=1e9, max_iter=1000)
	lr.fit(X_train, y_train)
	time_cost = time() - st_time
	
	evaluate("Logistic Regression", lr, X_test, y_test, time_cost, eval_roc=True)

def pred_ridge(X_train, y_train, X_test, y_test):
	st_time = time()
	# Linear Regression
	lr = RidgeClassifier()
	lr.fit(X_train, y_train)
	time_cost = time() - st_time
	
	evaluate("Ridge Regression", lr, X_test, y_test, time_cost)

def pred_dt(X_train, y_train, X_test, y_test):
	st_time = time()

	scaler = StandardScaler()
	X_train = scaler.fit_transform(X_train.data)
	X_test = scaler.transform(X_test.data)
	dt = DecisionTreeClassifier()
	dt.fit(X_train, y_train)
	time_cost = time() - st_time
	
	evaluate("Decision Tree", dt, X_test, y_test, time_cost, eval_roc=True)

def pred_rf(X_train, y_train, X_test, y_test):
	st_time = time()
	scaler = StandardScaler()
	X_train = scaler.fit_transform(X_train.data)
	X_test = scaler.transform(X_test.data)

	# Random Forest
	rf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
	rf.fit(X_train, y_train)
	time_cost = time() - st_time
	
	evaluate("Random Forest", rf, X_test, y_test, time_cost, eval_roc=True)

def pred_ab(X_train, y_train, X_test, y_test):
	st_time = time()
	scaler = StandardScaler()
	X_train = scaler.fit_transform(X_train.data)
	X_test = scaler.transform(X_test.data)

	# AdaBoost
	ab = AdaBoostClassifier(n_estimators=100, estimator=DecisionTreeClassifier(max_depth=3))
	ab.fit(X_train, y_train)
	time_cost = time() - st_time	

	evaluate("AdaBoost", ab, X_test, y_test, time_cost, eval_roc=True)

def pred_svc(X_train, y_train, X_test, y_test):
	st_time = time()
	scaler = StandardScaler()
	X_train = scaler.fit_transform(X_train.data)
	X_test = scaler.transform(X_test.data)

	# SVC
	svc = SVC(probability=True)
	svc.fit(X_train, y_train)
	time_cost = time() - st_time
	
	evaluate("SVC", svc, X_test, y_test, time_cost, eval_roc=True)

# finetune the logistic regression
def finetune_logistic(X_train, y_train, X_test, y_test):
	lr_l1 = LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000)
	lr_l2 = LogisticRegression(penalty='l2', solver='liblinear', max_iter=1000)
	lr_l2_2 = LogisticRegression(penalty='l2', solver='saga', max_iter=1000)

	st_time = time()
	lr_l1.fit(X_train, y_train)
	time_cost1 = time() - st_time
	st_time = time()
	lr_l2.fit(X_train, y_train)
	time_cost2 = time() - st_time

	st_time = time()
	lr_l2_2.fit(X_train, y_train)
	time_cost2_2 = time() - st_time

	evaluate("Logistic Regression L1", lr_l1, X_test, y_test, time_cost1, eval_roc=False)
	evaluate("Logistic Regression L2", lr_l2, X_test, y_test, time_cost2, eval_roc=True)
	evaluate("Logistic Regression L2 (saga)", lr_l2_2, X_test, y_test, time_cost2_2, eval_roc=False)

if __name__ == "__main__":
	# Load the data
	X_train, y_train = data.load_data("./Data/train", "train")

	X_test, y_test = data.load_data("./Data/test", "test")

	# Standardize the data
	
	# Predict using different classifiers
	pred_logistic(X_train, y_train, X_test, y_test)
	pred_ridge(X_train, y_train, X_test, y_test)
	pred_svc(X_train, y_train, X_test, y_test)
	pred_dt(X_train, y_train, X_test, y_test)
	pred_rf(X_train, y_train, X_test, y_test)
	pred_ab(X_train, y_train, X_test, y_test)

	print(f"best model : {best_model}")
	print(f"best accuracy : {best_accuracy}")

	print(f"finetune logistic regression")
	finetune_logistic(X_train, y_train, X_test, y_test)

	plt.tight_layout()
	plt.savefig("./roc.png", dpi=300)
