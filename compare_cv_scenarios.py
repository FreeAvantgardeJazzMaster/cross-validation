import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel, SelectKBest


def create_random_data(n_instances, n_features):
    X = 2*np.random.rand(n_instances, n_features) - 1
    scores =  2*np.random.rand(n_instances) - 1
    where_positive = scores > 0
    where_negative = scores <= 0
    scores[where_positive] = 1
    scores[where_negative] = 0
    y = np.array(scores.ravel().tolist(), dtype=np.int32)
    return X, y

def print_evaluation(y, predicted):
    #print("True: {0}/{1} = {2}".format(sum(y), len(y), float(sum(y))/len(y)))
    #print("Confusion matrix")
    cm = confusion_matrix(y, predicted)
    #print(cm)
    correct = cm[0, 0] + cm[1, 1]
    incorrect = cm[1, 0] + cm [0, 1]
    return float(correct)/(correct + incorrect)

def selection_outside_cv(X, y, folds, sfm = SelectFromModel(LogisticRegression())):
    #print("Scenario: Selection + CV(LR)")
    sfm.fit(X, y)
    X_selected = sfm.transform(X)

    predicted = cross_val_predict(LogisticRegression(), X_selected, y, cv=folds)
    return print_evaluation(y, predicted)

def selection_inside_cv(X, y, folds, sfm = SelectFromModel(LogisticRegression())):
    #print("Scenario: CV(Selection, LR)")
    clf = Pipeline([
        ('feature_selection', sfm),
        ('classification', LogisticRegression())
    ])
    predicted = cross_val_predict(clf, X, y, cv=folds)
    return print_evaluation(y, predicted)


def main():
    n_instances = 1000
    n_features = 1000
    folds = 10
    n_top_features = 2

    X, y = create_random_data(n_instances, n_features)

    #use one of the methods for feature selection
    #sfm = SelectFromModel(LogisticRegression())
    sfm = SelectKBest(k=n_top_features)
    outside_list = []
    inside_list = []
    for i in range(1, 10):
        outside_list.append(selection_outside_cv(X, y, folds, sfm))
        inside_list.append(selection_inside_cv(X, y, folds, sfm))

    print("Outside selection average: " + str(sum(outside_list)/len(outside_list)))
    print("Inside selection average: " + str(sum(inside_list)/len(inside_list)))


if __name__ == "__main__":
    main()

