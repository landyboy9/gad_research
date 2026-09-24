from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

def split_test_train(X, y, p, n, rng):
    '''
    p: number of parameters
    n: number of datapoints
    X: overall dataset
    y: overall labels                    '''

    # randomly select p columns
    selected_cols = rng.choice(X.shape[1], size=p, replace=False)
    X_selected = X[:, selected_cols]

    # keep trying until training set contains both classes
    while True:
        # randomly shuffle row indices
        shuffled_idx = rng.permutation(X.shape[0])
        train_idx = shuffled_idx[:n]
        test_idx = shuffled_idx[n:]

        y_train = y[train_idx]

        # Check that both classes are represented
        if len(np.unique(y_train)) == 2:
            break

    X_train = X_selected[train_idx]
    X_test = X_selected[test_idx]
    y_test = y[test_idx]

    return X_train, y_train, X_test, y_test, selected_cols



def run_logit(X_train,y_train,X_test,y_test):

    # fit logistic regression
    model = LogisticRegression(max_iter=1000)
    
    model.fit(X_train, y_train)

    # errors (1 - accuracy)
    train_error = 1 - accuracy_score(y_train, model.predict(X_train))
    test_error = 1 - accuracy_score(y_test, model.predict(X_test))

    return {
        "train_error": train_error,
        "test_error": test_error,
        "model": model}