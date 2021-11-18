import numpy as np
from decision_tree import DecisionTreeClassifier


def bootstrap_sample(X, y, feature_selection):
    n_samples = X.shape[0]
    n_features = X.shape[1]
    indices = np.random.choice(n_samples, size=n_samples, replace=True)
    if feature_selection:
        if feature_selection == 'sqrt':
            ft_size = round(np.sqrt(n_features))
        elif feature_selection == 'log2':
            ft_size = round(np.log2(n_features))
        features = np.random.choice(n_features, size=ft_size, replace=False)
        return X[np.ix_(indices, features)], y[indices]
    return X[indices], y[indices]


def calculate_leaf_value(y):
    ''' function to compute leaf node '''

    y = list(y)
    return max(y, key=y.count)


class RandomForest:

    def __init__(self, n_trees=100, min_samples_split=2, max_depth=100, feature_selection='sqrt'):
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.feature_selection = feature_selection
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTreeClassifier(min_samples_split=self.min_samples_split,
                                          max_depth=self.max_depth)
            X_sample, y_sample = bootstrap_sample(X, y, feature_selection=self.feature_selection)
            # X_sample, y_sample = X, y
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        y_pred = [calculate_leaf_value(tree_pred) for tree_pred in tree_preds]
        return np.array(y_pred)


# Testing
if __name__ == "__main__":
    # Imports
    import pandas as pd
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    csv = datasets.load_breast_cancer()
    data = pd.DataFrame(data=np.c_[csv['data'], csv['target']],
                        columns=np.append(csv['feature_names'], ['target']))
    print(data.head(5))

    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values.reshape(-1, 1)
    # X = data.data
    # y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

    # print(X_train[:5])
    # print(y_train[:5])
    # testx, testy = bootstrap_sample(X_train[:5], y_train[:5])
    # print(testx)
    # print(testy)

    clf = RandomForest(n_trees=100, max_depth=10, feature_selection='sqrt')

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_train)
    print('Train: ', accuracy_score(y_train, y_pred))

    y_pred = clf.predict(X_test)
    print('Test: ', accuracy_score(y_test, y_pred))
