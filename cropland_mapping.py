from random_forest import RandomForest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, precision_recall_fscore_support

sns.set()
start_time = datetime.now()


# Importing the dataset
dataset = pd.read_csv('../WinnipegDataset.txt')
dataset = dataset.sample(frac=1)
# dataset = data.iloc[:100000, :]


print(dataset.head(5))
print("Dataset shape: ", dataset.shape)

class_share = pd.Series(100 * dataset.label.value_counts(normalize=True)).sort_index()
print(class_share.shape)
print('\nClass share\n')
for i in range(0,7):
    print(f'Class {class_share.index[i]}: {class_share.iloc[i]:.2f} %')

fig, ax = plt.subplots(figsize=(8,5))
sns.barplot(ax=ax, x = class_share.index, y = class_share, palette='Greens_d')
plt.title('Cropland Class Share', fontsize=18)
plt.xlabel('Cropland Class', fontsize=14)
plt.ylabel('Share (%)', fontsize=14)
plt.show(block=False)

highly_correlated_feature_pairs = []
highly_correlated_features_to_drop = []

correlation_matrix = dataset.corr().abs()
upper_matrix = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
for row in range(upper_matrix.shape[0]):
    for column in range(upper_matrix.shape[1]):
        if upper_matrix.iloc[row, column] > 0.95:
            highly_correlated_feature_pairs.append([row, column, upper_matrix.iloc[row, column]])

print(f'Number of highly intercorrelated feature pairs: {len(highly_correlated_feature_pairs)}')

highly_correlated_feature_pairs = pd.DataFrame(highly_correlated_feature_pairs).sort_values(by=[2], ascending=False)
highly_correlated_feature_pairs.columns = ['feature_1', 'feature_2', 'correl']
highly_correlated_feature_pairs.head(10)

highly_correlated_features_to_drop = [column for column in upper_matrix.columns if any(upper_matrix[column] > 0.95)]
dataset = dataset.drop(dataset[highly_correlated_features_to_drop], axis=1)

nr_features = dataset.shape[1] - 1
print(f'Optimized number of features: {nr_features}')

X = dataset.iloc[:, 1:].values
y = dataset.iloc[:, 0:1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

clf = RandomForest(n_trees=100, feature_selection='sqrt')

clf.fit(X_train, y_train)
y_pred = clf.predict(X_train)
print('Train: ', accuracy_score(y_train, y_pred))

y_pred = clf.predict(X_test)
print('Test: ', accuracy_score(y_test, y_pred))

rf_cm = np.zeros((7,7))

rf_pred_correct = 0
rf_pred_incorrect = 0
for i in range(len(y_test)):
    rf_index_test = int(y_test[i] - 1)
    rf_index_pred = int(y_pred[i] - 1)
    rf_cm[rf_index_test][rf_index_pred] += 1
    if rf_index_test == rf_index_pred:
        rf_pred_correct += 1
    else:
        rf_pred_incorrect += 1

rf_cmatrix = pd.DataFrame(rf_cm.astype(int),
                          index = ['Corn', 'Pea', 'Canola', 'Soy', 'Oat', 'Wheat', 'Broadleaf'],
                          columns = ['Corn', 'Pea', 'Canola', 'Soy', 'Oat', 'Wheat', 'Broadleaf'])
rf_cmatrix

rf_accuracy = accuracy_score(y_test, y_pred)
rf_precision, rf_recall, rf_f_score, rf_support = precision_recall_fscore_support(y_test, y_pred, average='macro')
print(f'Accuracy: {rf_accuracy * 100:.2f} %')
print(f'Precision: {rf_precision * 100:.2f} %')
print(f'Recall: {rf_recall * 100:.2f} %')
print(f'F-Score: {rf_f_score * 100:.2f} %')

print('Accuracy per class\n')
for i in range(len(rf_cmatrix)):
    class_accuracy = rf_cmatrix.iloc[i,i] / rf_cmatrix.sum(axis=0)[i]
    print(f'{rf_cmatrix.columns[i]}: {class_accuracy*100:.2f} %')

plt.show()