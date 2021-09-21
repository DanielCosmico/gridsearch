from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_digits

digits = load_digits()
mlp = MLPClassifier(max_iter=10)

parameters = {
    'hidden_layer_sizes': [(50,50,50), (64,100,50), (100,)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.1, 0.05],
    'learning_rate': ['constant','adaptive'],
}

from sklearn.model_selection import GridSearchCV

clf = GridSearchCV(mlp, parameters, n_jobs=-1, cv=4)
clf.fit(digits.data, digits.target)

# Best paramete set
print('Best parameters found:\n', clf.bestparams)
print()

# All results
means = clf.cvresults['mean_test_score']
stds = clf.cvresults['std_test_score']
for mean, std, params in zip(means, stds, clf.cvresults['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
