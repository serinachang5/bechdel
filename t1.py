from collections import Counter
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
import numpy as np
import pickle
from project.src.util import get_data, check_distribution
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

def get_t1_data():
    combined_data = get_data()
    combined_data = sorted(combined_data.items(), key=lambda x: x[0])  # sort by id
    X = np.array([(x[0], x[1][5]) for x in combined_data])  # (id, char dict)
    y = np.array([1 if int(x[1][3]) >= 1 else 0 for x in combined_data], dtype=np.int)  # check if sample passes T1
    return X, y

def split_and_save(X, y, train_prop = .6, val_prop = .2, test_prop = .2):
    assert((train_prop + val_prop + test_prop) == 1)

    shuffle_indices = list(range(X.shape[0]))
    np.random.shuffle(shuffle_indices)
    X = X[shuffle_indices]
    y = y[shuffle_indices]

    train_cutoff = int(X.shape[0] * train_prop)
    X_train = X[:train_cutoff]
    y_train = y[:train_cutoff]

    val_cutoff = train_cutoff + int(X.shape[0] * val_prop)
    X_val = X[train_cutoff:val_cutoff]
    y_val = y[train_cutoff:val_cutoff]

    X_test = X[val_cutoff:]
    y_test = y[val_cutoff:]

    pickle.dump([X_train, X_val, X_test, y_train, y_val, y_test], open('t1_split.p', 'wb'))
    print('Saved t1_split.p')
    return X_train, X_val, X_test, y_train, y_val, y_test

def eval(true, pred, verbose = False):
    per_class = f1_score(true, pred, average=None)
    macro = f1_score(true, pred, average='macro')
    report = 'F1s: Per-class: {}. Macro: {}.'.format(per_class, round(macro, 4))
    if verbose:
        report += '\nAccuracy: ' + str(round(accuracy_score(true, pred), 4))
        report += '\n' + str(confusion_matrix(true, pred))
    return report

class RuleBased:
    def predict(self, X, mode = 'hard'):
        assert(mode == 'hard' or mode == 'soft')
        if mode == 'hard':
            pred = []
            for x in X:
                pred.append(self.predict_hard(x))
        else:
            pred = []
            for x in X:
                pred.append(self.predict_soft(x))
        return np.array(pred)

    def predict_hard(self, char_dict):
        fem_count = 0
        for root, (gen, score, variants) in char_dict.items():
            if gen == 'F':
                fem_count += 1
            if fem_count == 2:
                return 1
        return 0

    def predict_soft(self, char_dict):
        fem_count = 0
        for root, (gen, score, variants) in char_dict.items():
            if score != 'None' and float(score) > .5:
                fem_count += 1
            if fem_count == 2:
                return 1
        return 0

class T1Classifier:
    def __init__(self):
        self.clf = LinearSVC()
        self.trained = False

    def get_features(self, X):
        X_as_features = np.zeros((len(X), 6), dtype=np.int)
        for i,char_dict in enumerate(X):
            feats = np.zeros(6, dtype=np.int)
            for root, (gen, score, variants) in char_dict.items():
                if score != 'None':
                    score = float(score)
                    if score >= .5:
                        bucket = int(score/.1) - 5
                        feats[bucket] += 1
            X_as_features[i] = feats

        X = X_as_features
        return X

    def train(self, X, y, oversample = True):
        X = self.get_features(X)
        if oversample:
            oversampler = SMOTE()
            X, y = oversampler.fit_sample(X, y)
        self.clf.fit(X, y)
        self.trained = True

    def predict(self, X):
        if not self.trained:
            print('Should not predict before training.')
            return None
        X = self.get_features(X)
        return self.clf.predict(X)

if __name__ == "__main__":
    # X, y = get_t1_data()
    # X_train, X_val, X_test, y_train, y_val, y_test = split_and_save(X, y)
    #
    # print(Counter(y_train))
    # print(Counter(y_val))
    # print(Counter(y_test))

    X_train, X_val, X_test, y_train, y_val, y_test = pickle.load(open('t1_split.p', 'rb'))
    X_train = [x[1] for x in X_train]
    X_val = [x[1] for x in X_val]
    X_test = [x[1] for x in X_test]

    rb = RuleBased()

    print('RULE-BASED: HARD')
    pred = rb.predict(np.concatenate((X_train, X_val, X_test)), mode='hard')
    print(eval(np.concatenate((y_train, y_val, y_test)), pred, verbose=True))
    pred = rb.predict(X_test, mode='hard')
    print(eval(y_test, pred, verbose=True))

    print('\nRULE-BASED: SOFT')
    pred = rb.predict(np.concatenate((X_train, X_val, X_test)), mode='soft')
    print(eval(np.concatenate((y_train, y_val, y_test)), pred, verbose=True))
    pred = rb.predict(X_test, mode='hard')
    print(eval(y_test, pred, verbose=True))

    clf = T1Classifier()

    print('\nCLF: WITHOUT OVERSAMPLING')
    clf.train(np.concatenate((X_train, X_val)), np.concatenate((y_train, y_val)), oversample=False)
    pred = clf.predict(X_test)
    print(eval(y_test, pred, verbose=True))

    print('\nCLF: WITH OVERSAMPLING')
    clf.train(np.concatenate((X_train, X_val)), np.concatenate((y_train, y_val)), oversample=True)
    pred = clf.predict(X_test)
    print(eval(y_test, pred, verbose=True))