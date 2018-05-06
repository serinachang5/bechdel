import util as ut
import align_gender as ag
from get_scene_boundaries import get_boundaries_agarwal, get_boundaries_gorinski

from collections import Counter
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


def get_t1_data(test):
    assert(test == 'train' or test == 'agarwal')
    if test == 'train':
        X, _, y, _ = pickle.load(open('train_test.pkl', 'rb'))
        X = [(x[1], x[2]) for x in X]  # path, char_dict (don't need id)
    else:
        agarwal_data = ut.get_data(source='agarwal')
        data = list(agarwal_data.items())
        X = np.array([(x[1][4], x[1][5]) for x in data])  # path, char dict
        y = np.array([int(x[1][3]) for x in data])  # Bechdel label

    y = np.array([1 if label >= 1 else 0 for label in y])  # check if sample passes T1
    return X, y


'''EVAL METHODS'''
def eval(true, pred, verbose = False):
    per_class = f1_score(true, pred, average=None)
    macro = f1_score(true, pred, average='macro')
    report = 'F1s: Per-class: {}. Macro: {}.'.format(per_class, round(macro, 4))
    if verbose:
        report += '\nAccuracy: ' + str(round(accuracy_score(true, pred), 4))
        report += '\n' + str(confusion_matrix(true, pred))
    return report

def eval_rule_based(test):
    X, y = get_t1_data(test)
    rb = T1RuleBased()

    print('RULE-BASED: HARD')
    pred = rb.predict(X, mode='hard')
    print(eval(y, pred, verbose=True))

    print('\nRULE-BASED: SOFT')
    pred = rb.predict(X, mode='soft')
    print(eval(y, pred, verbose=True))

def eval_clf(test):
    X, y = get_t1_data(test)
    clf = T1Classifier()
    pred = clf.cross_val(X, y)
    print(eval(y, pred, verbose=True))


class T1RuleBased:
    def predict(self, X, mode = 'hard'):
        assert(mode == 'hard' or mode == 'soft')
        if mode == 'hard':
            pred = []
            for path, char_dict in X:
                pred.append(self.predict_hard(char_dict))
        else:
            pred = []
            for path, char_dict in X:
                pred.append(self.predict_soft(char_dict))
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

    def transform(self, X):
        feat_mat = np.zeros((len(X), 2), dtype=np.int)
        for i,(path,char_dict) in enumerate(X):
            feats = self.extract_features(path, char_dict)
            feat_mat[i] = feats
        return feat_mat

    def extract_features(self, path, char_dict):
        char_to_lines = ut.get_char_to_lines(path=path, char_dict=char_dict)
        # count num of fems with > 1 lines, num of fems with 1 line
        feats = np.zeros(2, dtype=np.int)
        for char, lines in char_to_lines.items():
            gen, score, var = char_dict[char]
            if score != 'None' and float(score) > .5:
                if len(lines) > 1:
                    feats[0] += 1
                else:
                    feats[1] += 1
        return feats

    def train(self, X, y, oversample = True):
        X = self.transform(X)
        if oversample:
            oversampler = SMOTE()
            X, y = oversampler.fit_sample(X, y)
        self.clf.fit(X, y)
        self.trained = True

    def predict(self, X):
        if not self.trained:
            print('Should not predict before training.')
            return None
        X = self.transform(X)
        return self.clf.predict(X)

    def cross_val(self, X, y, n = 5):
        # print('Distribution:', Counter(y))
        X = self.transform(X)
        pred = cross_val_predict(self.clf, X, y, cv=n)
        return pred


if __name__ == "__main__":
    # for test_type in ['train', 'agarwal']:
    #     print('\nEvaluating on', test_type.upper(), 'data...')
    #     eval_rule_based(test=test_type)

    for test_type in ['train', 'agarwal']:
        print('\nEvaluating on', test_type.upper(), 'data...')
        eval_clf(test=test_type)