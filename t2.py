from align_gender import get_char_diag_list, get_ff_conversations
from collections import Counter
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
import numpy as np
import pickle
from project.src.util import get_data, check_distribution, get_variant_as_key
from project.src.get_scene_boundaries import get_boundaries_agarwal, get_boundaries_gorinski
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

'''PREPARE DATA'''
def get_t2_data(source = 'combined'):
    data = get_data(source)
    data = sorted(data.items(), key=lambda x: x[0])  # sort by id
    X = []
    y = []
    for x in data:
        rating = int(x[1][3])
        if rating >= 1:  # only include those that passed 1
            X.append((x[0], x[1][4], x[1][5]))  # (id, path, char dict)
            label = 1 if rating >= 2 else 0  # check if sample passes T2
            y.append(label)
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

    print(Counter(y_train))
    print(Counter(y_val))
    print(Counter(y_test))

    pickle.dump([X_train, X_val, X_test, y_train, y_val, y_test], open('t2_split.p', 'wb'))
    print('Saved t2_split.p')
    return X_train, X_val, X_test, y_train, y_val, y_test


'''EVAL METHODS'''
def eval(true, pred, verbose = False):
    per_class = f1_score(true, pred, average=None)
    macro = f1_score(true, pred, average='macro')
    report = 'F1s: Per-class: {}. Macro: {}.'.format(per_class, round(macro, 4))
    if verbose:
        report += '\nAccuracy: ' + str(round(accuracy_score(true, pred), 4))
        report += '\n' + str(confusion_matrix(true, pred))
    return report

def eval_rule_based(test = 'all'):
    assert(test == 'all' or test == 'test' or test == 'agarwal')

    if test == 'all':
        X, y = get_t2_data()
    elif test == 'test':
        _, _, X, _, _, y = pickle.load(open('t2_split.p', 'rb'))
    else:
        X, y = get_t2_data(source = 'agarwal')
    X = [(x[1], x[2]) for x in X]

    rb = T2RuleBased()

    print('RULE-BASED: OVERLAP, HARD')
    pred = rb.predict(X, interact='overlap', mode='hard')
    print(eval(y, pred, verbose=True))

    print('\nRULE-BASED: OVERLAP, SOFT')
    pred = rb.predict(X, interact='overlap', mode='soft')
    print(eval(y, pred, verbose=True))

    print('\nRULE-BASED: CONSECUTIVE, HARD')
    pred = rb.predict(X, interact='consecutive', mode='hard')
    print(eval(y, pred, verbose=True))

    print('\nRULE-BASED: CONSECUTIVE, SOFT')
    pred = rb.predict(X, interact='consecutive', mode='soft')
    print(eval(y, pred, verbose=True))

def eval_clf(test = 'all_cv'):
    assert(test == 'all_cv' or test == 'agarwal_cv' or test == 'test' )
    clf = T2Classifier()
    if test == 'all_cv':
        X, y = get_t2_data()
        X = [(x[1], x[2]) for x in X]
        pred = clf.cross_val(X, y)
        print(eval(y, pred, verbose=True))
    elif test == 'agarwal_cv':
        X, y = get_t2_data(source='agarwal')
        X = [(x[1], x[2]) for x in X]
        pred = clf.cross_val(X, y)
        print(eval(y, pred, verbose=True))
    else:  # test on test set
        X_train, X_val, X_test, y_train, y_val, y_test = pickle.load(open('t2_split.p', 'rb'))
        X_train = np.concatenate((X_train, X_val))
        X_train = [(x[1], x[2]) for x in X_train]
        X_test = [(x[1], x[2]) for x in X_test]
        y_train = np.concatenate((y_train, y_val))

        clf.train(X_train, y_train)
        pred = clf.predict(X_test)
        print(eval(y_test, pred, verbose=True))


class T2RuleBased:
    def __init__(self, verbose = False):
        self.verbose = verbose

    def predict(self, X, interact = 'overlap', mode = 'hard'):
        assert(interact == 'overlap' or interact == 'consecutive')
        assert(mode == 'hard' or mode == 'soft')
        if interact == 'overlap':
            pred = []
            for path,char_dict in X:
                pred.append(self.predict_overlap(path, char_dict, mode))
        else:
            pred = []
            for path,char_dict in X:
                pred.append(self.predict_consecutive(path, char_dict, mode))
        return np.array(pred)

    def predict_overlap(self, path, char_dict, mode):
        if 'agarwal' in path:
            source = 'agarwal'
            scenes = get_boundaries_agarwal(path)
        else:
            source = 'gorinski'
            scenes = get_boundaries_gorinski(path)

        var2info = get_variant_as_key(char_dict)

        for scene in scenes:
            cdl = get_char_diag_list(scene, var2info, source)
            if self.overlap_in_scene(cdl, mode):
                return 1
        return 0

    def overlap_in_scene(self, cdl, mode):
        fem_chars = set()
        for (char, gen, score),diag in cdl:
            if mode == 'hard' and gen == 'F':
                fem_chars.add(char)
                if len(fem_chars) == 2:
                    fem_chars = list(fem_chars)
                    if self.verbose:
                        print('Found overlapping females: {} and {}'.format(fem_chars[0], fem_chars[1]))
                    return 1
            elif mode == 'soft' and score != 'None' and float(score) > .5:
                fem_chars.add((char, score))
                if len(fem_chars) == 2:
                    fem_chars = list(fem_chars)
                    if self.verbose:
                        print('Found overlapping females: {} ({}) and {} ({})'.format(fem_chars[0][0], fem_chars[0][1],
                                                                                  fem_chars[1][0], fem_chars[1][1]))
                    return 1
        return 0

    def predict_consecutive(self, path, char_dict, mode):
        if 'agarwal' in path:
            source = 'agarwal'
            scenes = get_boundaries_agarwal(path)
        else:
            source = 'gorinski'
            scenes = get_boundaries_gorinski(path)

        var2info = get_variant_as_key(char_dict)

        for scene in scenes:
            cdl = get_char_diag_list(scene, var2info, source)
            if self.consecutive_in_scene(cdl, mode):
                return 1

        return 0

    def consecutive_in_scene(self, cdl, mode):
        prev_char = ''
        prev_gen = ''
        prev_score = -1
        for (char, gen, score),diag in cdl:
            if mode == 'hard':
                if gen == 'F' and prev_gen == 'F' and prev_char != char:
                    if self.verbose:
                        print('Found consecutive females: {} and {}'.format(prev_char, char))
                    return 1
                prev_char = char
                prev_gen = gen
            else:  # mode == 'soft'
                if score != 'None' and float(score) > .5 and \
                        prev_score != 'None' and float(prev_score) > .5 and prev_char != char:
                    if self.verbose:
                        print('Found consecutive females: {} ({}) and {} ({})'.format(prev_char, prev_score, char, score))
                    return 1
                prev_char = char
                prev_score = score
        return 0


class T2Classifier:
    def __init__(self, verbose = False):
        self.clf = LinearSVC(class_weight={0:.6, 1:.4})
        self.rb = T2RuleBased(verbose=verbose)
        self.trained = False
        self.verbose = verbose

    def transform(self, X):
        feat_mat = np.zeros((len(X), 3), dtype=np.int)
        for i,(path,char_dict) in enumerate(X):
            feats = self.extract_features(path, char_dict)
            feat_mat[i] = feats
        return feat_mat

    def extract_features(self, path, char_dict):
        if self.verbose: print(path)

        if 'agarwal' in path:
            source = 'agarwal'
            scenes = get_boundaries_agarwal(path)
        else:
            source = 'gorinski'
            scenes = get_boundaries_gorinski(path)

        var2info = get_variant_as_key(char_dict)

        feats = np.zeros(3, dtype=np.int)  # counts per rating
        for scene in scenes:
            cdl = get_char_diag_list(scene, var2info, source)
            rating = self.rate_scene(cdl)
            if rating >= 1:
                feats[rating-1] += 1

        return feats

    # rate scene for its potential contribution to T2
    def rate_scene(self, cdl):
        ffs = get_ff_conversations(cdl)
        if len(ffs) > 0:
            for ff in ffs:
                if len(ff) > 4:  # has long ff
                    return 3
            return 2  # has consecutive soft
        check = self.rb.overlap_in_scene(cdl, 'soft')
        if check == 1:
            return 1  # has overlap soft
        return 0  # failed

    def train(self, X, y, oversample = False):
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
        print('Distribution:', Counter(y))
        X = self.transform(X)
        pred = cross_val_predict(self.clf, X, y, cv=n)
        return pred


if __name__ == "__main__":
    # X, y = get_t2_data()
    # X_train, X_val, X_test, y_train, y_val, y_test = split_and_save(X, y)

    # for test_type in ['all', 'agarwal']:
    #     print('\nEvaluating on', test_type.upper(), 'data...')
    #     eval_rule_based(test=test_type)

    for test_type in ['all_cv', 'agarwal_cv']:
        print('\nEvaluating on', test_type.upper(), 'data...')
        eval_clf(test=test_type)