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
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from t2 import T2RuleBased, T2Classifier

'''PREPARE DATA'''
def get_t3_data(source = 'combined'):
    data = get_data(source)
    data = sorted(data.items(), key=lambda x: x[0])  # sort by id
    X = []
    y = []
    for x in data:
        rating = int(x[1][3])
        if rating >= 2:  # only include those that passed 2
            X.append((x[0], x[1][4], x[1][5]))  # (id, path, char dict)
            label = 1 if rating == 3 else 0  # check if sample passes T3
            y.append(label)
    return X, y


def get_char_diag_list(scene, var2info, source):
    char_diag_list = []

    if source == 'agarwal':
        idx = 0
        while idx < len(scene):
            if scene[idx].startswith('C|'):
                # process variant
                var = scene[idx].split(None, 1)[1]  # cut off C| + white space
                if var.startswith('('):  # probably a description e.g. '(QUIETLY)'
                    idx += 1
                    continue
                var = var.strip()  # strip trailing white space
                if var in var2info:
                    curr_char = var2info[var]  # root, gen, score
                    idx += 1
                    diag = []
                    while idx < len(scene) and scene[idx].startswith('D|'):
                        line = scene[idx].split(None, 1)[1]  # cut off D| + white space
                        line = line.strip()
                        diag.append(line)
                        idx += 1
                    char_diag_list.append((curr_char, diag))
                else:
                    idx += 1
            else:
                idx += 1

    else:  # source == 'gorinski'
        idx = 0
        while idx < len(scene):
            possible = scene[idx].strip()
            if possible.startswith('('):
                idx += 1
                continue
            if possible in var2info:
                curr_char = var2info[possible]
                idx += 1
                if idx < len(scene):
                    diag = []
                    diag_ls = len(scene[idx]) - len(scene[idx].lstrip(' '))
                    while idx < len(scene) and diag_ls == len(scene[idx]) - len(scene[idx].lstrip(' ')):
                        diag.append(scene[idx].strip())
                        idx += 1
                    char_diag_list.append((curr_char, diag))
            else:
                idx += 1

    return char_diag_list


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
    assert(test == 'all' or test == 'agarwal')
    if test == 'all':
        X, y = get_t3_data()
    else:
        X, y = get_t3_data(source = 'agarwal')

    X = [(x[1], x[2]) for x in X]
    rb = T3RuleBased()
    pred = rb.predict(X)
    print(eval(y, pred, verbose=True))

def eval_clf(test = 'all_cv'):
    assert(test == 'all_cv' or test == 'agarwal_cv' or test == 'test' )
    clf = T2Classifier()
    if test == 'all_cv':
        X, y = get_t3_data()
        X = [(x[1], x[2]) for x in X]
        pred = clf.cross_val(X, y)
        print(eval(y, pred, verbose=True))
    elif test == 'agarwal_cv':
        X, y = get_t3_data(source='agarwal')
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


class T3RuleBased:
    def __init__(self, verbose = False):
        self.verbose = verbose

    def predict(self, X):
        pred = []
        for path,char_dict in X:
            pred.append(self._predict(path, char_dict))
        return np.array(pred)

    def _predict(self, path, char_dict):
        if 'agarwal' in path:
            source = 'agarwal'
            scenes = get_boundaries_agarwal(path)
        else:
            source = 'gorinski'
            scenes = get_boundaries_gorinski(path)

        male_chars = self.get_male_chars(char_dict)  # soft mode
        var2info = get_variant_as_key(char_dict)

        for scene in scenes:
            cdl = get_char_diag_list(scene, var2info, source)
            ffs = self.get_ff_conversations(cdl)
            # len(ffs) > 0 means it passes consecutive soft
            for ff in ffs:
                if self.no_man_conversation(ff, male_chars):
                    return 1
        return 0

    def get_male_chars(self, char_dict):
        male_chars = set()
        for char,(gen, score, variants) in char_dict.items():
            if score != 'None' and float(score) < .5:
                male_chars.add(char)
        return male_chars



    def no_man_conversation(self, ff, male_chars):
        male_pronouns = ['he','him','his']
        for char,line in ff:
            line = line.lower()
            for mc in male_chars:
                if mc in line:
                    return False
            for mp in male_pronouns:
                if mp in line:
                    return False
        return True


class T3Classifier:
    def __init__(self, verbose = False):
        self.clf = LinearSVC()
        self.trained = False
        self.verbose = verbose

    def transform(self, X):
        diag_per_movie = []  # diag per movie
        print('Transforming', len(X), 'samples')
        for i,(path,char_dict) in enumerate(X):  # for each movie
            this_diag = ''
            if i % 50 == 0: print(i)
            if 'agarwal' in path:
                source = 'agarwal'
                scenes = get_boundaries_agarwal(path)
            else:
                source = 'gorinski'
                scenes = get_boundaries_gorinski(path)
            var2info = get_variant_as_key(char_dict)
            for scene in scenes:  # for each scene
                cdl = get_char_diag_list(scene, var2info, source)
                for (char,gen,score),diag in cdl:  # for each character/line
                    if score != 'None' and float(score) > .5:
                        line = ' '.join(diag)
                        if len(line) > 0:
                            this_diag += ' ' + line
            diag_per_movie.append(this_diag)

        X = CountVectorizer(max_features=1000).fit_transform(diag_per_movie)
        print(X.shape)
        return X

    def train(self, X, y):
        X = self.transform(X)
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
    # eval_rule_based(test='all')

    X,y = get_t3_data(source='combined')
    print(Counter(y))
    X = [(x[1], x[2]) for x in X]
    clf = T2Classifier()
    pred = clf.cross_val(X,y,n=5)
    print(eval(pred, y, verbose=True))
