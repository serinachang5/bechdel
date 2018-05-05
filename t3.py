import align_gender as ag
from get_scene_boundaries import get_boundaries_agarwal, get_boundaries_gorinski
from t2 import T2RuleBased, T2Classifier
from util import get_data, check_distribution
import social_network_analysis as sna

from collections import Counter
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

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
        var2info = ag.get_variant_as_key(char_dict)

        for scene in scenes:
            cdl = ag.get_char_diag_list(scene, var2info, source)
            ffs = ag.get_ff_conversations(cdl)
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
    def __init__(self, only_ff = True, unigrams_count = 500, verbose = False):
        self.clf = LinearSVC(class_weight={0:.6, 1:.4})
        self.sna = sna.SNA()
        self.trained = False
        self.only_ff = only_ff
        self.uni_count = unigrams_count
        self.verbose = verbose

    def transform(self, X):
        diag_per_movie = []  # diag per movie
        print('Transforming', len(X), 'samples')
        # print('Building unigrams model...')
        # # make corpus to train unigrams model - either all female dialogue or all fem-fem dialogue
        # for i,(id, path,char_dict) in enumerate(X):
        #     this_diag = ''
        #     if i % 50 == 0: print(i)
        #     if 'agarwal' in path:
        #         source = 'agarwal'
        #         scenes = get_boundaries_agarwal(path)
        #     else:
        #         source = 'gorinski'
        #         scenes = get_boundaries_gorinski(path)
        #     var2info = ag.get_variant_as_key(char_dict)
        #     for scene in scenes:
        #         cdl = ag.get_char_diag_list(scene, var2info, source)
        #         if self.only_ff:
        #             ffs = ag.get_ff_conversations(cdl)
        #             for ff in ffs:
        #                 for char,line in ff:
        #                     this_diag += line
        #         else:
        #             for (char,gen,score),diag in cdl:  # for each character/line
        #                 if score != 'None' and float(score) > .5:
        #                     line = ' '.join(diag)
        #                     if len(line) > 0:
        #                         this_diag += ' ' + line
        #     diag_per_movie.append(this_diag)
        # unigrams = CountVectorizer(max_features=self.uni_count).fit_transform(diag_per_movie)
        # print(unigrams.shape)
        #
        sna_mode = 'consecutive'
        min_lines = 5
        centralities = ['degree', 'btwn', 'close', 'eigen']
        sn_feats = []
        for i,(id, path, char_dict) in enumerate(X):
            if i % 50 == 0: print(i)
            sn_feats.append(self.sna.transform_into_feats(id, sna_mode, min_lines, centralities))
        sn_feats = np.array(sn_feats)
        print(sn_feats.shape)
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

    X,y = get_t3_data()
    clf = T3Classifier(only_ff=False)
    pred = clf.cross_val(X,y,n=5)
    print(eval(pred, y, verbose=True))

