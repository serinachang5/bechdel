import align_gender as ag
from get_scene_boundaries import get_boundaries_agarwal, get_boundaries_gorinski
from t2 import T2RuleBased, T2Classifier
import util as ut
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
    data = ut.get_data(source)
    data = sorted(data.items(), key=lambda x: x[0])  # sort by id
    X = []
    y = []
    for x in data:
        rating = int(x[1][3])
        if rating >= 2:  # only include those that passed 2
            X.append((x[0], x[1][4], x[1][5]))  # (id, path, char dict)
            label = 1 if rating == 3 else 0  # check if sample passes T3
            y.append(label)
    return np.array(X), np.array(y)

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
        X, _, y, _ = pickle.load(open('t3_split.pkl', 'rb'))
    else:
        X, y = get_t3_data(source = 'agarwal')

    X = [(x[1], x[2]) for x in X]

    rb = T3RuleBased()
    pred = rb.predict(X)
    print(eval(y, pred, verbose=True))

def eval_clf(test = 'all_cv', **kwargs):
    assert(test == 'all_cv' or test == 'agarwal_cv')
    if test == 'all_cv':
        X, _, y, _ = pickle.load(open('t3_split.pkl', 'rb'))
    else:
        X, y = get_t3_data(source = 'agarwal')

    clf = T3Classifier(**kwargs)

    pred = clf.cross_val(X, y)
    print(eval(y, pred, verbose=True))


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
    def __init__(self, feats = None, uni_only_ff = True, uni_count = 500, sna_mode = 'consecutive', sna_min_lines = 5, sna_centralities = None, verbose = False):
        self.clf = LinearSVC(class_weight={0:.7, 1:.3})
        self.feats = ['UNI', 'SNA'] if feats is None else feats

        self.uni_only_ff = uni_only_ff
        self.uni_count = uni_count

        self.sna = sna.SNA()
        self.sna_mode = sna_mode
        self.sna_min_lines = sna_min_lines
        self.sna_centralities = ['btwn'] if sna_centralities is None else sna_centralities

        self.trained = False
        self.verbose = verbose

    def transform(self, X):
        print('Transforming {} samples into {}'.format(str(len(X)), ', '.join(self.feats)))
        feat_mats = []

        if 'UNI' in self.feats:
            print('Building unigrams model...')
            # corpus to train unigrams model - either all fem dialogue or all fem-fem dialogue
            diag_per_movie = []
            for i,(id, path,char_dict) in enumerate(X):
                this_diag = ''
                if self.verbose and i % 50 == 0: print(i)
                if 'agarwal' in path:
                    source = 'agarwal'
                    scenes = get_boundaries_agarwal(path)
                else:
                    source = 'gorinski'
                    scenes = get_boundaries_gorinski(path)
                var2info = ag.get_variant_as_key(char_dict)
                for scene in scenes:
                    cdl = ag.get_char_diag_list(scene, var2info, source)
                    if self.uni_only_ff:
                        ffs = ag.get_ff_conversations(cdl)
                        for ff in ffs:
                            for char,line in ff:
                                this_diag += line
                    else:
                        for (char,gen,score),diag in cdl:  # for each character/line
                            if score != 'None' and float(score) > .5:
                                line = ' '.join(diag)
                                if len(line) > 0:
                                    this_diag += ' ' + line
                diag_per_movie.append(this_diag)

            # transform into bag-of-words unigram model
            unigrams = CountVectorizer(max_features=self.uni_count).fit_transform(diag_per_movie)
            print('Unigrams:', unigrams.shape)
            feat_mats.append(unigrams.toarray())

        if 'SNA' in self.feats:
            print('Building SNA features...')
            sn_feats = []
            for i,(id, path, char_dict) in enumerate(X):
                if self.verbose and i % 50 == 0: print(i)
                sn_feats.append(self.sna.transform_into_feats(id, self.sna_mode, self.sna_min_lines, self.sna_centralities))
            sn_feats = np.array(sn_feats)
            print('SNA features:', sn_feats.shape)
            feat_mats.append(sn_feats)

        X = np.concatenate(feat_mats, axis=1)
        print('X-shape:', X.shape)

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
    # X, y = get_t3_data()
    # ut.split_and_save(X, y, save_file='t3_split.pkl')

    # for test_type in ['all', 'agarwal']:
    #     print('\nEvaluating on', test_type.upper(), 'data...')
    #     eval_rule_based(test=test_type)

    for test_type in ['all_cv', 'agarwal_cv']:
        print('\nEvaluating on', test_type.upper(), 'data...')
        eval_clf(test=test_type, uni_only_ff=False, feats=['UNI'])