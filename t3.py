import align_gender as ag
import csv
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
from sklearn.preprocessing import MinMaxScaler

'''PREPARE DATA'''
def get_t3_data(test):
    assert(test == 'train' or test == 'agarwal')
    if test == 'train':
        X, _, y, _ = pickle.load(open('train_test.pkl', 'rb'))
    else:
        agarwal_data = ut.get_data(source='agarwal')
        data = list(agarwal_data.items())
        X = np.array([(x[0], x[1][4], x[1][5]) for x in data])  # id, path, char dict
        y = np.array([int(x[1][3]) for x in data])  # Bechdel label

    X_include = []
    y_include = []
    for movie,label in zip(X,y):
        if label >= 2:  # only include movies that already passed T2
            X_include.append(movie)
            label = 1 if label == 3 else 0
            y_include.append(label)

    return X_include, y_include


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
    X, y = get_t3_data(test)
    X = [(x[1], x[2]) for x in X]  # don't need id for rule-based
    rb = T3RuleBased()
    pred = rb.predict(X)
    print(eval(y, pred, verbose=True))

def eval_clf(test, **kwargs):
    X, y = get_t3_data(test)
    clf = T3Classifier(**kwargs)

    pred = clf.cross_val(X, y)
    print(eval(y, pred, verbose=True))


class T3RuleBased:
    def __init__(self, binary = True, verbose = False):
        self.binary = binary
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

        no_man_ff = 0
        ff_count = 0
        for scene in scenes:
            cdl = ag.get_char_diag_list(scene, var2info, source)
            ffs = ag.get_ff_conversations(cdl)
            ff_count += len(ffs)
            # len(ffs) > 0 means it passes consecutive soft
            for ff in ffs:
                if self.no_man_conversation(ff, male_chars):
                    no_man_ff += 1
                    if self.binary:
                        return 1
        if self.binary:
            return 0
        return no_man_ff, ff_count

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
    def __init__(self, version = 'local', feats = None, uni_only_ff = True, uni_count = 1000, sna_mode = 'consecutive', sna_min_lines = 5, sna_centralities = None, frame_mode = 'both', verbose = False):
        assert(version == 'local' or version == 'global')
        if version == 'global':
            self.clf = LinearSVC(class_weight={0:.45, 1:.55}) # 0-2 vs 3
        else:
            self.clf = LinearSVC(class_weight={0:.72, 1:.28}) # 2 vs 3

        self.feats = ['SNA', 'RB'] if feats is None else feats

        if 'UNI' in self.feats:
            self.uni_only_ff = uni_only_ff
            self.uni_count = uni_count
            self.countvec = None

        if 'SNA' in self.feats:
            self.sna = sna.SNA()
            self.sna_mode = sna_mode
            self.sna_min_lines = sna_min_lines
            self.sna_centralities = ['btwn'] if sna_centralities is None else sna_centralities

        if 'FRA' in self.feats:
            self._load_frames()
            self.fr_mode = frame_mode

        if 'RB' in self.feats:
            self.rb = T3RuleBased(binary=False)

        self.trained = False
        self.verbose = verbose

    def _load_frames(self):
        all_data = ut.get_data()
        self.id2frames = dict((id, {'ff':np.zeros(6), 'fm':np.zeros(12), 'mm':np.zeros(6)}) for id in all_data)
        agarwal_ids = set()
        for source in ['agarwal', 'gorinski']:
            for type in ['ff', 'fm', 'mm']:
                fname = './frames_data/' + source + '/' + source + '_' + type + '.csv'
                with open(fname, 'r') as f:
                    # DictReader returns OrderedDict so values always come in the same order
                    reader = csv.DictReader(f)
                    for row in reader:
                        id = row['movie_id']
                        padding = '0' * (7 - len(id))
                        id = padding + id
                        # set scores if agarwal OR gorinski and agarwal didn't already set
                        if id in self.id2frames and (source == 'agarwal' or id not in agarwal_ids):
                            row.pop('movie_id', None)
                            row.pop('movie_year', None)
                            feats = np.array([float(x) for x in row.values()])
                            self.id2frames[id][type] = feats
                            if source == 'agarwal':
                                agarwal_ids.add(id)

    def transform(self, X):
        if self.verbose: print('Transforming {} samples into {}'.format(str(len(X)), ', '.join(self.feats)))
        feat_mats = []

        if 'UNI' in self.feats:
            if self.verbose: print('Building UNIGRAMS model...')
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
            if self.countvec is None:  # train
                self.countvec = CountVectorizer(max_features=self.uni_count)
                unigrams = self.countvec.fit_transform(diag_per_movie)
            else:  # test
                unigrams = self.countvec.transform(diag_per_movie)
            if self.verbose: print('Unigrams:', unigrams.shape)
            feat_mats.append(unigrams.toarray())

        if 'SNA' in self.feats:
            if self.verbose: print('Building SNA features...')
            sn_feats = []
            for i,(id, path, char_dict) in enumerate(X):
                if self.verbose and i % 50 == 0: print(i)
                sn_feats.append(self.sna.transform_into_feats(id, self.sna_mode, self.sna_min_lines, self.sna_centralities))
            sn_feats = np.array(sn_feats)
            if self.verbose: print('SNA features:', sn_feats.shape)
            feat_mats.append(sn_feats)

        if 'FRA' in self.feats:
            if self.verbose: print('Building FRAME features...')
            fr_feats = []
            for i,(id, path, char_dict) in enumerate(X):
                if self.verbose and i % 50 == 0: print (i)
                scores = self.id2frames[id]
                if self.fr_mode == 'both':
                    feats = np.concatenate((scores['ff'], scores['fm'], scores['mm']), axis=0)
                elif self.fr_mode == 'agency':
                    feats = np.concatenate((scores['ff'][:3], scores['fm'][:3], scores['fm'][6:9], scores['mm'][:3]), axis=0)
                elif self.fr_mode == 'power': # power
                    feats = np.concatenate((scores['ff'][3:], scores['fm'][3:6], scores['fm'][9:], scores['mm'][3:]), axis=0)
                elif self.fr_mode == 'ff':
                    feats = scores['ff']
                elif self.fr_mode == 'fm':
                    feats = scores['fm']
                elif self.fr_mode == 'ffmm':
                    feats = np.concatenate((scores['ff'], scores['mm']), axis=0)
                elif self.fr_mode == 'mm':
                    feats = scores['mm']
                else:
                    raise ValueError('Invalid frame mode:', self.fr_mode)
                fr_feats.append(feats)
            fr_feats = np.array(fr_feats)
            fr_feats = MinMaxScaler().fit_transform(fr_feats)
            if self.verbose: print('FRAME features:', fr_feats.shape)
            feat_mats.append(fr_feats)

        if 'RB' in self.feats:
            if self.verbose: print('Building RULE-BASED features...')
            X_rb = [(x[1], x[2]) for x in X]
            rb_feats = self.rb.predict(X_rb)
            rb_feats = np.array(rb_feats)
            if self.verbose: print('RB features:', rb_feats.shape)
            feat_mats.append(rb_feats)

        X = np.concatenate(feat_mats, axis=1)
        if self.verbose: print('X-shape:', X.shape)

        return X

    def train(self, X, y, add_feat = None):
        X = self.transform(X)
        if add_feat is not None:
            X = np.concatenate((X, add_feat), axis=1)
        self.clf.fit(X, y)
        self.trained = True

    def predict(self, X, add_feat = None):
        if not self.trained:
            print('Should not predict before training.')
            return None
        X = self.transform(X)
        if add_feat is not None:
            X = np.concatenate((X, add_feat), axis=1)
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
        eval_clf(test=test_type, feats=['FRA', 'RB'], uni_count=1000, uni_only_ff=False, sna_mode='consecutive', frame_mode ='ff', verbose=False)
