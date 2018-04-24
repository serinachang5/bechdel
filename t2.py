from collections import Counter
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
import numpy as np
import pickle
from project.src.util import get_data, check_distribution, get_variant_as_key
from project.src.get_scene_boundaries import get_boundaries_agarwal, get_boundaries_gorinski
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

def get_t2_data():
    combined_data = get_data()
    combined_data = sorted(combined_data.items(), key=lambda x: x[0])  # sort by id
    X = np.array([(x[0], x[1][4], x[1][5]) for x in combined_data])  # (id, path, char dict)
    y = np.array([1 if int(x[1][3]) >= 2 else 0 for x in combined_data], dtype=np.int)  # check if sample passes T1
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

    pickle.dump([X_train, X_val, X_test, y_train, y_val, y_test], open('t2_split.p', 'wb'))
    print('Saved t2_split.p')
    return X_train, X_val, X_test, y_train, y_val, y_test

def eval(true, pred, verbose = False):
    per_class = f1_score(true, pred, average=None)
    macro = f1_score(true, pred, average='macro')
    report = 'F1s: Per-class: {}. Macro: {}.'.format(per_class, round(macro, 4))
    if verbose:
        report += '\nAccuracy: ' + str(round(accuracy_score(true, pred), 4))
        report += '\n' + str(confusion_matrix(true, pred))
    return report

class RuleBased():
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
            gl = self.get_gender_list(scene, var2info, source)
            fem_chars = set()
            for char, gen, score in gl:
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

        # check scenes
        for scene in scenes:
            gl = self.get_gender_list(scene, var2info, source)
            prev_char = ''
            prev_gen = ''
            prev_score = -1
            for char, gen, score in gl:
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

    def get_gender_list(self, scene, var2info, source):
        char_gen_list = []

        if source == 'agarwal':
            for line in scene:
                if line.startswith('C|'):
                    # process variant
                    var = line.split(None, 1)[1]  # cut off C| + white space
                    if var.startswith('('):  # probably a description e.g. '(QUIETLY)'
                        continue
                    var = var.strip()  # strip trailing white space
                    if var in var2info:
                        char_gen_list.append(var2info[var])  # char, gen, score
        else:  # source == 'gorinski'
            for line in scene:  # try all lines bc no marker of character
                possible = line.strip() # strip leading and trailing white space
                if possible.startswith('('): # probably a description e.g. '(QUIETLY)'
                    continue
                if possible in var2info:
                    char_gen_list.append(var2info[possible])
        return char_gen_list


if __name__ == "__main__":
    # X, y = get_t2_data()
    # X_train, X_val, X_test, y_train, y_val, y_test = split_and_save(X, y)
    #
    # print(Counter(y_train))
    # print(Counter(y_val))
    # print(Counter(y_test))

    X_train, X_val, X_test, y_train, y_val, y_test = pickle.load(open('t2_split.p', 'rb'))
    X_train = [(x[1], x[2]) for x in X_train]
    X_val = [(x[1], x[2]) for x in X_val]
    X_test = [(x[1], x[2]) for x in X_test]

    rb = RuleBased()

    print('RULE-BASED: OVERLAP, HARD')
    pred = rb.predict(X_test, interact='overlap', mode='hard')
    print(eval(y_test, pred, verbose=True))
    pred = rb.predict(np.concatenate((X_train, X_val, X_test)), interact='overlap', mode='hard')
    print(eval(np.concatenate((y_train, y_val, y_test)), pred, verbose=True))

    print('\nRULE-BASED: OVERLAP, SOFT')
    pred = rb.predict(X_test, interact='overlap', mode='soft')
    print(eval(y_test, pred, verbose=True))
    pred = rb.predict(np.concatenate((X_train, X_val, X_test)), interact='overlap', mode='soft')
    print(eval(np.concatenate((y_train, y_val, y_test)), pred, verbose=True))

    print('\nRULE-BASED: CONSECUTIVE, HARD')
    pred = rb.predict(X_test, interact='consecutive', mode='hard')
    print(eval(y_test, pred, verbose=True))
    pred = rb.predict(np.concatenate((X_train, X_val, X_test)), interact='consecutive', mode='hard')
    print(eval(np.concatenate((y_train, y_val, y_test)), pred, verbose=True))

    print('\nRULE-BASED: CONSECUTIVE, SOFT')
    pred = rb.predict(X_test, interact='consecutive', mode='soft')
    print(eval(y_test, pred, verbose=True))
    pred = rb.predict(np.concatenate((X_train, X_val, X_test)), interact='consecutive', mode='soft')
    print(eval(np.concatenate((y_train, y_val, y_test)), pred, verbose=True))
