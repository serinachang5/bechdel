import numpy as np
import pickle
import project.src.util as util
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.naive_bayes import GaussianNB

'''
BASELINES
    1. 2-dims: gender proportions
    [[13 36]
     [ 8 39]]
    Accuracy: 0.542. F1: 0.639
    2. 1-dim: count of female interactions (two consecutive lines by females)
    [[43  6]
     [28 19]]
    Accuracy: 0.646. F1: 0.528
    3. 3-dims: gender proportions + count of female interactions
    [[41  8]
     [25 22]]
    Accuracy: 0.656. F1: 0.571
    4. 800-dims: unigrams of all female dialogue
    [[44  5]
     [24 23]]
    Accuracy: 0.698. F1: 0.613
'''

def get_vecs(data, X_vec=None):
    # where data is list of tuples in the form (title, (parsed_CD_tuples, label))
    all_parsed = [sample[1][0] for sample in data]
    if X_vec is None:
        X_vec = fit_X(all_parsed)
    X1 = baseline_gender_proportion(all_parsed)
    X2 = baseline_fem_interactions(all_parsed)
    X3 = np.concatenate((X1, X2), axis=1)
    X4 = baseline_fem_unigrams(all_parsed, X_vec)
    X = X1 # choose which baseline you want
    y = np.array([sample[1][1] for sample in data])
    return X, X_vec, y

def baseline_gender_proportion(all_parsed):
    fems = set(util.get_names("../data/female.txt"))
    males = set(util.get_names("../data/male.txt"))
    X = []
    for cd_list in all_parsed:
        characters = set([cd[0].lower() for cd in cd_list])
        counts = np.zeros(2)
        for c in characters:
            if c in fems and c not in males:
                counts[0] += 1
            elif c in males and c not in fems:
                counts[1] += 1
        if sum(counts) > 0:
            props = np.divide(counts, sum(counts))
        else:
            props = counts
        X.append(props)
    return np.array(X)

def baseline_fem_interactions(all_parsed):
    fems = set(util.get_names("../data/female.txt"))
    X = np.zeros((len(all_parsed),1))
    for i,cd_list in enumerate(all_parsed):
        count = 0
        for j,(C,D) in enumerate(cd_list):
            cnext = ""
            c = C.lower()
            if j < (len(cd_list)-1):
                cnext = cd_list[j+1][0].lower()
            if c in fems and cnext in fems:
                count += 1
        X[i][0] = count
    return X

def baseline_fem_unigrams(all_parsed, vectorizer):
    fems = set(util.get_names("../data/female.txt"))
    all_docs = [] # list of docs, where each doc equals all lines by females in the movie
    for cd_list in all_parsed:
        doc = []
        for i,(C,D) in enumerate(cd_list):
            c = C.lower()
            d = D.lower()
            if c in fems:
                doc.append(d)
        all_docs.append(" ".join(doc))
    X = vectorizer.transform(all_docs)
    return X.toarray()

def fit_X(all_parsed):
    fems = set(util.get_names("../data/female.txt"))
    all_diag = [] # list of all lines by females in the corpus
    for cd_list in all_parsed:
        for C,D in cd_list:
            c = C.lower()
            d = D.lower()
            if c in fems:
                all_diag.append(d)
    vectorizer = CountVectorizer(max_features=800)
    vectorizer.fit(all_diag)
    return vectorizer

def train_and_test_clf(X_train, y_train, X_test, y_test):
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    mat = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return mat, acc, f1

if __name__ == "__main__":
    train = pickle.load(open("../data/parsed/train.p", "rb"))
    X_train, vec, y_train = get_vecs(train)
    print("Train dimensions:", X_train.shape, y_train.shape)

    test = pickle.load(open("../data/parsed/val.p", "rb"))
    X_test, _ , y_test = get_vecs(test, X_vec=vec)
    print("Test dimensions:", X_test.shape, y_test.shape)

    mat, acc, f1 = train_and_test_clf(X_train, y_train, X_test, y_test)
    print(mat)
    print("Accuracy: {}. F1: {}".format(round(acc, 3), round(f1, 3)))