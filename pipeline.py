import t1
import t2
import t3
import pickle
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_predict
from collections import Counter


def get_test_data():
    _, X, _, y = pickle.load(open('train_test.pkl', 'rb'))
    y = np.array([1 if label == 3 else 0 for label in y])
    return X, y

def eval(true, pred, verbose = False):
    per_class = f1_score(true, pred, average=None)
    macro = f1_score(true, pred, average='macro')
    report = 'F1s: Per-class: {}. Macro: {}.'.format(per_class, round(macro, 4))
    if verbose:
        report += '\nAccuracy: ' + str(round(accuracy_score(true, pred), 4))
        report += '\n' + str(confusion_matrix(true, pred))
    return report

class Pipeline:
    def __init__(self, mode):
        assert(mode == 'hard' or mode == 'soft')
        self.mode = mode
        self.t1 = t1.T1Classifier()
        if self.mode == 'hard':
            self.t2 = t2.T2Classifier(version='local')
            self.t3 = t3.T3Classifier(version='local')
        else:
            self.t2 = t2.T2Classifier(version='global')
            self.t3 = t3.T3Classifier(version='global')

        self.trained = False

    def train(self):
        if self.mode == 'hard':
            print('Training T1...')
            X1, y1 = t1.get_t1_data('train')
            self.t1.train(X1, y1)

            print('Training T2 on local data...')
            X2, y2 = t2.get_t2_data('train')
            self.t2.train(X2, y2)

            print('Training T3 on local data...')
            X3, y3 = t3.get_t3_data('train')
            self.t3.train(X3, y3)
        else:
            X_train, _, y_train, _ = pickle.load(open('train_test.pkl', 'rb'))

            print('Training T1...')
            X1 = [(x[1], x[2]) for x in X_train]
            y1 = np.array([1 if label >= 1 else 0 for label in y_train])
            self.t1.train(X1, y1)

            print('Training T2 on global data with T1 prediction...')
            X2 = X1
            y2 = np.array([1 if label >= 2 else 0 for label in y_train])
            pred1 = self.t1.predict(X1).reshape(-1, 1)
            self.t2.train(X2, y2, add_feat=pred1)

            print('Training T3 on global data with T1 and T2 predictions...')
            X3 = X_train
            y3 = np.array([1 if label >= 3 else 0 for label in y_train])
            pred2 = self.t2.predict(X2, add_feat=pred1).reshape(-1, 1)
            prev_preds = np.concatenate((pred1, pred2), axis=1)
            self.t3.train(X3, y3, add_feat=prev_preds)

        self.trained = True

    def predict(self, X):
        if not self.trained:
            print('Should not predict before training.')
            return None

        if self.mode == 'hard':
            print('Making hard pipeline predictions...')
            pred = []
            for x in X:
                pred.append(self._pipe_predict(x))
        else:
            print('Making soft pipeline predictions...')
            X1 = [(x[1], x[2]) for x in X]
            pred1 = self.t1.predict(X1).reshape(-1, 1)
            X2 = X1
            pred2 = self.t2.predict(X2, add_feat=pred1).reshape(-1, 1)
            prev_preds = np.concatenate((pred1, pred2), axis=1)
            X3 = X
            pred = self.t3.predict(X3, add_feat=prev_preds)

        return pred

    def _pipe_predict(self, x):
        t1_pred = self.t1.predict([(x[1], x[2])])
        if t1_pred[0] == 0:
            return 0
        t2_pred = self.t2.predict([(x[1], x[2])])
        if t2_pred[0] == 0:
            return 0
        t3_pred = self.t3.predict([x])
        return t3_pred[0]


class GlobalClassifier:
    def __init__(self):
        self.t1 = t1.T1Classifier()
        self.t2 = t2.T2Classifier()
        self.t3 = t3.T3Classifier()
        self.clf = LinearSVC()
        self.trained = False

    def transform(self, X):
        X_trunc = [(x[1], x[2]) for x in X]
        X1 = self.t1.transform(X_trunc)
        X2 = self.t2.transform(X_trunc)
        X3 = self.t3.transform(X)
        X = np.concatenate((X1, X2, X3), axis=1)
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


if __name__ == "__main__":
    gc = GlobalClassifier()
    X_train, X_test, y_train, y_test = pickle.load(open('agarwal_train_test.pkl', 'rb'))
    y_train = np.array([1 if label == 3 else 0 for label in y_train])
    y_test = np.array([1 if label == 3 else 0 for label in y_test])
    gc.train(X_train, y_train)
    pred = gc.predict(X_test)
    print(eval(y_test, pred, verbose=True))

    # X_test, y_test = get_test_data()
    # pl = Pipeline(mode='hard')
    # pl.train()
    # pred = pl.predict(X_test)
    # print(eval(y_test, pred, verbose=True))