import t1
import t2
import t3
import pickle
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

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
    def __init__(self):
        self.t1 = t1.T1Classifier()
        self.t2 = t2.T2Classifier()
        self.t3 = t3.T3Classifier()
        self.trained = False

    def train(self):
        X, _, y, _ = pickle.load(open('train_test.pkl', 'rb'))

        print('Training T1...')
        X1 = [(x[1], x[2]) for x in X]
        y1 = np.array([1 if label >= 1 else 0 for label in y])
        self.t1.train(X1, y1)

        print('Training T2...')
        X2 = X1
        y2 = np.array([1 if label >= 2 else 0 for label in y])
        pred1 = self.t1.predict(X1).reshape(-1, 1)
        self.t2.train(X2, y2, add_feat=pred1)

        print('Training T3...')
        X3 = X
        y3 = np.array([1 if label >= 3 else 0 for label in y])
        pred2 = self.t2.predict(X2, add_feat=pred1).reshape(-1, 1)
        prev_preds = np.concatenate((pred1, pred2), axis=1)
        self.t3.train(X3, y3, add_feat=prev_preds)

        # print('Training T3...')
        # # X3, y3 = t3.get_t3_data(test='train') # only movies that passed T2
        #
        # X3 = []
        # y3 = []
        # for movie,label in zip(X,y):
        #     if label >= 1:  # only movies that passed T1
        #         X3.append(movie)
        #         label = 1 if label == 3 else 0
        #         y3.append(label)
        #
        # # X3 = X # all movies
        # # y3 = np.array([1 if label == 3 else 0 for label in y])
        # self.t3.train(X3, y3)

        self.trained = True

    def predict(self, X):
        if not self.trained:
            print('Should not predict before training.')
            return None
        print('Making predictions...')
        X1 = [(x[1], x[2]) for x in X]
        pred1 = self.t1.predict(X1).reshape(-1, 1)
        X2 = X1
        pred2 = self.t2.predict(X2, add_feat=pred1).reshape(-1, 1)
        prev_preds = np.concatenate((pred1, pred2), axis=1)
        X3 = X
        pred3 = self.t3.predict(X3, add_feat=prev_preds)
        # pred = []
        # for x in X:
        #     pred.append(self._predict(x))
        return pred3

    def _predict(self, x):
        t1_pred = self.t1.predict([(x[1], x[2])])
        if t1_pred[0] == 0:
            return 0
        t2_pred = self.t2.predict([(x[1], x[2])])
        if t2_pred[0] == 0:
            return 0
        t3_pred = self.t3.predict([x])
        return t3_pred[0]

if __name__ == "__main__":
    X_test, y_test = get_test_data()

    # pl = Pipeline()
    # pl.train()
    # pred = pl.predict(X_test)

    clf = t3.T3Classifier()
    X_train, _, y_train, _ = pickle.load(open('train_test.pkl', 'rb'))
    y_train = np.array([1 if label == 3 else 0 for label in y_train])
    clf.train(X_train, y_train)
    pred = clf.predict(X_test)

    print(eval(y_test, pred, verbose=True))