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

        X1, y1 = t1.get_t1_data(test='train')
        print('Training T1...')
        self.t1.train(X1, y1)

        X2, y2 = t2.get_t2_data(test='train')
        print('Training T2...')
        self.t2.train(X2, y2)

        # X3, y3 = t3.get_t3_data(test='train') # only movies that passed T2
        X3 = []
        y3 = []
        for movie,label in zip(X,y):
            if label >= 1:  # only movies that passed T1
                X3.append(movie)
                label = 1 if label == 3 else 0
                y3.append(label)
        # X3 = X # all movies
        # y3 = np.array([1 if label == 3 else 0 for label in y])
        print('Training T3...')
        self.t3.train(X3, y3)

        self.trained = True

    def predict(self, X):
        if not self.trained:
            print('Should not predict before training.')
            return None
        print('Making predictions...')
        pred = []
        for x in X:
            pred.append(self._predict(x))
        return np.array(pred)

    def _predict(self, x):
        # t1_pred = self.t1.predict([(x[1], x[2])])
        # if t1_pred[0] == 0:
        #     return 0
        # t2_pred = self.t2.predict([(x[1], x[2])])
        # if t2_pred[0] == 0:
        #     return 0
        t3_pred = self.t3.predict([x])
        return t3_pred[0]

if __name__ == "__main__":
    X_test, y_test = get_test_data()
    pl = Pipeline()
    pl.train()
    pred = pl.predict(X_test)
    print(eval(y_test, pred, verbose=True))