import numpy as np
import os
import pickle
import project.src.util as util

def parse_agarwal(content):
    lines = content.split("\n")
    tuples = []
    i = 0
    while i < len(lines):
        if lines[i].startswith("C|"):
            char = lines[i].rsplit(None, 1)[1]
            fragments = []
            i += 1
            while i < len(lines) and lines[i].startswith("D|"):
                frag = lines[i].split(None, 1)[1]
                fragments.append(frag)
                i += 1
            tuples.append((char, " ".join(fragments)))
        else:
            i += 1
    return tuples

def parse_walker(content):
    lines = []
    content = content.split("\n")
    for x in content:
        if len(x) > 0:
            lines.append(x)

    tuples = []
    i = 0
    while i < len(lines):
        if lines[i].isupper():
            char = lines[i]
            fragments = []
            i += 1
            while i < len(lines) and not lines[i].isupper():
                fragments.append(lines[i])
                i += 1
            tuples.append((char, " ".join(fragments)))
        else:
            i += 1
    return tuples

def get_data():
    title2data = {}
    for fname in os.listdir(util.AGARWAL_PASS):
        label = 1
        if fname[0].isdigit():
            title = str(fname.split('_', 1)[1]) # remove prefix id
            title = title.rsplit('.txt', 1)[0] # remove extension
            title = title.replace("_", " ")
            title = util.standardize(title)
            with open(util.AGARWAL_PASS+fname, "r") as f:
                content = f.read()
                parsed = parse_agarwal(content)
                title2data[title] = ((parsed, label))
    for fname in os.listdir(util.AGARWAL_FAIL):
        label = 0
        if fname[0].isdigit():
            title = str(fname.split('_', 1)[1]) # remove prefix id
            title = title.rsplit('.txt', 1)[0] # remove extension
            title = title.replace("_", " ")
            title = util.standardize(title)
            with open(util.AGARWAL_FAIL+fname, "r") as f:
                content = f.read()
                parsed = parse_agarwal(content)
                title2data[title] = (parsed, label)

    # WALKER
    bechdel_samples = util.parse_json(util.BECHDEL_PATH)
    ratings = util.get_title_to_rating(bechdel_samples)
    bechdel_titles = set(ratings.keys())
    dfiles = util.get_diag_files(util.WALKER_DIAG, bechdel_titles)
    for title in dfiles:
        label = 0 if ratings[title] < 3 else 1
        parsed = parse_walker(dfiles[title])
        title2data[title] = (parsed, label)

    data = sorted(title2data.items())
    return data

def train_val_test_split(data, train_split, val_split):
    np.random.shuffle(data)
    train_cutoff = int(len(data) * train_split)
    val_cutoff = int(len(data) * (train_split+val_split))
    train = data[:train_cutoff]
    val = data[train_cutoff:val_cutoff]
    test = data[val_cutoff:]
    return train, val, test

def get_label_stats(labels):
    return "{} total; {} failed, {} passed ({}%)".format(
        len(labels), labels.count(0), labels.count(1), round(100*labels.count(1)/len(labels), 1))

if __name__ == "__main__":
    data = get_data()
    labels = [x[1][1] for x in data]
    print("all:", get_label_stats(labels))
    pickle.dump(data, open("../data/parsed/all.p", "wb"))

    data = pickle.load(open("../data/parsed/all.p", "rb"))
    tr, va, te = train_val_test_split(data, train_split=.7, val_split=.15)

    tr_labels = [x[1][1] for x in tr]
    print("tr:", get_label_stats(tr_labels))
    pickle.dump(tr, open("../data/parsed/train.p", "wb"))

    va_labels = [x[1][1] for x in va]
    print("va:", get_label_stats(va_labels))
    pickle.dump(va, open("../data/parsed/val.p", "wb"))

    te_labels = [x[1][1] for x in te]
    print("te:", get_label_stats(te_labels))
    pickle.dump(te, open("../data/parsed/test.p", "wb"))