import json
import os
import numpy as np
import random

def parse_by_gender_file(path_to_file):
    with open(path_to_file, 'r') as f:
        lines = f.readlines()
        title = lines[0].strip()
        year = lines[1].strip().replace('Year: ', '')
        imdb_id = lines[2].strip().replace('IMDb id: ', '')
        bechdel = lines[3].strip().replace('Bechdel score: ', '')
        path = lines[4].strip().replace('File path: ', '')

        char_dict = {}
        line_idx = 7  # 5 is \n, 6 is CHARACTERS
        while line_idx < len(lines):
            root = lines[line_idx].strip().replace('Root: ', '')
            line_idx += 1
            gender = lines[line_idx].strip().replace('Gender: ', '')
            line_idx += 1
            score = lines[line_idx].strip().replace('Score: ', '')
            line_idx += 2 # skip Variants:
            variants = []
            while line_idx < len(lines) and not lines[line_idx].strip().startswith('Root: '):
                variant = lines[line_idx].strip()
                if len(variant) > 0:
                    variants.append(variant)
                line_idx += 1
            char_dict[root] = [gender, score, variants]
        info = [title, year, imdb_id, bechdel, path, char_dict]
        return imdb_id, info

def get_data(mode = 'combined'):
    path = './movie_by_gender/'

    if mode == 'agarwal' or mode == 'gorinski':
        id_to_info = {}
        for fname in os.listdir(path + mode + '/'):
            id, info = parse_by_gender_file(path + mode + '/' + fname)
            id_to_info[id] = info
        # print('Unique movies in ' + mode.capitalize() + ':', len(id_to_info))
        return id_to_info

    elif mode == 'combined':
        aga = get_data(mode='agarwal')
        gor =  get_data(mode='gorinski')
        combined = aga
        overlap = 0
        for id in gor:
            if id not in combined:
                combined[id] = gor[id]
            else:
                overlap += 1
        # print('Overlap between Agarwal and Gorinski:', overlap)
        print('Unique movies in Combined:', len(combined))
        return combined

def check_distribution(data, test = None):
    num_class = 4 if test is None else 2
    counts = np.zeros(num_class)
    for movie_id, info in data.items():
        title, year, imdb_id, bechdel, path, char_dict = info
        score = int(bechdel)
        if test is None:
            counts[score] += 1
        else:
            label = 1 if score >= test else 0
            counts[label] += 1
    return counts

def parse_json(fname):
    with open(fname, "r") as f:
        content = f.readline()
        parsed = json.loads(content)
        return parsed

def standardize(title):
    return "".join(title.split()).lower()

'''BECHDEL FUNCTIONS'''
PREFIX = "../data/"
BECHDEL_PATH = PREFIX+"allmovies.json"

# return counts for each rating [0,3]
def get_bechdel_stats(samples, titles=None):
    ratings = np.zeros(4, dtype=np.int)
    for sample in samples:
        if titles is not None:
            t = standardize(sample["title"])
            if t in titles:
                r = int(sample["rating"])
                ratings[r] += 1
        else:
            r = int(sample["rating"])
            ratings[r] += 1
    return ratings

def get_title_to_rating(data):
    mapping = {}
    for sample in data:
        title = standardize(sample["title"])
        rating = int(sample["rating"])
        mapping[title] = rating
    return mapping

# return set of titles in Bechdel data
def get_bechdel_titles(data):
    titles = set()
    for sample in data:
        title = standardize(sample["title"])
        titles.add(title)
    return titles

'''SCREENPLAY FUNCTIONS'''
WALKER_RAW = PREFIX+"walker2015_raw/"
WALKER_DIAG = PREFIX+"walker2015_scene_diag/dialogs/"
WALKER_SCENE = PREFIX+"walker2015_scene_diag/scenes/"

# return set of titles for walker data
def get_walker_titles():
    titles = set()
    for genre in os.listdir(WALKER_RAW):
        if genre.isalpha():
            for fname in os.listdir(WALKER_RAW+genre):
                title = str(fname.rsplit('.txt', 1)[0])
                titles.add(title)
    return titles

# return set of dialogue files given titles
def get_diag_files(diag_path, titles):
    titles = set([standardize(t) for t in titles])
    files = {}
    for genre in os.listdir(diag_path):
        if genre.isalpha():
            for fname in os.listdir(diag_path+genre):
                title = fname.rsplit('_dialog.txt', 1)[0]
                if title in titles and title not in files:
                    diag = get_diag(title, genre)
                    files[title] = diag
    return files

# return the raw text file for a given movie
def get_raw(movie, genre):
    title = "".join(movie.split()).lower()
    raw_name = WALKER_RAW + genre.capitalize() + "/" + title + ".txt"
    with open(raw_name, "r") as raw:
        return raw.read()

# return the dialogue file for a given movie
def get_diag(movie, genre):
    title = "".join(movie.split()).lower()
    diag_name = WALKER_DIAG + genre.capitalize() + "/" + title + "_dialog.txt"
    with open(diag_name, "r") as diag:
        return diag.read()

AGARWAL_PASS = PREFIX+"agarwal2015_screenplays/pass/"
AGARWAL_FAIL = PREFIX+"agarwal2015_screenplays/fail/"
# return set of titles for walker data
def get_agarwal_titles(folder="both"):
    titles = set()
    if folder == "both" or folder == "pass":
        for fname in os.listdir(AGARWAL_PASS):
            if fname[0].isdigit():
                title = str(fname.split('_', 1)[1]) # remove prefix id
                title = title.rsplit('.txt', 1)[0] # remove extension
                title = title.replace("_", " ")
                title = standardize(title)
                titles.add(title)
    if folder == "both" or folder == "fail":
        for fname in os.listdir(AGARWAL_FAIL):
            if fname[0].isdigit():
                title = str(fname.split('_', 1)[1]) # remove prefix id
                title = title.rsplit('.txt', 1)[0] # remove extension
                title = title.replace("_"," ")
                title = standardize(title)
                titles.add(title)
    return titles

# return the scene file for a given movie
def get_scene(movie, genre):
    title = "".join(movie.split()).lower()
    scene_name = WALKER_RAW + genre.capitalize() + "/" + title + "_scene.txt"
    with open(scene_name, "r") as scene:
        return scene.read()

def get_files(movie, genre):
    raw = get_raw(movie, genre)
    diag = get_diag(movie, genre)
    scene = get_scene(movie, genre)
    return raw, diag, scene

# return continuous dialogue in a movie by aligning
# its dialogue file to its raw file
def find_continuous_turns(D, R):
    D_with_splits = []
    D = [line.strip() for line in D.split("\n")]
    R = [line.strip() for line in R.split("\n")]
    di = 0
    ri = 0
    while di < len(D) and ri < len(R):
        curr = []
        while di < len(D) and ri < len(R) and D[di] == R[ri]:
            if len(D[di]) > 0:
                curr.append(D[di])
            di += 1
            ri += 1
        D_with_splits.append(curr)
        while di < len(D) and ri < len(R) and D[di] != R[ri]:
            ri += 1
    return D_with_splits

'''OTHER FUNCTIONS'''
def get_names(fname):
    names = []
    with open(fname, "r") as f:
        for line in f.readlines():
            line = line.strip().lower()
            if len(line) < 1 or line.startswith("#"):
                pass
            else:
                names.append(line)
    return names

if __name__ == "__main__":
    bechdel_samples = parse_json(BECHDEL_PATH)
    B = get_bechdel_titles(bechdel_samples)
    print("Number of Bechdel movies:", len(bechdel_samples))
    print("Distribution of Bechdel scores:", get_bechdel_stats(bechdel_samples))
    # Number of Bechdel movies: 7606
    # Distribution of Bechdel scores: [ 784 1671  760 4391]

    W = get_walker_titles()
    print("Number of movies in Walker:", len(W))
    overlap = B.intersection(W)
    print("Overlap between Walker and Bechdel:", len(overlap))
    print("Distribution of these movies:", get_bechdel_stats(bechdel_samples, titles=overlap))
    # Number of movies with dialogue: 958
    # Number of movies with dialogue AND Bechdel ratings: 464
    # Distribution of these movies: [ 28 146  69 250]

    A = get_agarwal_titles()
    print("Number of movies in Agarwal:", len(A))
    overlap = B.intersection(A)
    print("Overlap between Agarwal and Bechdel:", len(overlap))
    print("Distribution of these movies:", get_bechdel_stats(bechdel_samples, titles=overlap))

    WA = W.union(A)
    print("Number of movies in Walker + Agarwal:", len(WA))
    overlap = B.intersection(WA)
    print("Overlap between Walker-Agarwal and Bechdel:", len(overlap))
    print("Distribution of these movies:", get_bechdel_stats(bechdel_samples, titles=overlap))

    W_unique = W.difference(A)
    print("Number of unique movies in Walker:", len(W_unique))
    overlap = B.intersection(W_unique)
    print("Number of Bechdel aligned movies added by Walker:", len(overlap))
    # sample_movie = "Blade Runner"
    # sample_movie_genre = "Action"
    # raw, diag, scene = get_files(sample_movie, sample_movie_genre)
    # cont_turn = find_continuous_turns(diag, raw)
    # print(cont_turn)
    # print("Number of continuous turns in {}:".format(sample_movie), len(cont_turn))
    # print("Sample turns:", random.sample(cont_turn, 5))
    # Number of continuous turns in Blade Runner: 373
    # Sample turns: [['DECKARD', 'Fuck you, then.'], ['DECKARD', 'Listen, Sergeant...'], ['DECKARD', "Stop or you're dead!"], ['BATTY', "I'm sure glad you found us,", 'Sebastian.  What do you think,', 'Mary?', 'MARY', "I don't think there is another", 'human being in this whole world', 'who would have helped us.', 'BATTY', 'Pris?'], ['BATTY', "We're not used to the big city.", "Where we come from it's not so", 'easy to get lost.', 'MARY', 'You certainly have a nice place', 'here.', 'BATTY', 'Well stocked.']]    # Sample turns: [['DECKARD', 'Fuck you, then.'], ['DECKARD', 'Listen, Sergeant...'], ['DECKARD', "Stop or you're dead!"], ['BATTY', "I'm sure glad you found us,", 'Sebastian.  What do you think,', 'Mary?', 'MARY', "I don't think there is another", 'human being in this whole world', 'who would have helped us.', 'BATTY', 'Pris?'], ['BATTY', "We're not used to the big city.", "Where we come from it's not so", 'easy to get lost.', 'MARY', 'You certainly have a nice place', 'here.', 'BATTY', 'Well stocked.']]

    # male_names = get_names("./male.txt")
    # print(male_names[:10])
