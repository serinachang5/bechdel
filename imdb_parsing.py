from csv import DictReader
import pickle
from project.src.util import standardize
import imdb
import os
import string

DEBUGGING = False

def align_movie_info(type):
    movie_to_info = {}
    db = imdb.IMDb()
    found_id = 0
    found_dia = 0

    if type == 'walker':
        walker_raw = './data/walker2015_raw/'
        for genre in os.listdir(walker_raw):
            if genre != '.DS_Store':
                print('On walker genre:', genre)
                genre_path = walker_raw + genre + '/'
                for raw_file in os.listdir(genre_path):
                    raw_file_path = genre_path + raw_file
                    dialog_file_path = get_walker_diag(raw_file_path)
                    if DEBUGGING: print('Dialog path:', dialog_file_path)

                    if dialog_file_path is not None:
                        found_dia += 1
                        info = {'source':'walker', 'title':None, 'id':None}
                        title = find_walker_title(raw_file_path)
                        if DEBUGGING: print('Found title:', title)

                        if title is not None:
                            info['title'] = title
                            char_doc = parse_walker_chars(dialog_file_path)
                            if DEBUGGING: print('Parsed characters:', char_doc)

                            imdb_id = match_id(db, title, char_doc)
                            if DEBUGGING: print('Aligned id:', imdb_id)

                            if imdb_id is not None:
                                found_id += 1
                                info['id'] = imdb_id
                        movie_to_info[dialog_file_path] = info
                        print(dialog_file_path, info)
            pickle.dump(movie_to_info, open('walker_alignments.p', 'wb'))
        print('Found {} dialog files, found {} ids'.format(found_dia, found_id))

    if type == 'agarwal':
        agarwal_path = './data/agarwal2015_screenplays/'
        for dir in os.listdir(agarwal_path):
            if dir != '.DS_Store':
                print('Agarwal dir:', dir)
                dir_path = agarwal_path + dir + '/'

                for file in os.listdir(dir_path):
                    found_dia += 1
                    file_path = dir_path + file
                    if DEBUGGING: print('File_path:', file_path)

                    info = {'source':'agarwal', 'title':None, 'id':None}
                    title = find_agarwal_title(file_path)
                    info['title'] = title
                    if DEBUGGING: print('Title:', title)

                    char_count, char_doc = parse_agarwal_chars(file_path)
                    if DEBUGGING: print('Char count:', char_count, 'Chars:', char_doc)

                    if char_count > 5:
                        imdb_id = match_id(db, title, char_doc)
                        if imdb_id is not None:
                            found_id += 1
                            info['id'] = imdb_id

                    movie_to_info[file_path] = info
                    print(found_dia, file_path, info)
            pickle.dump(movie_to_info, open('agarwal_alignments.p', 'wb'))

        print('Found {} dialog files, found {} ids'.format(found_dia, found_id))

def match_id(imdb_instance, title, char_doc):
    search_results = imdb_instance.search_movie(title)
    for sr in search_results:
        movie = imdb_instance.get_movie(sr.movieID)
        if movie.has_key('cast'):
            imdb_cast = movie['cast']
            count = 0
            for person in imdb_cast:
                char = str(person.currentRole)
                if len(char)>0:
                    char = char.upper()
                    if char in char_doc:
                        count += 1
                        if count >= 5:
                            return sr.movieID

# WALKER FUNCTIONS
def get_walker_diag(raw_file_path):
    raw_path, genre, filename = raw_file_path.rsplit('/', 2)
    walker_diag = './data/walker2015_scene_diag/dialogs/'
    diag_file_path = walker_diag + genre + '/' + filename.rsplit('.txt', 1)[0] + '_dialog.txt'
    if os.path.isfile(diag_file_path):
        return diag_file_path
    return None

def find_walker_title(raw_file_path):
    with open(raw_file_path, 'r') as f:
        fname = raw_file_path.split('/')[-1]
        fname = fname.rsplit('.txt', 1)[0]
        translator = str.maketrans('', '', string.punctuation)
        i = 0
        while i < 100:
            line = f.readline()
            tokens = [t.lower() for t in line.split()]
            processed = [t.translate(translator) for t in tokens]
            if len(processed) > 0:
                if processed[0] == 'the':
                    processed.append(processed[0])
                    processed.remove(processed[0])
                condensed = ''.join(processed)
                if condensed == fname:
                    title = ' '.join(tokens)
                    return title
            i += 1

def parse_walker_chars(diag_file_path):
    with open(diag_file_path, 'r') as f:
        chars = set()
        for line in f.readlines():
            line = line.strip()
            if len(line) > 0 and line.isupper():
                chars.add(line)
        char_doc = ' '.join(list(chars))
        return char_doc

# AGARWAL FUNCTIONS
def find_agarwal_title(file_path):
    fname = file_path.split('/')[-1]
    fname = fname.rsplit('.txt', 1)[0]
    tokens = fname.split('_')[1:]
    tokens = [t.lower() for t in tokens]
    title = ' '.join(tokens)
    return title

def parse_agarwal_chars(file_path):
    with open(file_path, 'r') as f:
        chars = set()
        for line in f.readlines():
            if line.startswith('C|'):
                char = line.rsplit(None, 1)[1]
                chars.add(char)
        char_doc = ' '.join(list(chars))
        return len(chars), char_doc

align_movie_info(type='agarwal')