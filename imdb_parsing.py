import pickle
import csv
import json
import imdb
import os
import string

DEBUGGING = False

def align_movie_info(type):
    movie_to_info = {}
    db = imdb.IMDb()
    found_id = 0
    found_dia = 0

    if type == 'walker' or type == 'w':
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

    if type == 'agarwal' or type == 'a':
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

    elif type == 'gorinski' or type == 'g':
        gorinski_path = './data/gorinski/'
        for dir in os.listdir(gorinski_path):
            if dir != '.DS_Store':
                print('Gorinski dir:', dir)
                dir_path = gorinski_path + dir + '/'

                for movie_folder in os.listdir(dir_path):
                    if movie_folder != '.DS_Store':
                        found_dia += 1
                        file_path = dir_path + movie_folder + '/script_clean.txt'
                        if DEBUGGING: print('File_path:', file_path)

                        info = {'source':'gorinski', 'title':None, 'id':None}
                        title = find_gorinski_title(movie_folder)
                        info['title'] = title
                        if DEBUGGING: print('Title:', title)

                        char_count, char_doc = parse_gorinski_chars(file_path)
                        if DEBUGGING: print('Char count:', char_count, 'Chars:', char_doc)

                        if char_count > 5:
                            imdb_id = match_id(db, title, char_doc)
                            if imdb_id is not None:
                                found_id += 1
                                info['id'] = imdb_id

                        movie_to_info[file_path] = info
                        print(found_dia, file_path, info)
            pickle.dump(movie_to_info, open('gorinski_alignments.p', 'wb'))

    print('Found {} dialog files, found {} ids'.format(found_dia, found_id))

def write_to_csv(dictionary, save_file):
    with open(save_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Title', 'Source', 'File_path', 'IMDb_id'])
        sort_by_path = sorted(dictionary.items(), key=lambda x: x[0])
        for path,info in sort_by_path:
            source = info['source']
            title = info['title']
            id = info['id']
            writer.writerow([title, source, path, id])

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

def parse_bechdel():
    bechdel = json.load(open('./data/allmovies.json', 'r'))
    id_to_info = {}
    id_len = None
    for movie in bechdel:
        imdbid = movie['imdbid']
        if id_len is None:
            id_len = len(imdbid)
        id_to_info[imdbid] = movie
    return id_to_info, id_len

def align_id_to_bechdel(alignment_csv, bechdel_dict, id_len):
    new_fname = alignment_csv.replace('.csv', '_with_bechdel.csv')
    reader = csv.DictReader(open(alignment_csv, 'r'))
    fieldnames = reader.fieldnames
    fieldnames.append('Bechdel_rating')
    fieldnames.append('Year')
    writer = csv.DictWriter(open(new_fname, 'w'), fieldnames)
    first_row = {x:x for x in fieldnames}
    writer.writerow(first_row)

    found_bechdel = 0
    total = 0
    for row in reader:
        path = row['File_path']
        # skip agarwals without good tags
        if path.startswith('./data/agarwal2015_screenplays/fail_wo_tags'):
            continue
        total += 1
        id = row['IMDb_id']
        padded_id = pad_id(id, id_len)
        row['IMDb_id'] = padded_id
        if padded_id in bechdel_dict:
            found_bechdel += 1
            row['Bechdel_rating'] = bechdel_dict[padded_id]['rating']
            row['Year'] = bechdel_dict[padded_id]['year']
        else:
            row['Bechdel_rating'] = ''
        writer.writerow(row)

    print('Found {} scores for {} movies.'.format(found_bechdel, total))

def pad_id(id, id_len):
    padding = '0' * (id_len - len(id))
    return padding + id

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

# GORINKSI FUNCTIONS
def find_gorinski_title(movie_folder):
    title = movie_folder.split('(')[0]
    title = title.strip().lower()
    return title

def parse_gorinski_chars(file_path):
    with open(file_path, 'r') as f:
        chars = set()
        max_ls = 10
        for line in f.readlines():
            leading_space = len(line) - len(line.lstrip(' '))
            if leading_space >= max_ls:
                char = line.strip().upper()
                chars.add(char)
                max_ls = max(leading_space, max_ls)
        char_doc = ' '.join(list(chars))
        return len(chars), char_doc

if __name__ == "__main__":
    # align_movie_info('g')
    # dictionary = pickle.load(open('gorinski_alignments.p', 'rb'))
    # write_to_csv(dictionary, 'gorinski_alignments.csv')

    SOURCE = 'agarwal'

    bech_dict, id_len = parse_bechdel()
    print('Bechdel scores:', len(bech_dict))
    print('ID length:', id_len)
    align_id_to_bechdel('./data/' + SOURCE + '_alignments_with_IDs.csv', bech_dict, id_len)
