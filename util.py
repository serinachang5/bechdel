import os
import numpy as np

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

def get_data(source = 'combined'):
    path = './movie_by_gender/'

    if source == 'agarwal' or source == 'gorinski':
        id_to_info = {}
        for fname in os.listdir(path + source + '/'):
            id, info = parse_by_gender_file(path + source + '/' + fname)
            id_to_info[id] = info
        return id_to_info

    elif source == 'combined':
        aga = get_data(source='agarwal')
        gor =  get_data(source='gorinski')
        combined = aga
        overlap = 0
        for id in gor:
            if id not in combined:
                combined[id] = gor[id]
            else:
                overlap += 1
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

def get_variant_as_key(char_dict):
    var2info = {}
    for char,(gen, score, variants) in char_dict.items():
        for v in variants:
            var2info[v] = (char, gen, score)
    return var2info

if __name__ == "__main__":
    combined_data = get_data()
    check_distribution(combined_data)