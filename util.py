import align_gender as ag
import os
import numpy as np
import random
from get_scene_boundaries import get_boundaries_agarwal, get_boundaries_gorinski

def get_data(source = 'combined'):
    path = './movie_by_gender/'

    if source == 'agarwal' or source == 'gorinski':
        id_to_info = {}
        for fname in os.listdir(path + source + '/'):
            if not fname.startswith('.DS_Store'):
                id, info = ag.parse_by_gender_file(path + source + '/' + fname)
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

# get char mapped to all their lines
def get_char_to_lines(path, char_dict):
    if 'agarwal' in path:
        source = 'agarwal'
        scenes = get_boundaries_agarwal(path)
    else:
        source = 'gorinski'
        scenes = get_boundaries_gorinski(path)

    var2info = ag.get_variant_as_key(char_dict)

    char_to_lines = {}
    for scene in scenes:
        cdl = ag.get_char_diag_list(scene, var2info, source)
        for (root, gen, score), diag in cdl:
            if root in char_to_lines:
                char_to_lines[root].append(' '.join(diag))
            else:
                char_to_lines[root] = [' '.join(diag)]
    return char_to_lines

'''ERROR ANALYSES'''
# checking how we normalize from variant to root
def error_analysis_of_char_names():
    data = get_data()
    data = list(data.values())
    for i,info in enumerate(random.sample(data, 10)):
        print('Sample',i)
        char_dict = info[5]
        var_dict = ag.get_variant_as_key(char_dict)
        for var in var_dict:
            if len(var.split()) > 1:  # more than one word
                if '\'S' in var:
                    print(var)

# checking 'important' characters are missing genders
def error_analysis_of_important_chars():
    data = get_data()
    data = list(data.values())
    non_imp_chars = non_imp_chars_no_gen = imp_chars = imp_chars_no_gen = 0
    missing = []
    for i,info in enumerate(data):
        title = info[0]
        year = info[1]
        path = info[4]
        char_dict = info[5]
        try:
            char2lines = get_char_to_lines(path, char_dict)
        except ValueError:
            print(path)
            print(char_dict)
            return
        for char,lines in char2lines.items():
            if len(lines) > 100:
                imp_chars += 1
                if char_dict[char][1] == 'None':
                    imp_chars_no_gen += 1
                    if 'agarwal' in path:
                        source = 'Agarwal'
                    else:
                        source = 'Gorinski'
                    missing.append((title, year, source, char, len(lines)))
            else:
                non_imp_chars += 1
                if char_dict[char][1] == 'None':
                    non_imp_chars_no_gen += 1
    report = ''
    report += 'Number of important chars: {}\n'.format(imp_chars)
    report += 'Number of important chars wo gender: {} ({}%)\n'.format(imp_chars_no_gen, round(imp_chars_no_gen * 1.0/imp_chars, 5))
    report += 'Number of non-important chars: {}\n'.format(non_imp_chars)
    report += 'Number of non-important chars wo gender: {} ({}%)\n'.format(non_imp_chars_no_gen, round(non_imp_chars_no_gen * 1.0/non_imp_chars, 5))
    print(report)

    missing = sorted(missing, key=lambda x: x[4], reverse=True)  # order by most lines to least (priority)

    with open('./output/missing_imp_chars_gen.txt', 'w') as f:
        f.write(report)
        f.write('\n')
        for title, year, source, char, num_lines in missing:
            f.write('{} {} (from {}), {}, {} lines\n'.format(title, year, source, char, num_lines))

def testing_changes():
    data = get_data()
    sample_changed = data['0093409']
    print(sample_changed)
    title = sample_changed[0]
    year = sample_changed[1]
    path = sample_changed[4]
    char_dict = sample_changed[5]
    char2lines = get_char_to_lines(path, char_dict)
    print(char2lines)

if __name__ == "__main__":
    # combined_data = get_data()
    # print(check_distribution(combined_data))
    error_analysis_of_important_chars()
