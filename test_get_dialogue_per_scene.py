import glob
import nltk
import os
import pickle
import string

from align_gender import parse_by_gender_file
from align_gender import get_variant_as_key
from align_gender import get_char_diag_list

def zip_contents(source):

    if source == "agarwal":

        movie_gender_agarwal = os.listdir("data/movie_by_gender/agarwal")
        screenplays_agarwal = glob.glob("data/agarwal2015_screenplays/*/*")
        screenplays_agarwal = [name for name in screenplays_agarwal if name[-4:] != ".txt"]

        zipped = []
        for mg in movie_gender_agarwal:
            for s in screenplays_agarwal:
                sp = s.split("_")
                sp = sp[-2]
                mg_space = mg[:-9].split("_")
                mg_space_join = " ".join(mg_space)
                if sp == mg_space_join:
                    zipped.append((s, "data/movie_by_gender/agarwal/"+mg))

    if source == "gorinski":

        movie_gender_gorinski = os.listdir("data/movie_by_gender/gorinski")
        screenplays_gorinski = glob.glob("data/gorinski/*/*/*")
        screenplays_gorinski = [name for name in screenplays_gorinski if name[-4:] != ".txt"]

        zipped = []
        for mg in movie_gender_gorinski:
            for s in screenplays_gorinski:
                sp = s.split("/")[-1][:-7]
                mg_space = mg[:-9].split("_")
                mg_space_join = " ".join(mg_space)
                if sp  == mg_space_join:
                    zipped.append((s,"data/movie_by_gender/gorinski/"+mg))

    return zipped

def get_dialogue_per_scene(files, source):

    nested_scenes = {}

    for f in files:
        char_dict = parse_by_gender_file(f[1])
        char_dict = char_dict[-1][-1]
        var = get_variant_as_key(char_dict)

        nested_scenes[f[1]] = {}

        scenes = os.listdir(f[0])

        for s in scenes:
            name = f[0]+"/"+s
            with open(name) as fp:
                c = fp.readlines()
                k = get_char_diag_list(c, var, source)
                nested_scenes[f[1]][name] = k

    return nested_scenes

def get_dialogue_by_interactions(dialogue_dict):

    mm_exchanges = {}
    ff_exchanges = {}
    fm_exchanges = {}

    for key, val in dialogue_dict.items():
        mm_exchanges[key] = {}
        ff_exchanges[key] = {}
        fm_exchanges[key] = {}
        for inner_key, inner_val in val.items():
            interactions = set()
            for i in inner_val:
                interactions.add(i[0])
            if len(interactions) == 2:
                interactions_lst = list(interactions)
                if interactions_lst[0][1] == "M" and interactions_lst[1][1] == "M":
                    mm_exchanges[key][inner_key] = inner_val
                if interactions_lst[0][1] == "F" and interactions_lst[1][1] == "F":
                    ff_exchanges[key][inner_key] = inner_val
                if interactions_lst[0][1] == "M" and interactions_lst[1][1] == "F" or \
                interactions_lst[0][1] == "F" and interactions_lst[1][1] == "M":
                    fm_exchanges[key][inner_key] = inner_val

    return (mm_exchanges, ff_exchanges, fm_exchanges)

def get_num_interactions(interaction_dict):

    count = 0
    for key, val in interaction_dict.items():
        for inner_key, inner_val in val.items():
            count += 1
    return count

def get_power_agency_by_interaction_type_mm_ff(interaction_type, frames_dict):

    agency_pos_count = 0
    agency_neg_count = 0
    agency_equal_count = 0

    power_agent_count = 0
    power_theme_count = 0
    power_equal_count = 0

    count = 0
    for key, val in interaction_type.items():
        for inner_key, inner_val in val.items():
            for diag in inner_val:
                combine_diag = ",".join(diag[1])
                combine_diag = combine_diag.split(",")
                combine_diag = " ".join(combine_diag)
                combine_diag = "".join(l for l in combine_diag if l not in string.punctuation)
                tokens = nltk.word_tokenize(combine_diag)
                if tokens != []:
                    for w in tokens:
                        try:
                            if any(w in verb for verb in frames_dict):
                                frames_val = frames_dict[w]

                                if frames_val[0] == "agency_pos":
                                    agency_pos_count += 1
                                if frames_val[0] == "agency_neg":
                                    agency_neg_count += 1
                                if frames_val[0] == "agency_equal":
                                    agency_equal_count += 1

                                if frames_val[1] == "power_agent":
                                    power_agent_count += 1
                                if frames_val[1] == "power_theme":
                                    power_theme_count += 1
                                if frames_val[1] == "power_equal":
                                    power_equal_count += 1
                        except:
                            count +=1 

    print("agency_pos: % s \n" % (agency_pos_count), \
        "agency_neg: % s \n" % (agency_neg_count), \
        "agency_equal: % s \n" % (agency_equal_count), \
        "power_agent: % s \n" % (power_agent_count), \
        "power_theme: % s \n" % (power_theme_count), \
        "power_equal: % s \n" % (power_equal_count), \
        )

def get_power_agency_by_interaction_type_mm_ff_by_movie(interaction_type, frames_dict):

    count = 0
    for key, val in interaction_type.items():

        agency_pos_count = 0
        agency_neg_count = 0
        agency_equal_count = 0

        power_agent_count = 0
        power_theme_count = 0
        power_equal_count = 0

        for inner_key, inner_val in val.items():
            for diag in inner_val:
                combine_diag = ",".join(diag[1])
                combine_diag = combine_diag.split(",")
                combine_diag = " ".join(combine_diag)
                combine_diag = "".join(l for l in combine_diag if l not in string.punctuation)
                tokens = nltk.word_tokenize(combine_diag)
                if tokens != []:
                    for w in tokens:
                        try:
                            if any(w in verb for verb in frames_dict):
                                frames_val = frames_dict[w]

                                if frames_val[0] == "agency_pos":
                                    agency_pos_count += 1
                                if frames_val[0] == "agency_neg":
                                    agency_neg_count += 1
                                if frames_val[0] == "agency_equal":
                                    agency_equal_count += 1

                                if frames_val[1] == "power_agent":
                                    power_agent_count += 1
                                if frames_val[1] == "power_theme":
                                    power_theme_count += 1
                                if frames_val[1] == "power_equal":
                                    power_equal_count += 1
                        except:
                            count +=1 

        print(key)
        print("agency_pos: % s \n" % (agency_pos_count), \
            "agency_neg: % s \n" % (agency_neg_count), \
            "agency_equal: % s \n" % (agency_equal_count), \
            "power_agent: % s \n" % (power_agent_count), \
            "power_theme: % s \n" % (power_theme_count), \
            "power_equal: % s \n" % (power_equal_count), \
            )
        
        

def get_power_agency_by_interaction_type_fm(interaction_type, frames_dict):

    agency_pos_count_m = 0
    agency_neg_count_m = 0
    agency_equal_count_m = 0

    power_agent_count_m = 0
    power_theme_count_m = 0
    power_equal_count_m = 0

    agency_pos_count_f = 0
    agency_neg_count_f = 0
    agency_equal_count_f = 0

    power_agent_count_f = 0
    power_theme_count_f = 0
    power_equal_count_f = 0

    count = 0
    for key, val in interaction_type.items():
        for inner_key, inner_val in val.items():
            for diag in inner_val:
                combine_diag = ",".join(diag[1])
                combine_diag = combine_diag.split(",")
                combine_diag = " ".join(combine_diag)
                combine_diag = "".join(l for l in combine_diag if l not in string.punctuation)
                tokens = nltk.word_tokenize(combine_diag)
                if tokens != []:
                    for w in tokens:
                        try:
                            if any(w in verb for verb in frames_dict):
                                frames_val = frames_dict[w]

                                if diag[0][1] == "F":
                                    if frames_val[0] == "agency_pos":
                                        agency_pos_count_f += 1
                                    if frames_val[0] == "agency_neg":
                                     agency_neg_count_f += 1
                                    if frames_val[0] == "agency_equal":
                                        agency_equal_count_f += 1

                                    if frames_val[1] == "power_agent":
                                        power_agent_count_f += 1
                                    if frames_val[1] == "power_theme":
                                        power_theme_count_f += 1
                                    if frames_val[1] == "power_equal":
                                        power_equal_count_f += 1

                                if diag[0][1] == "M":
                                    if frames_val[0] == "agency_pos":
                                        agency_pos_count_m += 1
                                    if frames_val[0] == "agency_neg":
                                     agency_neg_count_m += 1
                                    if frames_val[0] == "agency_equal":
                                        agency_equal_count_m += 1

                                    if frames_val[1] == "power_agent":
                                        power_agent_count_m += 1
                                    if frames_val[1] == "power_theme":
                                        power_theme_count_m += 1
                                    if frames_val[1] == "power_equal":
                                        power_equal_count_m += 1

                        except:
                            count +=1 

    print("agency_pos_f: % s \n" % (agency_pos_count_f), \
        "agency_neg_f: % s \n" % (agency_neg_count_f), \
        "agency_equal_f: % s \n" % (agency_equal_count_f), \
        "power_agent_f: % s \n" % (power_agent_count_f), \
        "power_theme_f: % s \n" % (power_theme_count_f), \
        "power_equal_f: % s \n \n" % (power_equal_count_f), \
        "agency_pos_m: % s \n" % (agency_neg_count_m), \
        "agency_neg_m: % s \n" % (agency_neg_count_m), \
        "agency_equal_m: % s \n" % (agency_equal_count_m), \
        "power_agent_m: % s \n" % (power_agent_count_m), \
        "power_theme_m: % s \n" % (power_theme_count_m), \
        "power_equal_m: % s \n" % (power_equal_count_m), \
        )

if __name__ == "__main__":

    agarwal_files = zip_contents("agarwal")
    gorinski_files = zip_contents("gorinski")

    agarwal_dialogues = get_dialogue_per_scene(agarwal_files, "agarwal")
    gorinski_dialogues = get_dialogue_per_scene(gorinski_files, "gorinski")

    agarwal_interactions = get_dialogue_by_interactions(agarwal_dialogues)
    gorinski_interactions = get_dialogue_by_interactions(gorinski_dialogues)


    agarwal_interactions_mm = agarwal_interactions[0]
    agarwal_interactions_ff = agarwal_interactions[1]
    agarwal_interactions_fm = agarwal_interactions[2]

    gorinski_interactions_mm = gorinski_interactions[0]
    gorinski_interactions_ff = gorinski_interactions[1]
    gorinski_interactions_fm = gorinski_interactions[2]

    with open('frames.pickle', 'rb') as handle:
        frames = pickle.load(handle)

    agarwal_num_mm = get_num_interactions(agarwal_interactions_mm)
    agarwal_num_ff = get_num_interactions(agarwal_interactions_ff)
    agarwal_num_fm = get_num_interactions(agarwal_interactions_fm)

    print(agarwal_num_mm)
    print(agarwal_num_ff)
    print(agarwal_num_fm)

    get_power_agency_by_interaction_type_mm_ff(agarwal_interactions_mm, frames)
    get_power_agency_by_interaction_type_mm_ff(agarwal_interactions_ff, frames)
    get_power_agency_by_interaction_type_fm(agarwal_interactions_fm, frames)

    gorinski_num_mm = get_num_interactions(gorinski_interactions_mm)
    gorinski_num_ff = get_num_interactions(gorinski_interactions_ff)
    gorinski_num_fm = get_num_interactions(gorinski_interactions_fm)

    print(gorinski_num_mm)
    print(gorinski_num_ff)
    print(gorinski_num_fm)

    get_power_agency_by_interaction_type_mm_ff(gorinski_interactions_mm, frames)
    get_power_agency_by_interaction_type_mm_ff(gorinski_interactions_ff, frames)
    get_power_agency_by_interaction_type_fm(gorinski_interactions_fm, frames)


    # Normalize counts
    # divide mm_counts by overall words in general mm_counts< >
    # divide ff_counts by overall words in general ff_counts< >

    # divide mf_counts by overall words in general mm_words // ff_wordds

    # get_power_agency_by_interaction_type_mm_ff_by_movie(agarwal_interactions_ff, frames)
    # stem words using porter stemmer




                
                
















