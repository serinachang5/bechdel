import glob
import os

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
                    zipped.append((s, mg))

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
                    zipped.append((s,mg))

    return zipped

if __name__ == "__main__":
    # char_dict = parse_by_gender_file("data/movie_by_gender/agarwal/10_things_i_hate_about_you_1999.txt")
    # char_dict = char_dict[-1][-1]
    # var = get_variant_as_key(char_dict)

    # with open("data/agarwal2015_screenplays/pass/51e2fe98144cce5b901a9617_10_Things_I_Hate_About_You_10 things i hate about you_scenes/10.txt") as fp:
    #     contents = fp.readlines()
    #     k = get_char_diag_list(contents, var, "agarwal")
    #     print(k)

    print(zip_contents("agarwal"))




