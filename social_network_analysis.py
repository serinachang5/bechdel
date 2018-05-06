import align_gender as ag
from get_scene_boundaries import get_boundaries_agarwal, get_boundaries_gorinski
from util import get_data, check_distribution, get_char_to_lines
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


class SNA:
    def __init__(self, verbose = False):
        self.data = get_data()
        self.verbose = verbose

    def transform_into_feats(self, movie_id, sna_mode, min_lines, cent_modes):
        assert(len(cent_modes) > 0)
        G, char_dict = self.get_network(movie_id, mode=sna_mode, min_lines=min_lines)

        feat_len = len(cent_modes) * 4 + 2
        feats = np.zeros(feat_len)

        if len(G.nodes()) == 0:
            return feats

        # calculate centrality stats
        for i, cent_md in enumerate(cent_modes):
            start = i*4
            stats = self.get_gender_cent_stats(G, char_dict, mode=cent_md)
            feats[start:start+4] = stats

        # calculate additional features
        m_to_f, f_to_f = self.get_connected_to_woman_stats(G, char_dict)
        feats[len(cent_modes) * 4] = m_to_f
        feats[len(cent_modes) * 4 + 1] = f_to_f

        return feats

    def get_network(self, movie_id, mode, min_lines):
        if movie_id not in self.data:
            raise ValueError('Cannot find this movie id in database.')

        assert(mode == 'overlap' or mode == 'consecutive')

        movie_info = self.data[movie_id]
        if self.verbose: print('Found movie info for', movie_info[0])
        path = movie_info[4]
        char_dict = movie_info[5]
        G = self._build_network(path, char_dict, mode=mode, min_lines=min_lines)
        return G, char_dict

    def plot_network(self, G, char_dict, movie_title, mode, min_lines):
        cmap = []
        chars = G.nodes()
        for char in chars:
            gen, score, var = char_dict[char]
            if score != 'None' and float(score) > .5:
                cmap.append('#FFB6C1') # light pink
            elif score != 'None':
                cmap.append('#87CEFA') # sky blue
            else:
                cmap.append('#D8DCD6') # light grey

        nx.draw_networkx(G, node_size=200, node_color=cmap)
        plt.title('Social network from ' + movie_title + ' (mode={}, min_lines={})'.format(mode, min_lines))
        plt.show()

    def get_centralities(self, G, mode='degree'):
        assert(mode == 'degree' or mode == 'btwn' or mode == 'close' or mode == 'eigen')
        if mode == 'degree':
            return nx.degree_centrality(G)
        elif mode == 'btwn':
            return nx.betweenness_centrality(G)
        elif mode == 'close':
            return nx.closeness_centrality(G)
        else:
            return nx.eigenvector_centrality(G)

    def get_gender_cent_stats(self, G, char_dict, mode='degree'):
        cents = self.get_centralities(G, mode=mode)
        f_cent = []
        m_cent = []
        for char, cent in cents.items():
            gen_score = char_dict[char][1]
            if gen_score != 'None':
                if float(gen_score) > .5:
                    f_cent.append(cent)
                else:
                    m_cent.append(cent)
        f_avg = 0 if len(f_cent) == 0 else np.mean(f_cent)
        f_sum = np.sum(f_cent)
        m_avg = 0 if len(m_cent) == 0 else np.mean(m_cent)
        m_sum = np.sum(m_cent)

        return [f_avg, f_sum, m_avg, m_sum]

    def get_connected_to_woman_stats(self, G, char_dict):
        m_to_f = set()  # all men connected to a woman
        f_to_f = set()  # all women connected to a woman
        for char in G.nodes():
            score1 = char_dict[char][1]
            if score1 != 'None' and float(score1) > .5:
                neighbors = G.neighbors(char)
                for neigh in neighbors:
                    score2 = char_dict[neigh][1]
                    if score2 != 'None':
                        if float(score2) > .5:
                            f_to_f.add(neigh)
                        else:
                            m_to_f.add(neigh)
        return len(m_to_f), len(f_to_f)

    # make a social network of the characters who have at least <min_lines> lines
    def _build_network(self, path, char_dict, mode, min_lines = 5):
        if 'agarwal' in path:
            source = 'agarwal'
            scenes = get_boundaries_agarwal(path)
        else:
            source = 'gorinski'
            scenes = get_boundaries_gorinski(path)

        var2info = ag.get_variant_as_key(char_dict)
        char2lines = get_char_to_lines(path, char_dict)

        G = nx.Graph()
        for i,scene in enumerate(scenes):
            cdl = ag.get_char_diag_list(scene, var2info, source)

            if mode == 'overlap':
                # connect all characters in this scene
                char_tuples = set([cd[0] for cd in cdl])
                char_tuples = sorted(list(char_tuples), key=lambda x:x[0])  # sort by name
                for i,(cname1, _, _) in enumerate(char_tuples):
                    if len(char2lines[cname1]) >= min_lines:
                        for j,(cname2, gen2, score2) in enumerate(char_tuples[i+1:]):
                            if len(char2lines[cname2]) >= min_lines:
                                G.add_edge(cname1, cname2)
            else:
                # only connect characters who speak consecutively
                for i in range(len(cdl)-1):
                    cname1, _, _ = cdl[i][0]
                    if len(char2lines[cname1]) >= min_lines:
                        cname2, _, _ = cdl[i+1][0]
                        if len(char2lines[cname2]) >= min_lines:
                            G.add_edge(cname1, cname2)
        return G

def test_SNA():
    sna = SNA()
    id = '0250494'
    mode = 'overlap'
    min_lines = 5
    G, char_dict = sna.get_network(id, mode=mode, min_lines=min_lines)
    for cent in ['degree', 'btwn', 'close', 'eigen']:
        print('\nCentrality type:', cent)
        print(sna.get_gender_cent_stats(G, char_dict, mode=cent))

    sna.plot_network(G, char_dict, movie_title='10 THINGS', mode=mode, min_lines=min_lines)

if __name__ == "__main__":
    test_SNA()