import pandas as pd

# load datasets
agarwal_ff = pd.read_csv("agarwal_ff.csv")
gorinski_ff = pd.read_csv("gorinski_ff.csv")

agarwal_mm = pd.read_csv("agarwal_mm.csv")
gorinski_mm = pd.read_csv("gorinski_mm.csv")

agarwal_fm = pd.read_csv("agarwal_fm.csv")
gorinski_fm = pd.read_csv("gorinski_fm.csv")


# merge datasets based on conversation pairs
ff_all = pd.merge(agarwal_ff, gorinski_ff, on="movie_id")
mm_all = pd.merge(agarwal_mm, gorinski_mm, on="movie_id")
fm_all = pd.merge(agarwal_fm, gorinski_fm, on="movie_id")

# retrieve average for each frame category for ff conversation pairs
ff_all.groupby('bechdel_score_x', as_index=False)['agency_pos_x'].mean()
ff_all.groupby('bechdel_score_x', as_index=False)['agency_neg_x'].mean()
ff_all.groupby('bechdel_score_x', as_index=False)['agency_equal_x'].mean()
ff_all.groupby('bechdel_score_x', as_index=False)['power_agency_x'].mean()
ff_all.groupby('bechdel_score_x', as_index=False)['power_theme_x'].mean()
ff_all.groupby('bechdel_score_x', as_index=False)['power_equal_x'].mean()

# count number of movies associated with each bechdel score for ff conversation pairs
fm_all.groupby('bechdel_score_x').count()

# retrieve average for each frame category for mm conversation pairs
mm_all.groupby('bechdel_score_x', as_index=False)['agency_pos_x'].mean()
mm_all.groupby('bechdel_score_x', as_index=False)['agency_neg_x'].mean()
mm_all.groupby('bechdel_score_x', as_index=False)['agency_equal_x'].mean()
mm_all.groupby('bechdel_score_x', as_index=False)['power_agency_x'].mean()
mm_all.groupby('bechdel_score_x', as_index=False)['power_theme_x'].mean()
mm_all.groupby('bechdel_score_x', as_index=False)['power_equal_x'].mean()

# count number of movies associated with each bechdel score for mm conversation pairs
mm_all.groupby('bechdel_score_x').count()

# retrieve average for each frame category for fm conversation pairs
fm_all.groupby('bechdel_score_x', as_index=False)['agency_pos_f_x'].mean()
fm_all.groupby('bechdel_score_x', as_index=False)['agency_pos_m_x'].mean()
fm_all.groupby('bechdel_score_x', as_index=False)['agency_neg_f_x'].mean()
fm_all.groupby('bechdel_score_x', as_index=False)['agency_neg_m_x'].mean()
fm_all.groupby('bechdel_score_x', as_index=False)['agency_equal_f_x'].mean()
fm_all.groupby('bechdel_score_x', as_index=False)['agency_equal_m_x'].mean()
fm_all.groupby('bechdel_score_x', as_index=False)['power_agency_f_x'].mean()
fm_all.groupby('bechdel_score_x', as_index=False)['power_agency_m_x'].mean()
fm_all.groupby('bechdel_score_x', as_index=False)['power_theme_f_x'].mean()
fm_all.groupby('bechdel_score_x', as_index=False)['power_theme_m_x'].mean()
fm_all.groupby('bechdel_score_x', as_index=False)['power_equal_f_x'].mean()
fm_all.groupby('bechdel_score_x', as_index=False)['power_equal_m_x'].mean()

# count number of movies associated with each bechdel score for fm conversation pairs
fm_all.groupby('bechdel_score_x').count()

# NOTE: In the final paper, average scores were normalized [0,1] by subtracting the minimum
# in each conversation category from each element and dividing by the conversation category's range

