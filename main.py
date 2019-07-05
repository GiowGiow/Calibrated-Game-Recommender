import pandas as pd
import numpy as np
import time
from scipy.sparse import csr_matrix
from implicit.bpr import BayesianPersonalizedRanking
from implicit.evaluation import train_test_split, precision_at_k
import matplotlib.pyplot as plt
from ItemClass import create_item_mapping
from calibrated_recommender import *

# Folder containing CSVs
DATA_FILEPATH = 'data_csv'

# Define settings
topn = 5
user_id = 46546
lmbda = 0.5

# Define columns to import
item_col = "item_id"
title_col = "name"
genre_col = "genres"

# Read Game info dataframe
df_games = pd.read_csv('{}/steam_games.csv'.format(DATA_FILEPATH), index_col="Unnamed: 0")
df_games['item_id'] = df_games.index
#print(df_games.iloc[10, 0])

# Create Item Mapping
item_mapping = create_item_mapping(df_games, item_col, title_col, genre_col)

# Read matrix item/playtime
user_items_playtime_ = '{}/user_items_playtime.csv'.format(DATA_FILEPATH)
df_matrix = pd.read_csv(user_items_playtime_)
index2item = pd.Series(list(df_matrix.columns.values), dtype="category").cat.categories

# nonlinear scaling (normalize hours into 0-1 interval for rating)
xmax = 50 * 60  # 50 hrs
df_scaled_matrix = np.tanh(df_matrix * 2 / xmax)

# compress matrix
csr_df_matrix = csr_matrix(df_scaled_matrix)
np.random.seed()

# Train
user_item_train, user_item_test = train_test_split(csr_df_matrix, train_percentage=0.8)
bpr = BayesianPersonalizedRanking(iterations=10)
bpr.fit(user_item_train.T.tocsr())

# look at some user
user_id = 13584

print(user_item_train[user_id])
interacted_ids = user_item_train[user_id].nonzero()[1]
index2item = index2item.astype('int32')

interacted_items = [item_mapping[index2item[index]] for index in interacted_ids if
                    index2item[index] in item_mapping.keys()]

# it returns the recommended index and their corresponding score
topn = 100
reco = bpr.recommend(user_id, user_item_train, N=topn)
print(reco)

# map the index to Item
reco_items = [item_mapping[index2item[index]] for index, _ in reco if index2item[index] in item_mapping.keys()]

print(reco_items)




# we can check that the probability does in fact add up to 1
# np.array(list(interacted_distr.values())).sum()
print(interacted_items)
interacted_distr = compute_genre_distr(interacted_items)
print(interacted_distr)

reco_distr = compute_genre_distr(reco_items[:topn])


# change default style figure and font size
plt.rcParams['figure.figsize'] = 10, 8
plt.rcParams['font.size'] = 12


def distr_comparison_plot(interacted_distr, reco_distr, width=0.3):
    # the value will automatically be converted to a column with the
    # column name of '0'
    interacted = pd.DataFrame.from_dict(interacted_distr, orient='index')
    reco = pd.DataFrame.from_dict(reco_distr, orient='index')
    df = interacted.join(reco, how='outer', lsuffix='_interacted')

    n = df.shape[0]
    index = np.arange(n)
    plt.barh(index, df['0_interacted'], height=width, label='interacted distr')
    plt.barh(index + width, df['0'], height=width, label='reco distr')
    plt.yticks(index, df.index)
    plt.legend(bbox_to_anchor=(1, 0.5))
    plt.title('Genre Distribution between User Historical Interaction v.s. Recommendation')
    plt.ylabel('Genre')
    plt.show()


distr_comparison_plot(interacted_distr, reco_distr)


compute_kl_divergence(interacted_distr, reco_distr)





items = generate_item_candidates(bpr, user_item_train, user_id, index2item, item_mapping)
print('number of item candidates:', len(items))


start = time.time()
calib_reco_items = calib_recommend(items, interacted_distr, topn, lmbda=0.99)
print(items)
elapsed = time.time() - start
print('elapsed: ', elapsed)
print(calib_reco_items)
calib_reco_items = calib_reco_items[:topn]

print(len(calib_reco_items))
calib_reco_distr = compute_genre_distr(calib_reco_items[:topn])
print(calib_reco_distr)
calib_reco_kl_div = compute_kl_divergence(interacted_distr, calib_reco_distr)
reco_kl_div = compute_kl_divergence(interacted_distr, reco_distr)
print('\noriginal reco kl-divergence score:', reco_kl_div)
print('calibrated reco kl-divergence score:', calib_reco_kl_div)

print(calib_reco_distr)
distr_comparison_plot(interacted_distr, calib_reco_distr)


reco_precision = precision(user_item_test, user_id, reco_items, index2item)
calib_reco_precision = precision(user_item_test, user_id, calib_reco_items, index2item)
print('original reco precision score:', reco_precision)
print('calibrated reco precision score:', calib_reco_precision)

print(items)
calib_reco_items = calib_recommend(items, interacted_distr, topn, lmbda=0.5)
elapsed = time.time() - start

calib_reco_distr = compute_genre_distr(calib_reco_items)
calib_reco_kl_div = compute_kl_divergence(interacted_distr, calib_reco_distr)
calib_reco_precision = precision(user_item_test, user_id, calib_reco_items, index2item)
print('calibrated reco kl-divergence score:', calib_reco_kl_div)
print('calibrated reco precision score:', calib_reco_precision)

calib_reco_distr = compute_genre_distr(calib_reco_items)
distr_comparison_plot(interacted_distr, calib_reco_distr)


reco = bpr.recommend(user_id, user_item_train, N=topn)
reco_items = [item_mapping[index2item[index]] for index, _ in reco if index2item[index] in item_mapping.keys()]
print(reco_items)

interacted_ids = user_item_train[user_id].nonzero()[1]
interacted_items = [item_mapping[index2item[index]] for index in interacted_ids if
                    index2item[index] in item_mapping.keys()]
interacted_distr = compute_genre_distr(interacted_items)
print(interacted_distr)

items = generate_item_candidates(bpr, user_item_train, user_id, index2item, item_mapping)
print(len(items))
calib_reco_items = calib_recommend(items, interacted_distr, topn, lmbda)
print(calib_reco_items)
calib_reco_distr = compute_genre_distr(calib_reco_items)

calib_reco_kl_div = compute_kl_divergence(interacted_distr, calib_reco_distr)
calib_reco_precision = precision(user_item_test, user_id, calib_reco_items, index2item)
print('calibrated reco kl-divergence score:', calib_reco_kl_div)
print('calibrated reco precision score:', calib_reco_precision)
distr_comparison_plot(interacted_distr, calib_reco_distr)

reco_kl_div = compute_kl_divergence(interacted_distr, reco_distr)
reco_precision = precision(user_item_test, user_id, reco_items, index2item)
print('original reco kl-divergence score:', reco_kl_div)
print('original reco precision score:', reco_precision)
distr_comparison_plot(interacted_distr, reco_distr)