import numpy as np
import datetime
from scipy.stats import gaussian_kde
import peakutils as pu
from scipy.signal import argrelextrema

def kde_scipy(x, x_grid, bandwidth=0.2):
    """Kernel Density Estimation with Scipy"""
    # Note that scipy weights its bandwidth by the covariance of the
    # input data.  To make the results comparable to the other methods,
    # we divide the bandwidth by the sample standard deviation here.
    kde = gaussian_kde(x, bw_method=bandwidth / x.std(ddof=1))
    return kde.evaluate(x_grid)

read_file = np.loadtxt('data/metadata.txt', dtype=np.dtype("i4, i4, f4, i4, S10"))
read_file_short = read_file

data_list = []
for row in read_file_short:
    data_list.append([row[0], row[1], datetime.datetime.strptime(row[4].decode("utf-8"), "%Y-%m-%d").date()])
data = np.asarray(data_list) # convert list of tuples to array
data = data[data[:, 1].argsort()]
items = set(data[:,1])

item_data = np.empty(0)
item_data = np.append(item_data, np.split(data, np.where(np.diff(data[:,1]))[0]+1))
for i in range(0,len(item_data)):
    item_data[i] = item_data[i][item_data[i][:,2].argsort()]
durations = np.empty(0)
for i in range(0,len(items)):   # calculate durations from t_m - t_1 (in python t_0 - t_m-1 because of sorting and indexing)
    m = len(item_data[i])
    durations = np.append(durations, item_data[i][m-1][2]-item_data[i][0][2])
BSize = datetime.timedelta(days=14) # bin size for reviews

item_data_bin = []
X = []
X_grid = []
for i in range (0,len(item_data)):
    new_list = []
    for j in range(0,len(item_data[i])):
        new_list.append(int((item_data[i][j][2] - item_data[i][0][2]).days / 14))
    item_data_bin.append(new_list)
    x = np.asarray(new_list)
    X.append(x)
    x_grid = np.arange(0,np.max(x)+1)
    X_grid.append(x_grid)

bins_in_bursts = []
for k in range(0,len(X)):
    if len(X_grid[k])>1:
        kde_result = kde_scipy(X[k], X_grid[k], bandwidth=2)
        indices = np.asarray(argrelextrema(kde_result, np.greater))[0]
        reviews_per_bin = [item_data_bin[k].count(x) for x in range(np.min(item_data_bin[k]),np.max(item_data_bin[k])+1)]
        average_bin_reviews = np.average(reviews_per_bin)

        indices_after_threshold = []
        for i in range(0,len(indices)):
            if reviews_per_bin[indices[i]] > average_bin_reviews:
                if reviews_per_bin[indices[i]] > 1:
                    indices_after_threshold.append(indices[i])
        bins_in_burst = set()
        for i in range(0,len(indices_after_threshold)):
            j = 0
            while indices_after_threshold[i]+j < len(reviews_per_bin) and reviews_per_bin[indices_after_threshold[i]+j] > average_bin_reviews and reviews_per_bin[indices_after_threshold[i]+j] > 1:
                bins_in_burst = bins_in_burst.union([indices_after_threshold[i]+j])
                j = j+1
            j = 0
            while indices_after_threshold[i] - j >= 0 and reviews_per_bin[indices_after_threshold[i] - j] > average_bin_reviews and reviews_per_bin[indices_after_threshold[i]-j] > 1:
                bins_in_burst = bins_in_burst.union([indices_after_threshold[i] - j])
                j = j+1
        bins_in_bursts.append(bins_in_burst)
    else:
        bins_in_bursts.append(set())

user_prod = []
for row in read_file_short:
    user_prod.append([row[0], row[1]])
print("putting together user_prod_inburst...")
user_prod_inburst = []
for i in range(0,len(user_prod)):
    user_id = user_prod[i][0]
    product_id = user_prod[i][1]
    row_0 = [row[0] for row in item_data[product_id]]
    bin_value = item_data_bin[product_id][row_0.index(user_id)]
    is_in_burst = bin_value in bins_in_bursts[product_id]
    user_prod_inburst.append([user_id,product_id, is_in_burst])

user_prod_inburst = np.asarray(user_prod_inburst)
user_prod_inburst = user_prod_inburst[user_prod_inburst[:, 0].argsort()]
np.save("custom_data/user_prod_inburst",user_prod_inburst)









