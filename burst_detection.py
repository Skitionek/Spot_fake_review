import numpy as np
import datetime

read_file = np.loadtxt('metadata', dtype=np.dtype("i4, i4, f4, i4, S10"))
read_file_short = read_file[0:1000]

data_list = []
for row in read_file_short:
    data_list.append([row[0], row[1], datetime.datetime.strptime(row[4], "%Y-%m-%d").date()])
data = np.asarray(data_list) # convert list of tuples to array
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
for i in range (0,len(item_data)):
    new_list = []
    for j in range(0,len(item_data[i])):
        new_list.append((item_data[i][j][2] - item_data[i][0][2]).days / 14)
    item_data_bin.append(new_list)
data = np.asarray(new_list)
   # item_data_bin = np.append(item_data, np.split(item_data, np.where(np.diff(item_data[i][:,2]))[0]+BSize))
[[x,item_data_bin[0].count(x)] for x in range(np.min(item_data_bin[0]),np.max(item_data_bin[0])+1)]


# one product p has m reviews with m dates, t_1...t_m as well as duration
# dur is calculated from t_m - t_1
# average number of reviews within each bin should be calculated







