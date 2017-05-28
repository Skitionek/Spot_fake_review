import numpy as np
import datetime

def search(lst, target):
  min = 0
  max = len(lst)-1
  avg = int((min+max)/2)
  while (min < max):
    if (lst[avg] == target):
      return avg
    elif (lst[avg] < target):
      return avg + 1 + search(lst[avg+1:], target)
    else:
      return search(lst[:avg], target)
  return avg

user_prod_inburst_date = np.asarray(np.load("custom_data/user_prod_inburst_date.npy"))
user_set = set(user_prod_inburst_date[:,0])

user_dates = []
for i in range(0, len(user_prod_inburst_date)): # extract review dates for all users with reviews in bursts
    user_id = user_prod_inburst_date[i][0]
    time = user_prod_inburst_date[i][3]
    if user_prod_inburst_date[i][2] == True:
        user_dates.append([user_id, time])
user_dates = np.asarray(user_dates)
user_dates = user_dates[user_dates[:, 0].argsort()]
user_dates_split = np.empty(0)
user_dates_split = np.append(user_dates_split, np.split(user_dates, np.where(np.diff(user_dates[:,0]))[0]+1))
burst_user_set = set(user_dates[:,0])

dist_lambda = 61
dist = 0

rb_values = []
for i in range(0,len(user_dates_split)):
    user_dates_split[i] = user_dates_split[i][user_dates_split[i][:, 1].argsort()]
    l = len(user_dates_split[i])
    rb = 0
    if l > 1:
        dist = (user_dates_split[i][l - 1][1] - user_dates_split[i][0][1]).days
        if dist > dist_lambda:
            rb = 0
        else:
            rb = 1 - (dist / dist_lambda)
    rb_values.append([user_dates_split[i][0][0], dist, rb, l])

rb_array = np.asarray(rb_values)
rb_user_list = rb_array[:,0].tolist()

RB = []
for user in user_set:
    if user in burst_user_set:
        index = search(rb_user_list,user)
        RB.append([user, rb_values[index][2]])
    else:
        RB.append([user,0])
np.save("custom_data/RB", RB)
