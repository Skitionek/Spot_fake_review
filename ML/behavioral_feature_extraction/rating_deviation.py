import numpy as np

def average_second_column(array):
    averaged_array = []
    for i in range(0, len(array)):
        averaged_array.append([array[i][0], np.average(array[i][1])])
    return averaged_array

def sortAndCollapse2dArray(array):
    array_sorted = array[array[:, 0].argsort()]
    user_and_ratings = []
    last_index=0
    for i in range(1,len(array_sorted)):
        if array_sorted[i][0] != array_sorted[last_index][0]:
            newRow = []
            newRow.append(array_sorted[last_index][0])
            newRow.append(array_sorted[last_index:i,1:])
            user_and_ratings.append(newRow)
            last_index = i
            continue
    newRow = []
    newRow.append(array_sorted[last_index][0])
    newRow.append(array_sorted[last_index:,1:])
    user_and_ratings.append(newRow)
    return user_and_ratings

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

file = np.loadtxt('data/metadata.txt', dtype=np.dtype("i4, i4, f4, i4, S10"))

file_array = np.asarray([[row[0],row[1],row[2],row[3]] for row in file])

user_ratings_products = file_array[:,[0,2,1]]
user_ratings_products_collapsed = sortAndCollapse2dArray(user_ratings_products)

file_short_users_ratings = file_array[:,[0,2]]
user_and_ratings = sortAndCollapse2dArray(file_short_users_ratings)

products_ratings = file_array[:,[1,2]]
product_ratings_collapsed = sortAndCollapse2dArray(products_ratings)
product_avgratings = average_second_column(product_ratings_collapsed)

user_and_rd = []
for i in range(0,len(user_ratings_products_collapsed)):
    user_id = user_ratings_products_collapsed[i][0]
    ratings_products = user_ratings_products_collapsed[i][1]
    deviations_for_user = []
    for j in range(0,len(ratings_products)):
        product_rating = ratings_products[j][0]
        product_avg_rating = product_avgratings[int(ratings_products[j][1])][1]
        deviations_for_user.append(np.absolute(product_rating - product_avg_rating) / 4.0)
    user_deviation = np.average(np.asmatrix(deviations_for_user))
    user_and_rd.append([user_id,user_deviation])

np.save("custom_data/rating_deviation",user_and_rd)