import numpy as np

def sortAndCollapse2dArray(array):
    array_sorted = array[array[:, 0].argsort()]
    collapsed_array = []
    last_index=0
    for i in range(1,len(array_sorted)):
        if array_sorted[i][0] != array_sorted[last_index][0]:
            newRow = []
            newRow.append(array_sorted[last_index][0])
            newRow.append(array_sorted[last_index:i,1:])
            collapsed_array.append(newRow)
            last_index = i
            continue
    newRow = []
    newRow.append(array_sorted[last_index][0])
    newRow.append(array_sorted[last_index:,1:])
    collapsed_array.append(newRow)
    return collapsed_array

user_prod_inburst = np.load("custom_data/user_prod_inburst.npy")
user_prod_inburst_collapsed = sortAndCollapse2dArray(user_prod_inburst)

B_a_abs = []
V_a_abs = []
for i in range(0, len(user_prod_inburst_collapsed)):
    user_id = user_prod_inburst_collapsed[i][0]
    reviews_and_inburst = user_prod_inburst_collapsed[i][1]
    reviews_in_burst = set()
    for j in range(0,len(reviews_and_inburst)):
        if reviews_and_inburst[j][1] == 1:
            reviews_in_burst = reviews_in_burst.union([reviews_and_inburst[j][0]])
    B_a_abs.append([user_id, len(reviews_in_burst)])
    V_a_abs.append([user_id, len(reviews_and_inburst)])
B_a_abs = np.asarray(B_a_abs)
V_a_abs = np.asarray(V_a_abs)

BBR = np.transpose(np.asarray([B_a_abs[:,0],np.divide(B_a_abs[:,1],V_a_abs[:,1])]))
np.save("custom_data/BBR",BBR)

