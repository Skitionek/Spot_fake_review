import numpy as np

read_file = np.loadtxt('./Data/YelpZip/metadata', dtype=np.dtype("i4, i4, f4, i4, S10"))
read_file_short = read_file

data_list = []
for row in read_file_short:
    data_list.append(row[3])
data = np.asarray(data_list) # convert list of tuples to array
np.save("./Data/label",data)
