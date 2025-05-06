import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv
import numpy as np

sites = []
g = []
t_1_max = []
t_2_max = []
energy = []
energy_converge = []

base_path = r"/Users/gilfrim/Desktop/Quantum\ Chemistry\ Coop"
plots_file = base_path + r"\plot_points.csv"

#goes through the plot points csv and acquires values
with open(plots_file, mode = "r") as csvfile:
    reader = csv.reader(csvfile, delimiter = ",")
    next(reader)
    for line in reader:
        sites.append(float(line[0]))
        g.append(float(line[1]))
        t_1_max.append(float(line[2]))
        t_2_max.append(float(line[3]))
        energy.append(float(line[4]))
        if float(line[0]) == 49:
            e_converge = float(line[4])
    for values in energy:
        energy_converge.append(np.log(values - e_converge))
#
site = []
max_t = []

#goes through max y csv
max_y_file = base_path + r"\max_per_y.csv"
with open(max_y_file, mode = "r") as csvfile_max:
    reader_max = csv.reader(csvfile_max, delimiter = ",")
    next(reader_max)
    for index in reader_max:
        site.append(int(index[0]))
        #max_t.append(np.log(float(index[1])))
        max_t.append(float(index[1]))

# fig = plt.figure()
# ax = fig.add_subplot(111, projection = '3d')

# scatter = ax.scatter(sites, g)
# ax.set_xlabel(f"{sites}")
# ax.set_ylabel(f"{sites}")
# ax.set_zlabel('t_2_max')
energy_per_site = []
for i in range(len(energy)):
    energy_per_site.append((energy[i] - e_converge)/ sites[i])

plt.figure()
#change this to the points you want to plot
plt.scatter(site, np.log(max_t))
#title and labels
plt.title(f"g = {g[0]}")
plt.xlabel("sites")
plt.ylabel("ln(t max)")
#plt.grid(True)
plt.show()
