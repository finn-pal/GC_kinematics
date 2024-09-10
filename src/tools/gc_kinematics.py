import halo_analysis as halo
import numpy as np
import pandas as pd
from utils import get_descendants, naming_value

#################### function inputs
it = 1

fire_dir = "/Volumes/My Passport for Mac/m12i_res7100"
data_dir = "/Users/z5114326/Documents/GitHub/GC_kinematics/data/result/m12i"

# processing file flags
file_real_flag = 1  # (1 for just real, 0 for just not real, None for both)
file_survive_flag = None  # (1 for survive, 0 for disrupted, None for both)
file_accretion_flag = None  # (1 for accreted, 0 for not accreted, None for both)

##### TIEM VARYING IOM WILL BE HARD TO DO WITH GC THAT HAVE DIED IN TERMS OF PARTICLE TRACKING. THINK ON THIS

# flags of interest
real_flag = 1
survive_flag = 1  # has survived
accretion_flag = 0  # formed in-situ

#################### start function

# file details
pro_dir = data_dir + "/processed/"

r = naming_value(file_real_flag)
s = naming_value(file_survive_flag)
a = naming_value(file_accretion_flag)

# save file to hdf5
pro_file = pro_dir + "pro_it%d_r%d_s%d_a%d.hdf5" % (it, r, s, a)

halt = halo.io.IO.read_tree(simulation_directory=fire_dir)

# open interim data file
pro_df = pd.read_hdf(pro_file, "df")
pro_df = pro_df.sort_values(by=["snapnum(zform)"], ascending=True)
pro_df = pro_df.reset_index(drop=True)

halo_group_lst = []
group_id = []

for accretion in pro_df["accretion_flag"]:
    if accretion == 0:
        group_id.append(0)
    else:
        group_id.append(-1)

# print(pro_df)

for idx in range(0, len(pro_df)):
    print(idx)
    if group_id[idx] == 0:
        continue

    else:
        halo_tid = pro_df["halo(zform)"][idx]

        new_halo_group = get_descendants(halo_tid, halt)
        halo_group_lst.append(new_halo_group)

        for count, halo_group in enumerate(halo_group_lst):
            group = count + 1
            if halo_tid in halo_group:
                group_id[idx] = group
            else:
                new_group_num = len(halo_group_lst) + 1
                group_id[idx] = new_group_num
                new_halo_group = get_descendants(halo_tid, halt)
                halo_group_lst.append(new_halo_group)


print(len(halo_group_lst))

# halo_group = get_descendants(halo_tid, halt)

# halo_group_lst.append(halo_group)


################################################

# fil_df = pro_df.copy()  # filtered dataframe

# fil_df = fil_df[
#     (fil_df["real_flag"] == real_flag)
#     & (fil_df["survive_flag"] == survive_flag)
#     & (fil_df["accretion_flag"] == accretion_flag)
# ]

# fil_df = fil_df.reset_index(drop=True)

# min_snap = np.min(fil_df["pubsnap(zform)"])

# print(min_snap)
