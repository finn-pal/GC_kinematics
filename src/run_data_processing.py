# from tools.convert_data_v1 import convert_data
# from tools.process_data_v1 import process_data

from tools.convert_data_v2 import convert_data

# from tools.process_data_yt import process_data

# variables
it_lst = [0, 1, 2, 3, 100]
offset = 4

# directories
fire_dir = "/Volumes/My Passport for Mac/m12i_res7100"
data_dir = "/Users/z5114326/Documents/GitHub/GC_kinematics/data/result/m12i"
snapshot_times_pub = "/Users/z5114326/Documents/GitHub/GC_kinematics/data/external/snapshot_times_public.txt"

# processing flags
real_flag = 1  # (1 for just real, 0 for just not real, None for both)
survive_flag = None  # (1 for survive, 0 for disrupted, None for both)
accretion_flag = None  # (1 for accreted, 0 for not accreted, None for both)

for it in it_lst:
    convert_data(it, offset, fire_dir, data_dir, snapshot_times_pub)
    # process_data(it, fire_dir, data_dir, real_flag, survive_flag, accretion_flag)
