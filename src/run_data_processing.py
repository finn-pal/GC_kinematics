from tqdm import tqdm

from tools.convert_data import convert_data
from tools.gc_kinematics import get_kinematics
from tools.process_data import process_data
from tools.utils import get_halo_tree, open_snapshot

# variables
it_lst = [0, 1, 2, 3, 100]
snap_lst = [600]
it_lst = [0]
offset = 4

# directories
fire_dir = "/Volumes/My Passport for Mac/m12i_res7100"
# fire_dir = "/Users/z5114326/Documents/SampleData/m12i_res7100"
data_dir = "/Users/z5114326/Documents/GitHub/GC_kinematics/data/result/m12i"
snapshot_times_pub = "/Users/z5114326/Documents/GitHub/GC_kinematics/data/external/snapshot_times_public.txt"

main_halo_tid = 25236877

# processing flags
real_flag = 1  # (1 for real, 0 for not real, None for both)
survive_flag = None  # (1 for survive, 0 for disrupted, None for both)
accretion_flag = None  # (1 for accreted, 0 for not accreted, None for both)

halt = get_halo_tree(fire_dir)

for it in it_lst:
    convert_data(it, offset, fire_dir, data_dir, snapshot_times_pub)
    process_data(it, fire_dir, data_dir, main_halo_tid, halt, real_flag, survive_flag, accretion_flag)

# for kinematics should be a bit different and instead should iterate through snpas rather than it
# within loop of snapshot, imbed loop through it

for snap in snap_lst:
    part = open_snapshot(snap, fire_dir)
    for it in tqdm(
        it_lst, total=len(it_lst), ncols=150, desc="Retrieving Kinematics for Snapshot %d..." % snap
    ):
        get_kinematics(part, it, snap, data_dir)
