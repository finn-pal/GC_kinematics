import gizmo_analysis as gizmo
import h5py
import halo_analysis as halo
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.io import ascii
from utils import naming_value

#################### function inputs
it = 0

fire_dir = "/Volumes/My Passport for Mac/m12i_res7100"
data_dir = "/Users/z5114326/Documents/GitHub/GC_kinematics/data/result/m12i"
public_snapshot_fil = "/Users/z5114326/Documents/GitHub/GC_kinematics/data/external/snapshot_times_public.txt"

# processing file flags
file_real_flag = 1  # (1 for just real, 0 for just not real, None for both)
file_survive_flag = None  # (1 for survive, 0 for disrupted, None for both)
file_accretion_flag = None  # (1 for accreted, 0 for not accreted, None for both)

##### TIEM VARYING IOM WILL BE HARD TO DO WITH GC THAT HAVE DIED IN TERMS OF PARTICLE TRACKING. THINK ON THIS

# flags of interest (sort this out in terms of None, atm I just ignore that filter)
real_flag = 1
survive_flag = 1  # has survived
accretion_flag = None  # formed in-situ

# snap shots to investigate
snap_lst = []


#################### Classes #############


#################### Functions #############


def extract(lst: list, idx: int) -> list[float]:
    return [item[idx] for item in lst]


#################### start function

# file details
pro_dir = data_dir + "/processed/"

r = naming_value(file_real_flag)
s = naming_value(file_survive_flag)
a = naming_value(file_accretion_flag)

# save file to hdf5
pro_file = pro_dir + "pro_it%d_r%d_s%d_a%d.hdf5" % (it, r, s, a)

# halt = halo.io.IO.read_tree(simulation_directory=fire_dir)

# open interim data file
pro_df = pd.read_hdf(pro_file, "df")
fil_df = pro_df.copy()  # filtered dataframe

fil_df = fil_df[(fil_df["real_flag"] == real_flag) & (fil_df["survive_flag"] == survive_flag)]
fil_df = fil_df.reset_index(drop=True)

if snap_lst == []:
    # open file of public snapshots into a table
    with open(public_snapshot_fil) as f:
        content = f.readlines()
        content = content[13:]
    pub_snap = ascii.read(content)["index"]

min_snap = np.min(fil_df["pubsnap(zform)"])
snap_lst = [snap for snap in snap_lst if snap >= min_snap]

snap = 600
part = gizmo.io.Read.read_snapshots(["star", "dark"], "index", snap, fire_dir)

gc_id_lst = []
ptype_lst = []
group_id_lst = []
snapform_lst = []
position_lst = []
velocity_lst = []
ang_mom_lst = []
ek_lst = []
ep_lst = []

for gc_id, ptype, group_id, snap_form in zip(
    fil_df["GC_ID"], fil_df["ptype"], fil_df["group_id"], fil_df["snapnum(zform)"]
):
    idx = np.where(part[ptype]["id"] == gc_id)[0][0]
    position = part[ptype].prop("host1.distance", idx)
    velocity = part[ptype].prop("host1.velocity", idx)
    # position = part[ptype]["position"][idx]
    # velocity = part[ptype]["velocity"][idx]
    # ep = part[ptype]["potential"][idx]
    ep = part[ptype].prop("potential", idx)
    ek = 0.5 * np.linalg.norm(velocity) ** 2
    lx = position[1] * velocity[2] - position[2] * velocity[1]
    ly = position[2] * velocity[0] - position[0] * velocity[2]
    lz = position[0] * velocity[1] - position[1] * velocity[0]
    l_vec = [lx, ly, lz]

    gc_id_lst.append(gc_id)
    ptype_lst.append(ptype)
    group_id_lst.append(group_id)
    snapform_lst.append(snap_form)
    position_lst.append(position)
    velocity_lst.append(velocity)
    ang_mom_lst.append(l_vec)
    ek_lst.append(ek)
    ep_lst.append(ep)

kin_dict = {
    "GC_ID": gc_id_lst,
    "ptype": ptype_lst,
    "group_id": group_id_lst,
    "snapform": snapform_lst,
    "x": extract(position_lst, 0),
    "y": extract(position_lst, 1),
    "z": extract(position_lst, 2),
    "vx": extract(velocity_lst, 0),
    "vy": extract(velocity_lst, 1),
    "vz": extract(velocity_lst, 2),
    "lx": extract(ang_mom_lst, 0),
    "ly": extract(ang_mom_lst, 1),
    "lz": extract(ang_mom_lst, 2),
    "ek": ek_lst,
    "ep": ep_lst,
}

kin_df = pd.DataFrame(kin_dict)

r = naming_value(real_flag)
s = naming_value(survive_flag)
a = naming_value(accretion_flag)

save_file = pro_dir + "kin_it%d_r%d_s%d_a%d.hdf5" % (it, r, s, a)
kin_df.to_hdf(save_file, key="df", mode="w")
# kin_df.to_csv("kin_out.csv", index=False)
