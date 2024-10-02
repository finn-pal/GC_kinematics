import halo_analysis as halo
import numpy as np
import pandas as pd
from astropy.io import ascii
from tqdm import tqdm

from tools.utils import (
    block_print,
    enable_print,
    get_descendants_halt,
    get_halo_cid,
    main_prog_halt,
    naming_value,
)


def particle_type(quality: int) -> str:
    """
    Using the data quality flag from gc model return particle type. From the gc model:
    # 2: good (star with t_s > t_form - time_lag):
    # 1: half (star with t_s < t_form - time_lag)
    # 0: bad (dm particle)

    Args:
        quality (int): Quality flag output from gc formation model (assign function).

    Returns:
        str: Particle type. Either "star" or "dark".
    """
    if quality == 0:
        part = "dark"
    else:
        part = "star"
    return part


def get_accretion(halt, halo_tid: int, tid_main_lst: list, fire_dir: str, t_dis: float = -1) -> dict:
    """
        Find if the gc has been accreted or formed in-situ. If accreted find details of its accretion.

    Args:
        halt (_type_): Halo tree
        halo_tid (int): Halo tree halo id
        tid_main_lst (list): List of halo tree halo ids (tid) tracing the main progenitors of the most massive
            galaxy at z = 0.
        fire_dir (str): Directory of the FIRE simulation data (of form "/m12i_res7100")
        t_dis (_type_): If applicabale, this is the time of gc disruption.

    Returns:
        dict:
            0 -> Accretion flag (1 for ex-situ formation, 0 for in-situ formation)
            1 -> Time of accretion (Time when gc is now assigned to a most massive progenitor of the main
                galaxy)
            2 -> Halo tree halo id of the gc at the time of accretion
            3 -> Halo catalogue halo id of the gc at the time of accretion
            4 -> Snapshot at the time of accretion
            5 -> Halo tree halo id of the gc at the snapshot before accretion
            6 -> Halo catalogue halo id of the gc at the snapshot before accretion
            7 -> Snapshot at the snapshot before accretion
            8 -> Survived accretion flag. If gc is disrupted at accretion set to 0. If discrupted before
                accretion or if not relevant set to -1 otherwise if survived accretion set to 1
    """
    # snapshot times file
    snapshot_times = "/snapshot_times.txt"

    # if gc is not accreted then list acc_flag as 0 and other all values as -1
    if halo_tid in tid_main_lst:
        acc_flag = 0
        t_acc = -1
        halo_acc_tid = -1
        halo_acc_cid = -1
        snap_acc = -1
        halo_pre_acc_tid = -1
        halo_pre_acc_cid = -1
        snap_pre_acc = -1
        acc_survive = -1

    else:
        # open table of snapshot information
        with open(fire_dir + snapshot_times) as f:
            content = f.readlines()
            content = content[5:]
        snap_all = ascii.read(content)

        # get list of descendents (tid's) of the halo
        desc_lst = get_descendants_halt(halo_tid, halt)

        # find which descendent of the halo of formation has been accreted into the main galaxy
        idx_lst = np.array([1 if halt["tid"][idx] in tid_main_lst else 0 for idx in desc_lst])
        idx_acc = np.where(idx_lst == 1)[0][0]

        # get the time of accretion
        snap_acc = halt["snapshot"][desc_lst[idx_acc]]
        t_acc = snap_all["time[Gyr]"][snap_acc]
        t_acc = np.round(t_acc, 3)

        # if gc is not disrupted or disrupted after accretion then get all details of accretion
        if (t_dis == -1) or (t_dis > t_acc):
            idx_pre_acc = idx_acc - 1

            acc_flag = 1
            t_acc = t_acc
            halo_acc_tid = halt["tid"][desc_lst[idx_acc]]
            halo_acc_cid, snap_acc = get_halo_cid(halt, halo_acc_tid, fire_dir)
            halo_pre_acc_tid = halt["tid"][desc_lst[idx_pre_acc]]
            halo_pre_acc_cid, snap_pre_acc = get_halo_cid(halt, halo_pre_acc_tid, fire_dir)
            acc_survive = 1  # survived

        # if gc disrupted at the time of accretion then get all details of accretion
        elif t_dis == t_acc:
            idx_pre_acc = idx_acc - 1

            acc_flag = 1
            t_acc = t_acc
            halo_acc_tid = halt["tid"][desc_lst[idx_acc]]
            halo_acc_cid, snap_acc = get_halo_cid(halt, halo_acc_tid, fire_dir)
            halo_pre_acc_tid = halt["tid"][desc_lst[idx_pre_acc]]
            halo_pre_acc_cid, snap_pre_acc = get_halo_cid(halt, halo_pre_acc_tid, fire_dir)
            acc_survive = 0  # did not survived

        # if gc disrupted before halo is accreted then set all values to -1
        else:
            acc_flag = 0
            t_acc = -1
            halo_acc_tid = -1
            halo_acc_cid = -1
            snap_acc = -1
            halo_pre_acc_tid = -1
            halo_pre_acc_cid = -1
            snap_pre_acc = -1
            acc_survive = -1

    accretion_dict = {
        "accretion_flag": acc_flag,
        "accretion_time": t_acc,
        "accretion_halo_tid": halo_acc_tid,
        "accretion_halo_cid": halo_acc_cid,
        "accretion_snapshot": snap_acc,
        "pre_accretion_halo_tid": halo_pre_acc_tid,
        "pre_accretion_halo_cid": halo_pre_acc_cid,
        "pre_accretion_snapshot": snap_pre_acc,
        "survived_accretion": acc_survive,
    }

    return accretion_dict


def group_accretion(df: pd.DataFrame) -> list[int]:
    """
    Group accretion's together for easy identification. Group 0 is in-situ formation, -1 is gc's disrupted
    before accretion and all other values relate to the halo tid of the gc the snapshot before accretion.

    Args:
        df (pd.DataFrame): processed data frame

    Returns:
        list[int]: list of group id's to be added to dataframe
    """
    group_id_lst = []

    for pre_acc_halo_tid, accretion_flag in zip(df["pre_accretion_halo_tid"], df["accretion_flag"]):
        if accretion_flag == 0:
            group_id_lst.append(0)

        elif pre_acc_halo_tid == -1:
            group_id_lst.append(-1)

        else:
            group_id_lst.append(pre_acc_halo_tid)

    return group_id_lst


def process_data(it: int, fire_dir: str, data_dir: str, real_flag=1, survive_flag=None, accretion_flag=None):
    """
    Process interim data and add additional information necessary for analysis. This includes deriving
    accretion information about the gc particles. There is also the option to filter out based on flags.

    Args:
        it (int): Iteration number. This realtes to the randomiser seed used in the gc model.
        fire_dir (str): Directory of the FIRE simulation data (of form "/m12i_res7100").
        data_dir (str): Directory where raw / interim / processed data is stored (of form "result/m12i").
        real_flag (int, optional): 0 means not real (see convert_data function for details). 1 means real.
            None means to include both. Defaults to 1.
        survive_flag (_type_, optional): 0 means has not survived. 1 means has survived. None means to include
            both. Defaults to None.
        accretion_flag (_type_, optional): 0 means has not been accreted. 1 means has been accreted. None
            means to include both. Defaults to None.
    """
    for _ in tqdm(range(10), desc="Retrieving Tree..."):
        block_print()
        halt = halo.io.IO.read_tree(simulation_directory=fire_dir)
        enable_print()

    # get list of main progenitors of most massive galaxy across all redshifts
    tid_main_lst = main_prog_halt(halt)

    # file details
    int_dir = data_dir + "/interim/"
    pro_dir = data_dir + "/processed/"

    # open interim data file
    int_df = pd.read_hdf(int_dir + "int_it_%d.hdf5" % it, "df")
    pro_df = int_df.copy()

    # filter based on real flag
    if real_flag is None:
        pro_df = pro_df
    else:
        pro_df = pro_df[pro_df["real_flag"] == real_flag]
        pro_df = pro_df.reset_index(drop=True)

    # filter based on survive flag
    if survive_flag is None:
        pro_df = pro_df
    else:
        pro_df = pro_df[pro_df["survive_flag"] == survive_flag]
        pro_df = pro_df.reset_index(drop=True)

    # get accretion flag
    halo_tid_lst = pro_df["halo(zform)"]
    acc_flag_list = []
    for halo_tid in halo_tid_lst:
        if halo_tid in tid_main_lst:
            acc_flag_list.append(0)
        else:
            acc_flag_list.append(1)
    pro_df.loc[:, "accretion_flag"] = acc_flag_list

    # filter based on accretion flag
    if accretion_flag is None:
        pro_df = pro_df
    else:
        pro_df = pro_df[pro_df["accretion_flag"] == accretion_flag]

    # empty lists to be filled
    t_acc_lst = []
    halo_acc_tid_lst = []
    halo_acc_cid_lst = []
    snap_acc_lst = []
    halo_pre_acc_tid_lst = []
    halo_pre_acc_cid_lst = []
    snap_pre_acc_lst = []
    acc_survive_lst = []

    # get accretion information where relevant
    for halo_form, t_dis in tqdm(
        zip(pro_df["halo(zform)"], pro_df["t_dis"]), total=len(pro_df), desc="Processing Data..."
    ):
        block_print()
        accretion_dict = get_accretion(halt, halo_form, tid_main_lst, fire_dir, t_dis)
        enable_print()

        t_acc_lst.append(accretion_dict["accretion_time"])
        halo_acc_tid_lst.append(accretion_dict["accretion_halo_tid"])
        halo_acc_cid_lst.append(accretion_dict["accretion_halo_cid"])
        snap_acc_lst.append(accretion_dict["accretion_snapshot"])
        halo_pre_acc_tid_lst.append(accretion_dict["pre_accretion_halo_tid"])
        halo_pre_acc_cid_lst.append(accretion_dict["pre_accretion_halo_cid"])
        snap_pre_acc_lst.append(accretion_dict["pre_accretion_snapshot"])
        acc_survive_lst.append(accretion_dict["survived_accretion"])

    # add accretion information to dataframe
    pro_df.loc[:, "accretion_time"] = t_acc_lst
    pro_df.loc[:, "accretion_halo_tid"] = halo_acc_tid_lst
    pro_df.loc[:, "accretion_halo_cid"] = halo_acc_cid_lst
    pro_df.loc[:, "accretion_snapshot"] = snap_acc_lst
    pro_df.loc[:, "pre_accretion_halo_tid"] = halo_pre_acc_tid_lst
    pro_df.loc[:, "pre_accretion_halo_cid"] = halo_pre_acc_cid_lst
    pro_df.loc[:, "pre_accretion_snapshot"] = snap_pre_acc_lst
    pro_df.loc[:, "survived_accretion"] = acc_survive_lst

    ptype_lst = []
    for qual in pro_df["quality"]:
        ptype = particle_type(qual)
        ptype_lst.append(ptype)

    pro_df.loc[:, "ptype"] = ptype_lst

    # add accretion group
    group_id_lst = group_accretion(pro_df)
    pro_df.loc[:, "group_id"] = group_id_lst

    # file naming based on flags used such as to ensure different filtered data is not overwritten
    r = naming_value(real_flag)
    s = naming_value(survive_flag)
    a = naming_value(accretion_flag)

    # save file to hdf5
    save_file = pro_dir + "pro_it%d_r%d_s%d_a%d.hdf5" % (it, r, s, a)
    pro_df.to_hdf(save_file, key="df", mode="w")
