import os

import h5py
import numpy as np
import pandas as pd
from astropy.io import ascii
from tqdm import tqdm

from tools.utils import iteration_name


def convert_data(it: int, offset: int, fire_dir: str, data_dir: str, public_snapshot_fil: str):
    """
    Filter raw data to only include important information and add extra data that enables ease of analysis
    / filtering later. Will print this data table to hdf5 format in the interim data directory.

    Args:
        it (int): Iteration number. This realtes to the randomiser seed used in the gc model.
        offset (int): Offset used to convert from yt form into gizmo form (used in the interface).
        fire_dir (str): Directory of the FIRE simulation data (of form "/m12i_res7100").
        data_dir (str): Directory where raw / interim / processed data is stored (of form "result/m12i").
        public_snapshot_fil (str): Location of file that stores which snapshots have been publically released.
    """
    # ensure group naming is consitent with three digits
    it_group = iteration_name(it)

    for _ in tqdm(range(1), ncols=150, desc=it_group + " Converting Data...................."):
        # data directory
        raw_dir = data_dir + "/raw/it_%d/" % it

        # output files from the gc model
        main_fil = "allcat_s-%d_p2-7_p3-1.txt" % it
        disrupt_fil = "allcat_s-%d_p2-7_p3-1_k-1.5_logm_snap596.txt" % it
        t_disrupt_fil = "allcat_s-%d_p2-7_p3-1_k-1.5_t_disrupt_snap596.txt" % it
        gc_id_fil = "allcat_s-%d_p2-7_p3-1_gcid.txt" % it
        # simulation snapshot times
        snapshot_fil = "/snapshot_times.txt"

        # convert main data file into pandas dataframe
        with open(raw_dir + main_fil) as f:
            content = f.readlines()

        dat = ascii.read(content)

        # rename columns
        dat["col1"].name = "halo(z=0)"
        dat["col2"].name = "logMh(z=0)"
        dat["col3"].name = "logM*(z=0)"
        dat["col4"].name = "logMh(zfrom)"
        dat["col5"].name = "logM*(zform)"
        dat["col6"].name = "logM(tform)"
        dat["col7"].name = "zform"
        dat["col8"].name = "feh"
        dat["col9"].name = "isMPB"
        dat["col10"].name = "halo(zform)"
        dat["col11"].name = "snapnum(zform)"

        # convert table to pandas dataframe for easier analysis
        main_df = dat.to_pandas()

        # convert gc disrupt file into pandas dataframe
        dis_df = pd.read_csv(raw_dir + disrupt_fil, header=None)
        dis_df.columns = ["logM(z=0)"]  # -1 means did not survive to z = 0

        # convert gc time of disrupt file into pandas dataframe
        tdis_df = pd.read_csv(raw_dir + t_disrupt_fil, header=None)
        tdis_df.columns = ["t_dis"]  # -1 means did not survive to z = 0

        # convert gc_id file into pandas dataframe
        gcid_df = pd.read_csv(raw_dir + gc_id_fil, sep=" ").drop(["ID"], axis=1)
        gcid_df.columns = ["GC_ID", "quality"]

        # combine all tables into a single file for ease of analysis
        raw_df = pd.concat([main_df, dis_df, gcid_df, tdis_df], axis=1, join="inner")

        # select only columns of interest
        int_df = raw_df[
            [
                "GC_ID",  # particle id to which the gc has been assigned
                "quality",  # quality of particle, this is used to infer what particle type it is
                "halo(z=0)",  # halo tid at the z = 0 (should be the same for all particles)
                "logM(tform)",  # mass of the gc at time of gc formation
                "zform",  # redshift of gc formation
                "halo(zform)",  # halo tid at the time of gc formation
                "snapnum(zform)",  # snapshot of gc formation
                "isMPB",  # halo of formation is a main progenitor of the most massive galaxy at z = 0
                "feh",  # metallicty at z = 0
                "logM(z=0)",  # mass of the gc at z = 0
                "t_dis",  # time of disruption of the gc (-1 if gc survives at z = 0 )
            ]
        ].copy()

        columns = {
            "GC_ID": "gc_id",
            "halo(z=0)": "halo_z0",
            "logM(tform)": "logm_tform",
            "halo(zform)": "halo_zform",
            "snapnum(zform)": "snap_zform",
            "isMPB": "is_mpb",
            "logM(z=0)": "logm_z0",
        }

        # rename columns
        int_df = int_df.rename(columns=columns)

        # convert snapshots from yt form into gizmo form by adding offset used in the interface
        int_df.loc[:, "snap_zform"] = int_df.loc[:, "snap_zform"] + offset
        # create a flag for gc particles that have survived until z = 0
        int_df.loc[:, "survive_flag"] = np.where(int_df["logm_z0"] != -1, 1, 0)

        # open file of all snapshots into a table
        with open(fire_dir + snapshot_fil) as f:
            content = f.readlines()
            content = content[5:]
        snap_all = ascii.read(content)

        # get the formation times and lookback formation times for the gcs
        idx_lst = [np.where(np.round(snap_all["redshift"], 3) == red)[0][0] for red in int_df["zform"]]
        int_df["form_time"] = [snap_all["time[Gyr]"][idx] for idx in idx_lst]
        int_df["form_lbt"] = [snap_all["lookback-time[Gyr]"][idx] for idx in idx_lst]

        # open file of public snapshots into a table
        with open(public_snapshot_fil) as f:
            content = f.readlines()
            content = content[13:]
        snap_pub = ascii.read(content)["index"]

        # get the closest public snapshot available after the formation of the gc
        pub_idx = np.searchsorted(np.array(snap_pub), int_df["snap_zform"])
        pub_idx = [len(pub_idx) if idx == len(pub_idx) else idx for idx in pub_idx]
        int_df["pubsnap_zform"] = [snap_pub[idx] for idx in pub_idx]

        # get last snap where alive
        dis_idx = np.searchsorted(np.array(snap_all["time[Gyr]"]), int_df["t_dis"])
        dis_idx = [len(dis_idx) if idx == len(dis_idx) else idx for idx in dis_idx]
        int_df["last_snap"] = [snap_all["i"][idx - 1] for idx in dis_idx]

        # set a "real flag" which excludes gc's that have been disrupted but do not have a time of discruption
        # the candidates for this are usually repeated gc particle ids in adjacent public snapshots
        int_df.loc[:, "real_flag"] = np.where((int_df["logm_z0"] == -1) & (int_df["t_dis"] == -1), 0, 1)

    sim = data_dir[-4:]  # simulation name (e.g. m12i)
    save_file = data_dir + "/" + sim + "_processed.hdf5"  # save location

    if not os.path.exists(save_file):
        h5py.File(save_file, "w")

    with h5py.File(save_file, "a") as hdf:
        if it_group in hdf.keys():
            grouping = hdf[it_group]
        else:
            grouping = hdf.create_group(it_group)
        if "source" in grouping.keys():
            source = grouping["source"]
        else:
            source = grouping.create_group("source")
        for key in int_df.keys():
            if key in source.keys():
                del source[key]
            source.create_dataset(key, data=np.array(int_df[key]))
