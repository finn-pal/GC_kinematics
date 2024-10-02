import os
import sys

import halo_analysis as halo
import numpy as np


def block_print():
    """
    Prevents prining of statements. Useful for when using halo tools as a lot of erroneous print statements.
    """

    sys.stdout = open(os.devnull, "w")


def enable_print():
    """
    Enables prints statements.
    """

    sys.stdout = sys.__stdout__


def get_halo_cid(halt, halo_tid: int, fire_dir: str) -> tuple[int, int]:
    """
    Finds the corresponding halo catalogue id (cid) from a provided halo id from the halo tree (tid).

    Args:
        halt (_type_): Halo tree
        halo_tid (int): Halo id from the halo tree
        fire_dir (str): Directory of the FIRE simulation data (of form "/m12i_res7100")

    Returns:
        tuple[int, int]: Returns the halo catalogue id (cid) and the corresponding snapshot
    """

    # get index of halo in halo tree
    idx = np.where(halt["tid"] == halo_tid)[0][0]
    # get the corresponding snapshot (halo catalogue) and index of the halo in the halo catalogue
    snap = halt["snapshot"][idx]
    halo_idx = halt["catalog.index"][idx]
    # import the relevant halo catalogue
    hal = halo.io.IO.read_catalogs("index", snap, simulation_directory=fire_dir, species=None)
    # get the halo catalogue id (cid)
    halo_cid = hal["id"][halo_idx]
    return halo_cid, snap


def main_prog_halt(halt) -> list[int]:
    """
    Get a list of the halo tree ids for the most massive progenitors of the main galaxy from gizmo halo tree

    Args:
        halt (_type_): Halo tree

    Returns:
        list[int]: List of halo tree halo ids (tid) tracing the main progenitors of the most massive galaxy at
        z = 0.
    """
    # main galaxy has index 0 in the halo tree
    main_halo_lst = [0]

    # FIRE has 600 snapshots but progenitor has usally not formed much earlier than snapshot 10
    for _ in range(1, 590):
        idx = halt["progenitor.main.index"][main_halo_lst[-1]]
        main_halo_lst.append(idx)

    tid_main_lst = halt["tid"][main_halo_lst]

    return tid_main_lst


def main_prog_yt(f_tree, main_halo) -> list[int]:
    """
    Get a list of the halo tree ids for the most massive progenitors of the main galaxy from GC YT tree

    Args:
        f_tree (_type_): Halo tree

    Returns:
        list[int]: List of halo tree halo ids (tid) tracing the main progenitors of the most massive galaxy at
        z = 0.
    """
    # get index of main halo in tree
    main_halo_idx = np.where(np.array(f_tree["SubhaloID"]) == main_halo)[0][0]
    main_halo_lst = [main_halo_idx]

    for _ in range(1, 590):
        halo_id = f_tree["FirstProgenitorID"][main_halo_lst[-1]]
        idx = np.where(np.array(f_tree["SubhaloID"]) == halo_id)[0][0]
        main_halo_lst.append(idx)

    tid_main_lst = f_tree["SubhaloID"][main_halo_lst]

    return tid_main_lst


def naming_value(flag) -> int:
    """
    Converts flag value to interger for input into file naming for processed data

    Args:
        flag (_type_): Flag value (can be 0, 1, or None)

    Returns:
        int: Flag for file name (0, 1 or 2)
    """
    if flag is None:
        flag_name = 2
    else:
        flag_name = flag
    return flag_name


def get_descendants_halt(halo_tid: int, halt) -> list[int]:
    """
    Get list of descendents (tid's) of the halo in question

    Args:
        halo_tid (int): Halo id from the halo tree
        halt (_type_): Halo tree

    Returns:
        list[int]: A list of descendant halos (list of tid's)
    """
    halo_idx = np.where(halt["tid"] == halo_tid)[0][0]
    halo_snap = halt["snapshot"][halo_idx]
    desc_lst = [halo_idx]

    # get a list of all descendents of the halo up to z = 0
    for _ in range(halo_snap, 600):
        idx = halt["descendant.index"][desc_lst[-1]]
        desc_lst.append(idx)

    return desc_lst


def get_descendants_yt(halo_tid: int, offset: int, f_tree) -> list[int]:
    """
    Get list of descendents (tid's) of the halo in question

    Args:
        halo_tid (int): Halo id from the halo tree
        offset (int): Offset from GC formation model interface
        f_tree (_type_): Halo tree

    Returns:
        list[int]: A list of descendant halos (list of tid's)
    """
    halo_idx = np.where(np.array(f_tree["SubhaloID"]) == halo_tid)[0][0]
    halo_snap = f_tree["SnapNum"][halo_idx]
    desc_lst = [halo_idx]

    for _ in range(halo_snap, 600 - offset):
        halo_des = f_tree["DescendantID"][desc_lst[-1]]
        idx = np.where(np.array(f_tree["SubhaloID"]) == halo_des)[0][0]
        desc_lst.append(idx)

    return desc_lst
