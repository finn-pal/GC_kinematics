# %%

import agama
import h5py
import numpy as np
import utilities as ut

from tools.utils import iteration_name, snapshot_name

# function inputs
# fire_dir = "/Volumes/My Passport for Mac/m12i_res7100"
# data_dir = "/Users/z5114326/Documents/GitHub/GC_kinematics/data/result/m12i"
# snap_num = 600
# it = 0
# host_index = 0  # this should be fixed

# The below function should be implemented in run_data_processing:


# should be a bit different should iterate through snpas rather than it
# this function should just take a single it and snap as inputs


def get_kinematics(part, it: int, snap_num: int, data_dir: str, host_index: int = 0):
    it_id = iteration_name(it)

    sim = data_dir[-4:]  # simulation name (e.g. m12i)
    data_file = data_dir + "/" + sim + "_processed.hdf5"  # file location
    potential_file = data_dir + "/potentials/snap_%d.ini" % snap_num

    pro_data = h5py.File(data_file, "a")  # open processed data file
    pot_nbody = agama.Potential(potential_file)

    gc_id = pro_data[it_id]["source"]["gc_id"]
    analyse_flag = pro_data[it_id]["source"]["analyse_flag"]
    group_id = pro_data[it_id]["source"]["group_id"]

    snap_zform = pro_data[it_id]["source"]["snap_zform"]
    snap_last = pro_data[it_id]["source"]["last_snap"]

    ptypes_byte = pro_data[it_id]["source"]["ptype"]
    ptypes = [ptype.decode("utf-8") for ptype in ptypes_byte]

    # need to make a function to state if the gc is alive at the snapshot in question
    gc_id_kin = []
    group_id_kin = []
    ptype_kin = []

    for gc, group, ptype, a_flag, snap_form, snap_disr in zip(
        gc_id, group_id, ptypes, analyse_flag, snap_zform, snap_last
    ):
        if a_flag == 0:
            continue

        if (snap_num < snap_form) or (snap_disr < snap_num):
            continue

        gc_id_kin.append(gc)
        group_id_kin.append(group)
        ptype_kin.append(ptype)

    # pro_data.close()

    host_name = ut.catalog.get_host_name(host_index)
    af = agama.ActionFinder(pot_nbody, interp=False)

    x_lst = []
    y_lst = []
    z_lst = []
    vx_lst = []
    vy_lst = []
    vz_lst = []

    r_cyl_lst = []
    phi_cyl_lst = []
    vr_cyl_lst = []
    vphi_cyl_lst = []

    r_lst = []
    r_per_lst = []
    r_apo_lst = []

    ep_fire_lst = []
    ep_agama_lst = []
    ek_lst = []
    et_lst = []

    lx_lst = []
    ly_lst = []
    lz_lst = []

    jr_lst = []
    jz_lst = []
    jphi_lst = []

    for gc, ptype in zip(gc_id_kin, ptype_kin):
        idx = np.where(part[ptype]["id"] == gc)[0][0]
        pos_xyz = part[ptype].prop(f"{host_name}.distance.principal", idx)
        vel_xyz = part[ptype].prop(f"{host_name}.velocity.principal", idx)

        pos_cyl = part[ptype].prop(f"{host_name}.distance.principal.cylindrical", idx)
        vel_cyl = part[ptype].prop(f"{host_name}.velocity.principal.cylindrical", idx)

        ep_fir = part[ptype]["potential"][idx]

        init_cond = np.concatenate((pos_xyz, vel_xyz))

        ep_aga = pot_nbody.potential(pos_xyz)
        r_per, r_apo = pot_nbody.Rperiapo(init_cond)

        ek = 0.5 * np.linalg.norm(vel_xyz) ** 2
        et = ek + ep_aga

        x, y, z = pos_xyz
        vx, vy, vz = vel_xyz

        r_cyl, phi_cyl, _ = pos_cyl
        vr_cyl, vphi_cyl, _ = vel_cyl

        r = np.linalg.norm(pos_xyz)

        lx = y * vz - z * vy
        ly = z * vx - x * vz
        lz = x * vy - y * vx

        jr, jz, jphi = af(init_cond)

        x_lst.append(x)
        y_lst.append(y)
        z_lst.append(z)
        vx_lst.append(vx)
        vy_lst.append(vy)
        vz_lst.append(vz)

        r_cyl_lst.append(r_cyl)
        phi_cyl_lst.append(phi_cyl)
        vr_cyl_lst.append(vr_cyl)
        vphi_cyl_lst.append(vphi_cyl)

        r_lst.append(r)
        r_per_lst.append(r_per)
        r_apo_lst.append(r_apo)

        ep_fire_lst.append(ep_fir)
        ep_agama_lst.append(ep_aga)
        ek_lst.append(ek)
        et_lst.append(et)

        lx_lst.append(lx)
        ly_lst.append(ly)
        lz_lst.append(lz)

        jr_lst.append(jr)
        jz_lst.append(jz)
        jphi_lst.append(jphi)

    kin_dict = {
        "gc_id": gc_id_kin,
        "ptype": ptype_kin,
        "group_id": group_id_kin,
        "x": x_lst,
        "y": y_lst,
        "z": z_lst,
        "vx": vx_lst,
        "vy": vy_lst,
        "vz": vz_lst,
        "r_cyl": r_cyl_lst,
        "phi_cyl": phi_cyl_lst,
        "vr_cyl": vr_cyl_lst,
        "vphi_cyl": vphi_cyl_lst,
        "r": r_lst,
        "r_peri": r_per_lst,
        "r_apoo": r_apo_lst,
        "ep_fire": ep_fire_lst,
        "ep_agama": ep_agama_lst,
        "ek": ek_lst,
        "et": et_lst,
        "lx": lx_lst,
        "ly": ly_lst,
        "lz": lz_lst,
        "jr": jr_lst,
        "jz": jz_lst,
        "jphi": jphi_lst,
    }

    snap_id = snapshot_name(snap_num)

    # with h5py.File(data_file, "a") as hdf:
    if it_id in pro_data.keys():
        grouping = pro_data[it_id]
    else:
        grouping = pro_data.create_group(it_id)
    if "kinematics" in grouping.keys():
        kinematics = grouping["kinematics"]
    else:
        kinematics = grouping.create_group("kinematics")
    if snap_id in kinematics.keys():
        snap_group = kinematics[snap_id]
    else:
        snap_group = kinematics.create_group(snap_id)
    for key in kin_dict.keys():
        if key in snap_group.keys():
            del snap_group[key]
        if key == "ptype":
            snap_group.create_dataset(key, data=kin_dict[key])
        else:
            snap_group.create_dataset(key, data=np.array(kin_dict[key]))

    pro_data.close()
