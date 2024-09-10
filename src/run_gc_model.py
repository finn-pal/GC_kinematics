import os

# from GC_formation_model import run
from params.m12i_params import params
from tools.gc_model import run_gc_model

it = 2

resultspath = params["resultspath"]
resultspath = resultspath + str("it_%d/" % it)
params["resultspath"] = resultspath
params["seed"] = int(it)

if not os.path.exists(resultspath):
    os.makedirs(resultspath)

run_gc_model(params)
