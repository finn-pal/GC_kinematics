from GC_formation_model import astro_utils, logo
from GC_formation_model.assign import assign
from GC_formation_model.evolve import evolve
from GC_formation_model.form import form
from GC_formation_model.get_tid import get_tid
from GC_formation_model.offset import offset

__all__ = ["run_gc_model"]


def run_gc_model(params):
    if params["verbose"]:
        logo.print_logo()
        logo.print_version()
        print("\nWe refer to the following papers for model details:")
        logo.print_papers()
        print("\nRuning model on %d halo(s)." % len(params["subs"]))

    allcat_name = params["allcat_base"] + "_s-%d_p2-%g_p3-%g.txt" % (
        params["seed"],
        params["p2"],
        params["p3"],
    )

    run_params = params
    run_params["allcat_name"] = allcat_name

    run_params["cosmo"] = astro_utils.cosmo(
        h=run_params["h100"], omega_baryon=run_params["Ob"], omega_matter=run_params["Om"]
    )

    form(run_params)
    offset(run_params)
    assign(run_params)
    get_tid(run_params)
    evolve(run_params, return_t_disrupt=True)

    if params["verbose"]:
        print("\nModel was run on %d halo(s).\n" % len(params["subs"]))
