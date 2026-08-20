#!/usr/bin/env python3
"""Regenerate ALL figures for the paper into ../figs/ (as vector PDFs).

Run from anywhere:
    python3 regenerate_all_figs.py

Requires: numpy, matplotlib. Takes ~3-5 minutes on a single CPU core.
Produces (in ../figs/):
  fig_ramp, fig_bound, fig_iql, fig_single, fig_pg, fig_exec   (base experiments E1-E6)
  fig_env_schematic, fig_env_render                            (environment schematic + render)
  fig_scale_surface, fig_scale_survival, fig_scale_bound, fig_scale_cost  (L x N scaling)
  fig_core_prune, fig_core_reuse                               (core-effect statistics)
  fig_continual_control                                        (registered E10 confirmation)
  fig_lbf_core_erosion_exploratory                             (exploratory E11 LBF analysis)
  fig_cue_mnist_confirmation                                   (registered E12 confirmation)
"""
import json
import os
import sys
import time

# make sure we run inside code/ so the "../figs/" relative paths resolve
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
os.makedirs("../figs", exist_ok=True)
sys.path.insert(0, HERE)


def banner(msg):
    print("\n" + "=" * 60 + "\n" + msg + "\n" + "=" * 60)


def main():
    t0 = time.time()
    all_results = {}

    banner("Environment figures (schematic + rendered instance)")
    import env_figures
    env_figures.schematic()
    env_figures.gridworld()
    print("  -> fig_env_schematic.pdf, fig_env_render.pdf")

    banner("Base experiments E1, E2, E5")
    import experiments as E125
    a, m, p = E125.e1(); E125.e2(a, m, p); E125.e5()
    all_results.update(E125.results)
    json.dump(E125.results, open("../figs/results_e125.json", "w"), indent=2)
    print("  -> fig_ramp.pdf, fig_bound.pdf, fig_exec.pdf")

    banner("Co-adaptation experiments E3 (argmax peer), E4 (policy-gradient peer)")
    import experiments2 as E34
    E34.e3(); E34.e4()
    all_results.update(E34.res)
    json.dump({"e3": E34.res.get("e3")}, open("../figs/results_e3.json", "w"), indent=2)
    json.dump({"e4": E34.res.get("e4")}, open("../figs/results_e4.json", "w"), indent=2)
    print("  -> fig_iql.pdf, fig_pg.pdf")

    banner("Single-agent control E6")
    import runpy
    runpy.run_path("e6.py", run_name="__main__")
    try:
        all_results["e6"] = json.load(open("../figs/results_e6.json"))
    except Exception as exc:
        print("  (could not read results_e6.json: %s)" % exc)
    print("  -> fig_single.pdf")

    banner("L x N scaling (surfaces, survival slices, bound validity, core cost)")
    import experiments_scaling as ES
    ES.destruction_and_drift()
    ES.survival_slices()
    ES.bound_validity()
    ES.core_cost()
    all_results["scaling"] = ES.res
    json.dump(ES.res, open("../figs/results_scaling.json", "w"), indent=2)
    print("  -> fig_scale_surface.pdf, fig_scale_survival.pdf, fig_scale_bound.pdf, fig_scale_cost.pdf")

    banner("Core-effect statistics (pruning power, reuse value)")
    import experiments_core_effect as EC
    EC.pruning_power()
    EC.reuse_value()
    all_results["core_effect"] = EC.res
    json.dump(EC.res, open("../figs/results_core_effect.json", "w"), indent=2)
    print("  -> fig_core_prune.pdf, fig_core_reuse.pdf")

    banner("Direct theorem validations E7-E9 (sampling and stability tightness)")
    import experiments_theory_tests as TT
    TT.e7_sample_complexity()
    TT.e8_tightness()
    TT.e9_kernel_tightness()
    all_results["theory_tests"] = TT.res
    json.dump(TT.res, open("../figs/results_theory_tests.json", "w"), indent=2)
    print("  -> fig_samplecomplexity.pdf, fig_tightness.pdf, fig_kerneltight.pdf")

    banner("Registered single-agent continual-control confirmation (E10)")
    import plot_continual_control_confirmation as CC
    CC.make_figure()
    print("  -> fig_continual_control.pdf")

    banner("Exploratory Level-Based Foraging core-erosion analysis (E11)")
    import plot_lbf_core_erosion_exploratory as LBF
    LBF.make_figure()
    print("  -> fig_lbf_core_erosion_exploratory.pdf")

    banner("Registered continual cue-MNIST confirmation (E12)")
    import plot_cue_mnist_confirmation as MNIST
    MNIST.make_figure()
    print("  -> fig_cue_mnist_confirmation.pdf")

    banner("Publication figure styling (data-linked final pass)")
    import make_oral_figures as STYLED
    STYLED.fig_continual_control()
    STYLED.fig_cue_mnist()
    STYLED.fig_src_schematic()
    STYLED.fig_reduction()
    import plot_main_experiments as MAIN
    MAIN.make_figure()
    print("  -> restyled E10/E11 composite, E12, the SRC schematic, and the reduction diagram")

    # Persist the combined record. Without this the per-experiment numbers quoted in
    # the paper (e.g. E2's median slack) can silently go stale relative to the figures,
    # because the modules' own __main__ blocks do NOT run when they are imported here.
    json.dump(all_results, open("../figs/all_results.json", "w"), indent=2)
    banner("DONE in %.1f min. Figures and results written to ../figs/" % ((time.time() - t0) / 60.0))
    print(json.dumps(all_results, indent=2)[:2000])


if __name__ == "__main__":
    main()
