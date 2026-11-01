"""Design a welded beam of least cost, with its limits on what an analysis of the beam reports.

A beam is welded to a support and carries a 6,000 lb load at its tip, 14 inches out. The weld's
thickness and length, and the beam's thickness and width, set its cost. The analysis of each
design reports the shear stress in the weld, the bending stress in the beam, the load at which
the beam buckles and how far its tip deflects, and each has a limit. Here the analysis is a
handful of formulas, but in practice it is often a finite element model that takes minutes, so
the optimizer only learns that a design breaks a limit after it has paid for the analysis. The
best published design costs 1.724852.

The limits on what the analysis reports are written with Output, alongside the two geometric
limits, which need no analysis and so are checked before evaluating. Bayesian optimization
models each reported limit, so it learns where the designs that break one lie.
"""
import math

from hone import BayesianConfig, BayesianOptimizer, Output, RangeParameter, SearchSpace

LOAD, LENGTH = 6000.0, 14.0  # the load (lb) and the beam's length (in)
YOUNG, SHEAR_MODULUS = 30e6, 12e6  # the steel's moduli (psi)

weld = RangeParameter("weld", 0.1, 2.0)  # the weld's thickness, in inches
weld_length = RangeParameter("weld_length", 0.1, 10.0)  # the weld's length, in inches
thickness = RangeParameter("thickness", 0.1, 10.0)  # the beam's thickness, in inches
width = RangeParameter("width", 0.1, 2.0)  # the beam's width, in inches

search_space = SearchSpace(
    [weld, weld_length, thickness, width],
    constraints=[
        # Reported by the analysis, so judged after evaluating.
        Output("shear") <= 13_600,  # psi, in the weld
        Output("bending") <= 30_000,  # psi, in the beam
        Output("buckling_load") >= LOAD,  # lb: the beam mustn't buckle under the load
        Output("deflection") <= 0.25,  # in, at the tip
        # Geometry, so checked before evaluating.
        weld <= width,
        0.10471 * weld**2 + 0.04811 * thickness * width * (14 + weld_length) <= 5,
    ],
)


def analyse(weld, weld_length, thickness, width):
    """The beam's cost, and the stresses, buckling load and deflection its limits apply to."""
    primary = LOAD / (math.sqrt(2) * weld * weld_length)
    moment = LOAD * (LENGTH + weld_length / 2)
    radius = math.sqrt(weld_length**2 / 4 + ((weld + thickness) / 2) ** 2)
    polar = 2 * math.sqrt(2) * weld * weld_length * (weld_length**2 / 12 + ((weld + thickness) / 2) ** 2)
    secondary = moment * radius / polar
    shear = math.sqrt(primary**2 + primary * secondary * weld_length / radius + secondary**2)
    buckling = (
        4.013 * YOUNG * math.sqrt(thickness**2 * width**6 / 36) / LENGTH**2
        * (1 - thickness / (2 * LENGTH) * math.sqrt(YOUNG / (4 * SHEAR_MODULUS)))
    )
    return {
        "objective": 1.10471 * weld**2 * weld_length + 0.04811 * thickness * width * (14 + weld_length),
        "shear": shear,
        "bending": 6 * LOAD * LENGTH / (width * thickness**2),
        "buckling_load": buckling,
        "deflection": 4 * LOAD * LENGTH**3 / (YOUNG * thickness**3 * width),
    }


if __name__ == "__main__":
    result = BayesianOptimizer(search_space, BayesianConfig(max_evaluations=150, seed=0)).optimize(analyse)
    design, reported = result.best_params, result.best_outputs
    assert reported is not None
    print(f"cost {result.best_fitness:.4f}, {result.best_fitness / 1.724852 - 1:.2%} above the best published design")
    print(", ".join(f"{name} {value:.4g}" for name, value in design.items()))
    print(f"shear {reported['shear']:.0f} psi, bending {reported['bending']:.0f} psi, "
          f"buckling load {reported['buckling_load']:.0f} lb, deflection {reported['deflection']:.4f} in")
    print(f"{result.infeasible} of {result.evaluations} designs analysed broke a reported limit")
