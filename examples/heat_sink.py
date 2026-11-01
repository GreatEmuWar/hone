"""Find which of a heat sink's eight settings matter, fix the rest, and optimize what's left.

A 40 W chip sits under a finned heat sink, 60 mm square, with a fan blowing air along the fins.
The design has eight settings, from the number of fins to the surface finish and the number of
mounting screws, and the aim is the coolest chip. A thermal model scores each design; a design
whose fins don't fit on the base fails, as a real simulation might.

Rather than optimize all eight at once, sensitivity() measures first which settings drive the
chip's temperature. report.reduced() then fixes the ones that don't, at the best design measured,
and the optimizer searches only those that matter. With a budget of 60 evaluations, as a slow
simulation might allow, it finds a cooler design than a search over all eight.

Fixing a setting still costs something. A setting can barely change the temperature across the
designs measured and still have a best value: more screws and a thinner base are always a little
better. Fixed at the best design measured, the four cost up to 0.2 °C: over five seeds, the
reduced search's median is 33.3 °C at 60 evaluations and at 300, while a search over all eight
has a median of 33.7 °C at 60 and 33.5 °C at 300, though its best run reaches 33.1 °C. To fix a
setting at a value you know is good, pass it as at=, such as at={"screws": 6}.
"""
import math

from hone import ChoiceParameter, EvaluationFailed, FixedParameter, RangeParameter, choose_optimizer, sensitivity

POWER = 40.0  # W, from the chip
WIDTH = LENGTH = 0.060  # m, the base
MATERIALS = {"aluminium": 205.0, "copper": 385.0}  # thermal conductivity, W/(m K)
EMISSIVITY = {"bare": 0.1, "anodized": 0.85}

search_space = [
    RangeParameter("fins", 4, 24, integer=True),
    RangeParameter("fin_height", 10, 50),  # mm
    RangeParameter("fin_thickness", 0.5, 2.5),  # mm
    RangeParameter("base_thickness", 2, 10),  # mm
    ChoiceParameter("material", list(MATERIALS)),
    RangeParameter("airflow", 0.5, 5, log=True),  # m/s, from the fan
    ChoiceParameter("finish", list(EMISSIVITY)),
    RangeParameter("screws", 2, 6, integer=True),
    FixedParameter("ambient", 25.0),  # °C
]


def chip_temperature(fins, fin_height, fin_thickness, base_thickness, material, airflow, finish, screws, ambient):
    """The chip's temperature in °C, from a simple model of the heat sink's thermal resistances."""
    k = MATERIALS[material]
    height, thickness, base = fin_height / 1000, fin_thickness / 1000, base_thickness / 1000
    gap = (WIDTH - fins * thickness) / (fins - 1)
    if gap < 0.0008:
        raise EvaluationFailed("the fins don't fit on the base")
    # Convection along the fins: more air helps, and gaps too narrow to let it through hurt.
    h = (8 + 22 * airflow**0.8) * (1 - math.exp(-gap / 0.002))
    m = math.sqrt(2 * h / (k * thickness))
    efficiency = math.tanh(m * height) / (m * height)
    fin_area = 2 * fins * height * LENGTH
    base_area = (WIDTH - fins * thickness) * LENGTH
    convection = h * (efficiency * fin_area + base_area)
    # Radiation from the outside of the sink only, since the fins mostly see each other.
    radiation = 6.0 * EMISSIVITY[finish] * (WIDTH * LENGTH + 2 * height * LENGTH)
    spreading = base / (k * WIDTH * LENGTH)
    contact = 0.03 + 0.04 / screws  # more screws press the sink harder onto the chip
    return ambient + POWER * (spreading + contact + 1 / (convection + radiation))


if __name__ == "__main__":
    report = sensitivity(chip_temperature, search_space, max_evaluations=2000, seed=0)
    print(report)
    print(f"fixed: {', '.join(report.unimportant())}")
    print()

    reduced = report.reduced(search_space)
    budget = 60
    focused = choose_optimizer(reduced, max_evaluations=budget, seed=0).optimize(chip_temperature)
    everything = choose_optimizer(search_space, max_evaluations=budget, seed=0).optimize(chip_temperature)
    print(f"best design measured: {report.best_value:.1f} °C")
    print(f"optimizing the settings that matter, {budget} evaluations: {focused.best_fitness:.1f} °C")
    print(f"optimizing all eight, {budget} evaluations: {everything.best_fitness:.1f} °C")
    print(focused.best_params)
