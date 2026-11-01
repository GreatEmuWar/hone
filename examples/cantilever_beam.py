"""Design a cantilever beam that is light, cheap and stiff: three objectives, and the front of best trade-offs.

A beam of rectangular section, one metre long, is fixed at one end and carries 5 kN at the other.
Steel is cheap but heavy, carbon fibre light and stiff but expensive, and aluminium in between,
and any material bends less the more of it there is. So there is no single best design. Instead
of fixing a weighting of mass, cost and deflection in advance, the search finds every design for
which nothing found is better on all three, and the choice is made afterwards: the stiffest beam
under a mass and a cost limit, or the most balanced. Each material must also keep its bending
stress below its strength, with a safety factor; that limit is a function constraint, since it
depends on the material chosen.
"""
from hone import ChoiceParameter, Constraint, RangeParameter, SearchSpace, choose_pareto_optimizer

LENGTH = 1.0  # m
LOAD = 5_000.0  # N, at the free end
SAFETY = 1.5

# Young's modulus (Pa), density (kg/m³), strength (Pa) and a rough price (per kg) of each material.
MATERIALS = {
    "steel": (200e9, 7850, 250e6, 1.5),
    "aluminium": (69e9, 2700, 276e6, 4.0),
    "carbon fibre": (135e9, 1600, 1000e6, 40.0),  # a unidirectional laminate along the beam
}

material = ChoiceParameter("material", list(MATERIALS))
width = RangeParameter("width", 10, 100)  # mm
height = RangeParameter("height", 10, 150)  # mm


def stress_margin(material, width, height):
    """The bending stress at the fixed end, less the material's allowed stress, in MPa."""
    b, h = width / 1000, height / 1000
    stress = 6 * LOAD * LENGTH / (b * h**2)
    return (stress - MATERIALS[material][2] / SAFETY) / 1e6


search_space = SearchSpace([material, width, height], constraints=[Constraint.function(stress_margin)])


def beam(material, width, height):
    """The beam's mass in kg, its material cost and its tip deflection in mm."""
    modulus, density, _, price = MATERIALS[material]
    b, h = width / 1000, height / 1000
    mass = density * b * h * LENGTH
    deflection = LOAD * LENGTH**3 / (3 * modulus * b * h**3 / 12)
    return {"mass": mass, "cost": price * mass, "deflection": 1000 * deflection}


def describe(point):
    p, v = point.params, point.values
    return (
        f"{p['material']:12} {p['width']:5.1f} × {p['height']:5.1f} mm: "
        f"{v['mass']:5.2f} kg, costs {v['cost']:6.1f}, deflects {v['deflection']:5.2f} mm"
    )


if __name__ == "__main__":
    objectives = {"mass": "minimize", "cost": "minimize", "deflection": "minimize"}
    optimizer = choose_pareto_optimizer(search_space, objectives, max_evaluations=3000, seed=0)  # NSGA2, for this budget
    front = optimizer.optimize(beam).front
    print(f"front of {len(front.points)} designs; the lightest for each material:")
    for name in MATERIALS:
        designs = [p for p in front.points if p.params["material"] == name]
        if designs:
            print("  " + describe(min(designs, key=lambda point: point.values["mass"])))
    print("stiffest under 10 kg and 100: " + describe(front.best("deflection", where={"mass": 10, "cost": 100})))
    print("most balanced:                " + describe(front.closest()))
