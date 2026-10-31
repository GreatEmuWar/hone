"""Design a pressure vessel of least cost: a standard constrained engineering problem.

A cylindrical vessel with hemispherical heads must hold at least 1,296,000 cubic inches. Its
shell and heads are rolled from plate that comes in sixteenths of an inch, and each must be
thick enough for the vessel's radius. The cost covers material, forming and welding. The best
published design costs 6059.71.
"""
import math

from hone import RangeParameter, SearchSpace, choose_optimizer

shell = RangeParameter("shell", 1, 99, integer=True)  # shell thickness, in sixteenths of an inch
head = RangeParameter("head", 1, 99, integer=True)  # head thickness, in sixteenths of an inch
radius = RangeParameter("radius", 10, 200)  # inner radius, in inches
length = RangeParameter("length", 10, 200)  # length of the cylindrical section, in inches

search_space = SearchSpace(
    [shell, head, radius, length],
    constraints=[
        shell / 16 >= 0.0193 * radius,  # the shell is thick enough for the hoop stress
        head / 16 >= 0.00954 * radius,  # and so are the heads
        math.pi * radius**2 * length + 4 / 3 * math.pi * radius**3 >= 1_296_000,  # it holds enough
    ],
)


def cost(shell, head, radius, length):
    ts, th = shell / 16, head / 16
    return 0.6224 * ts * radius * length + 1.7781 * th * radius**2 + 3.1661 * ts**2 * length + 19.84 * ts**2 * radius


if __name__ == "__main__":
    # With 4,000 evaluations and integer thicknesses, choose_optimizer picks the genetic optimizer.
    # A single run can settle on a design that is good locally but not the best, such as a wide,
    # short vessel, and a larger budget doesn't get it out; independent runs usually differ. So
    # run four, with different seeds, and keep the best.
    results = [choose_optimizer(search_space, max_evaluations=4000, seed=seed).optimize(cost) for seed in range(4)]
    for seed, result in enumerate(results):
        print(f"seed {seed}: cost {result.best_fitness:.2f}")
    best = min(results, key=lambda result: result.best_fitness)
    design = best.best_params
    print(f"best: cost {best.best_fitness:.2f}, {best.best_fitness / 6059.714 - 1:.2%} above the best published design")
    print(f"shell {design['shell']}/16 in, head {design['head']}/16 in, radius {design['radius']:.3f} in, length {design['length']:.3f} in")
