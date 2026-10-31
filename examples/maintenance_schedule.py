"""Choose a preventive maintenance schedule for a production line, scored by simulating a year of it.

Three machines run in series, so a breakdown of any one stops the line. Each machine is more
likely to break down the longer it has gone since its last service. The schedule sets how often
each is serviced, which shift the service happens on, and how many spare-part kits to keep:

- night: no production is lost, but technicians are paid 1.5 times as much
- weekend: no production is lost, technicians are paid twice as much, and the service waits for Saturday
- any: the line stops while the service is done, at the ordinary rate

A breakdown costs repairs and lost production, and takes far longer to fix when no spare kit is
in stock; a kit used is replaced 30 days later. Every simulated year comes out differently, so
the optimizer runs with noise=True.
"""
import numpy as np

from hone import ChoiceParameter, Constraint, RangeParameter, SearchSpace, choose_optimizer

# For each machine: hours a service takes, its characteristic life in days and the shape of its
# wear (a Weibull distribution: above 1, breakdowns become likelier with age), and hours to repair
# a breakdown with a spare kit to hand.
MACHINES = {"press": (6, 60, 2.5, 4), "oven": (4, 120, 2.0, 6), "packer": (3, 45, 1.5, 3)}
LOST_PER_HOUR = 2_000  # production lost while the line is stopped
REPAIR_COST = 5_000  # parts and call-out for each breakdown
WAIT_FOR_PARTS = 48  # extra hours stopped when a breakdown finds no spare kit
TECHNICIAN_RATE = 80  # per technician hour, at the ordinary rate
PAY = {"night": 1.5, "weekend": 2.0, "any": 1.0}
KIT_PER_YEAR = 1_500  # the cost of keeping one kit in stock
RESTOCK_DAYS = 30
TECHNICIAN_HOURS = 300  # hours a year the technicians have for planned work
PLANNED_STOP_HOURS = 40  # hours a year production agrees to stop for planned work

press = RangeParameter("press_days", 7, 180, integer=True)  # days between services
oven = RangeParameter("oven_days", 7, 180, integer=True)
packer = RangeParameter("packer_days", 7, 180, integer=True)
shift = ChoiceParameter("service_shift", ["night", "weekend", "any"])
spares = RangeParameter("spare_kits", 0, 6, integer=True)


def planned_stops(press_days, oven_days, packer_days, service_shift):
    """Hours a year the line stops for services beyond what production agreed, if they happen during production."""
    if service_shift != "any":
        return -1.0  # services happen outside production
    hours = 365 * (6 / press_days + 4 / oven_days + 3 / packer_days)
    return hours - PLANNED_STOP_HOURS


search_space = SearchSpace(
    [press, oven, packer, shift, spares],
    constraints=[
        # Planned services must fit in the technicians' hours.
        365 * (6 / press + 4 / oven + 3 / packer) <= TECHNICIAN_HOURS,
        # A function, since it depends on a choice parameter.
        Constraint.function(planned_stops),
    ],
)


def simulate_year(press_days, oven_days, packer_days, service_shift, spare_kits, rng):
    """The cost of running the line for a year on this schedule."""
    intervals = {"press": press_days, "oven": oven_days, "packer": packer_days}
    age = dict.fromkeys(MACHINES, 0)  # days since each machine was serviced or repaired
    due = dict.fromkeys(MACHINES, False)  # waiting for the weekend to be serviced
    kits, restocks = spare_kits, []
    cost = KIT_PER_YEAR * spare_kits
    for day in range(365):
        kits += sum(1 for arrives in restocks if arrives == day)
        for name, (service_hours, life, shape, repair_hours) in MACHINES.items():
            if age[name] >= intervals[name]:
                due[name] = True
            if due[name] and (service_shift != "weekend" or day % 7 == 5):
                cost += service_hours * TECHNICIAN_RATE * PAY[service_shift]
                if service_shift == "any":
                    cost += service_hours * LOST_PER_HOUR
                age[name], due[name] = 0, False
                continue
            # The chance of breaking down today, given it has lasted age[name] days.
            wear = ((age[name] + 1) / life) ** shape - (age[name] / life) ** shape
            if rng.random() < 1 - np.exp(-wear):
                stopped = repair_hours
                if kits > 0:
                    kits -= 1
                    restocks.append(day + RESTOCK_DAYS)
                else:
                    stopped += WAIT_FOR_PARTS
                cost += REPAIR_COST + stopped * (LOST_PER_HOUR + TECHNICIAN_RATE)
                age[name] = 0
            else:
                age[name] += 1
    return cost


_rng = np.random.default_rng(0)  # each evaluation simulates a different year, as each real year would differ


def yearly_cost(press_days, oven_days, packer_days, service_shift, spare_kits):
    return simulate_year(press_days, oven_days, packer_days, service_shift, spare_kits, _rng)


def expected_cost(schedule, years=500):
    """The average cost of a schedule over many fresh simulated years, for comparing schedules fairly."""
    rng = np.random.default_rng(12345)
    return float(np.mean([simulate_year(**schedule, rng=rng) for _ in range(years)]))


def plan(max_evaluations=150, seed=0):
    # 150 evaluations is the budget for a slow simulator, so choose_optimizer picks Bayesian optimization.
    # Here it found a better schedule in 150 simulated years than the genetic optimizer did in 3,000.
    optimizer = choose_optimizer(search_space, max_evaluations=max_evaluations, noise=True, seed=seed)
    return optimizer, optimizer.optimize(yearly_cost)


if __name__ == "__main__":
    by_hand = {"press_days": 30, "oven_days": 30, "packer_days": 30, "service_shift": "night", "spare_kits": 2}
    optimizer, result = plan()
    found = result.best_params
    print(f"{type(optimizer).__name__}, {result.evaluations} simulated years")
    print(f"found:   {found}")
    print(f"by hand: {by_hand}")
    print(f"expected yearly cost over 500 fresh years: found {expected_cost(found):,.0f}, by hand {expected_cost(by_hand):,.0f}")
