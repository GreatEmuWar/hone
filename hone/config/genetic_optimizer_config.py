from dataclasses import dataclass, field

@dataclass
class GeneticOptimizerConfig:
    """Configuration for the genetic optimizer."""
    population_size: int = field(default=100)
    generations: int = field(default=25)
    mutation_rate: float = field(default=0.05)
    mutation_threshold: float = field(default=0.1)
    crossover_rate: float = field(default=0.8)
    elitism: bool = field(default=True)
    elitism_rate: float = field(default=0.1)
    selection_method: str = field(default='tournament')
    tournament_size: int = field(default=4)
    tournament_p: float = field(default=0.75)
    selection_p: float = field(default=0.5)
    crossover_method: str = field(default='single_point')
    maximize: bool = field(default=True)
    threshold: float | int | None = field(default=None)