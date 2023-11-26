from dataclasses import dataclass, field
from pathlib import Path
import json
import yaml

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


    @classmethod
    def from_jsonfile(cls, filepath: Path):
        """Create a GeneticOptimizerConfig from a file."""
        try:

            with open(filepath, 'r') as f:
                config = cls(**json.loads(f.read()))
                f.close()

            return config

        except Exception as e:
            print(f'Error loading config file: {e}')


    @classmethod
    def from_dict(cls, config_dict: dict):
        """Create a GeneticOptimizerConfig from a dictionary."""
        try:
            return cls(**config_dict)
        except Exception as e:
            print(f'Error loading config dict: {e}')

    @classmethod
    def from_yamlfile(cls, filepath: Path):
        """Create a GeneticOptimizerConfig from a YAML file."""
        
        try:
            with open(filepath, 'r') as f:
                config = cls(**yaml.safe_load(f.read()))
            return config
        
        except Exception as e:
            print(f'Error loading config file: {e}')