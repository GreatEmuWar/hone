# Examples

Each example is a complete script in the repository's [`examples`](https://github.com/GreatEmuWar/hone/tree/main/examples) folder. Run one with `python examples/pressure_vessel.py`. The model-tuning examples need scikit-learn: `pip install scikit-learn`.

| Example | Shows |
| --- | --- |
| [Pressure vessel](pressure_vessel.md) | Integer and continuous parameters under expression constraints, with `choose_optimizer` |
| [Maintenance schedule](maintenance_schedule.md) | A noisy simulation, a choice parameter and a function constraint |
| [Choosing a model](choose_a_model.md) | Conditional parameters: a classifier and its settings in one search |
| [Tuning on iris](tune_iris.md) | Bayesian optimization of log-scale hyperparameters |
| [Cantilever beam](cantilever_beam.md) | Three objectives and choosing a design from the front |
