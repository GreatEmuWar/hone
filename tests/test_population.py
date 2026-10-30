from hone.optimizers import Individual, Population


def test_individual_stores_params_and_unique_id():
    a, b = Individual(x=1, y=2), Individual()
    assert a.params == {"x": 1, "y": 2}
    assert b.params == {}
    assert a.fitness is None
    assert a.individual_id != b.individual_id


def test_population_tracks_size():
    population = Population()
    population.add_individual(Individual(x=1))
    population.add_individuals([Individual(x=2), Individual(x=3)])
    assert len(population) == 3
    assert [ind.params["x"] for ind in population] == [1, 2, 3]
    assert population[1].params["x"] == 2


def test_population_size_from_constructor():
    assert len(Population(individuals=[Individual(), Individual()])) == 2


def test_population_ids_are_unique():
    assert Population().population_id != Population().population_id
