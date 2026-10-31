"""Hypothesis strategies for hone's parameters and search spaces, shared by the property tests."""
import math

from hypothesis import strategies as st

from hone.parameters import ChoiceParameter, FixedParameter, RangeParameter, SearchSpace

names = st.text(alphabet="abcdefghijklmnopqrstuvwxyz_", min_size=1, max_size=8)
bounded_floats = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)
choice_values = st.one_of(
    st.integers(min_value=-100, max_value=100),
    st.floats(min_value=-100, max_value=100, allow_nan=False).filter(lambda x: not float(x).is_integer()),
    st.text(min_size=1, max_size=5),
)


@st.composite
def range_parameters(draw, name=None, integer=None, log=None, degenerate=True):
    """A valid RangeParameter: any bounds, integer and log flags allowed by its validator."""
    name = name if name is not None else draw(names)
    integer = draw(st.booleans()) if integer is None else integer
    log = draw(st.booleans()) if log is None else log
    # A range is either a single value or measurably wide: one a few float ulps wide can't be searched.
    single = degenerate and draw(st.booleans())
    if log:
        low = draw(st.floats(min_value=1e-6, max_value=1e4))
        high = low if single else low * draw(st.floats(min_value=1.001, max_value=1e6))
    else:
        low = draw(bounded_floats)
        high = low if single else low + max(draw(st.floats(min_value=1e-3, max_value=1e6)), 1e-6 * abs(low))
    if integer:
        if single:
            low = high = float(math.ceil(low))
        elif degenerate and draw(st.booleans()):
            # One integer inside a wider range, such as 0.6 to 1.3.
            k = max(math.ceil(low), 1)
            low, high = k - draw(st.floats(0.1, 0.45)), k + draw(st.floats(0.1, 0.45))
        elif math.ceil(low) >= math.floor(high):
            high = math.ceil(low) + 1.0  # at least two integers
    return RangeParameter(name, low, high, integer=integer, log=log)


@st.composite
def choice_parameters(draw, name=None, choose_n=None):
    name = name if name is not None else draw(names)
    choices = draw(st.lists(choice_values, min_size=1, max_size=6, unique_by=repr))
    # Keep choices distinct by equality too (1 and 1.0 aside, which the strategy avoids).
    choices = [c for i, c in enumerate(choices) if c not in choices[:i]]
    if choose_n is True:
        choose_n = draw(st.integers(min_value=1, max_value=len(choices)))
    return ChoiceParameter(name, choices, choose_n=choose_n or None)


@st.composite
def fixed_parameters(draw, name=None):
    name = name if name is not None else draw(names)
    return FixedParameter(name, draw(choice_values))


@st.composite
def search_spaces(draw, kinds=("range", "choice", "fixed", "choose_n"), min_size=1, max_size=5, need_range=False, integer=None):
    count = draw(st.integers(min_value=min_size, max_value=max_size))
    parameter_names = draw(st.lists(names, min_size=count, max_size=count, unique=True))
    parameters = []
    for i, name in enumerate(parameter_names):
        kind = "range" if need_range and i == 0 else draw(st.sampled_from(kinds))
        if kind == "range":
            parameters.append(draw(range_parameters(name=name, integer=integer, degenerate=not (need_range and i == 0))))
        elif kind == "choice":
            parameters.append(draw(choice_parameters(name=name)))
        elif kind == "choose_n":
            parameters.append(draw(choice_parameters(name=name, choose_n=True)))
        else:
            parameters.append(draw(fixed_parameters(name=name)))
    return SearchSpace(parameters)
