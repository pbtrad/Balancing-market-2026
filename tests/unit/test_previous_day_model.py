import pytest

from ml.baseline.previous_day_model import PreviousDayBaselineModel


def test_previous_day_model_repeats_prior_day_pattern() -> None:
    model = PreviousDayBaselineModel()
    history = [float(i) for i in range(96)] + [float(100 + i) for i in range(96)]

    forecast = model.forecast(values=history, horizon_steps=8, steps_per_day=96)

    assert forecast == [float(100 + i) for i in range(8)]


def test_previous_day_model_falls_back_to_last_observation() -> None:
    model = PreviousDayBaselineModel()
    history = [1.0, 2.0, 3.0]

    forecast = model.forecast(values=history, horizon_steps=2, steps_per_day=96)

    assert forecast == [3.0, 3.0]


def test_previous_day_model_validates_input() -> None:
    model = PreviousDayBaselineModel()
    with pytest.raises(ValueError):
        model.forecast(values=[], horizon_steps=1, steps_per_day=96)
