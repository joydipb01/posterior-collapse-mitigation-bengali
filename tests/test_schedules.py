from bcvae.training.schedules import beta_at


def test_linear_warmup():
    assert beta_at(0, schedule="linear", warmup_steps=10) == 0.0
    assert beta_at(5, schedule="linear", warmup_steps=10) == 0.5
    assert beta_at(10, schedule="linear", warmup_steps=10) == 1.0
    assert beta_at(99, schedule="linear", warmup_steps=10) == 1.0


def test_constant():
    assert beta_at(0, schedule="constant", warmup_steps=0) == 1.0


def test_cyclical_restarts():
    kwargs = dict(schedule="cyclical", warmup_steps=0, total_steps=100, cycles=4, ratio=0.5)
    assert beta_at(0, **kwargs) == 0.0
    assert beta_at(12, **kwargs) == 1.0
    assert beta_at(25, **kwargs) == 0.0  # next cycle
    assert beta_at(30, **kwargs) < 1.0
