"""A5/C2: explicit activation->strategy table with motivations; unmapped
modules must produce an explicit error, never a silent default."""
import pytest
import torch.nn as nn

from entropylens import strategies


def test_table_mappings():
    assert strategies.strategy_for(nn.ReLU()) == "relu"
    assert strategies.strategy_for(nn.ReLU6()) == "relu"
    assert strategies.strategy_for(nn.Softplus()) == "relu"
    assert strategies.strategy_for(nn.Sigmoid()) == "relu"
    assert strategies.strategy_for(nn.Softmax(dim=-1)) == "relu"
    assert strategies.strategy_for(nn.GELU()) == "relu"
    assert strategies.strategy_for(nn.SiLU()) == "relu"
    assert strategies.strategy_for(nn.LeakyReLU()) == "relu"
    assert strategies.strategy_for(nn.Tanh()) == "abs"
    assert strategies.strategy_for(nn.Softsign()) == "abs"
    assert strategies.strategy_for(nn.Hardtanh()) == "abs"


def test_unmapped_module_raises_with_guidance():
    with pytest.raises(strategies.UnmappedActivationError) as e:
        strategies.strategy_for(nn.Linear(4, 4))
    msg = str(e.value)
    assert "explicit" in msg and "relu" in msg and "abs" in msg


def test_every_entry_has_motivation():
    for typ, (strat, why) in strategies.DEFAULT_STRATEGIES.items():
        assert strat in ("relu", "abs")
        assert len(why) > 20, f"{typ.__name__}: motivation missing"


def test_exact_type_match_not_subclass():
    class MyReLU(nn.ReLU):
        pass
    with pytest.raises(strategies.UnmappedActivationError):
        strategies.strategy_for(MyReLU())
