"""E4: Keras activation->strategy table mirrors strategies.py (D3):
exact class or activation-function identity; unmapped -> explicit error."""
import pytest

tf = pytest.importorskip("tensorflow")
from tensorflow import keras  # noqa: E402

from entropylens import strategies_tf  # noqa: E402
from entropylens.strategies import UnmappedActivationError  # noqa: E402


def test_layer_class_mappings():
    assert strategies_tf.strategy_for_keras(keras.layers.ReLU()) == "relu"
    assert strategies_tf.strategy_for_keras(keras.layers.LeakyReLU()) == "relu"
    assert strategies_tf.strategy_for_keras(keras.layers.Softmax()) == "relu"


def test_activation_attribute_mappings():
    assert strategies_tf.strategy_for_keras(
        keras.layers.Dense(4, activation="relu")) == "relu"
    assert strategies_tf.strategy_for_keras(
        keras.layers.Dense(4, activation="tanh")) == "abs"
    assert strategies_tf.strategy_for_keras(
        keras.layers.Activation("sigmoid")) == "relu"
    assert strategies_tf.strategy_for_keras(
        keras.layers.Activation("softsign")) == "abs"
    assert strategies_tf.strategy_for_keras(
        keras.layers.Dense(4, activation="gelu")) == "relu"


def test_linear_dense_is_unmapped():
    with pytest.raises(UnmappedActivationError) as e:
        strategies_tf.strategy_for_keras(keras.layers.Dense(4))
    msg = str(e.value)
    assert "explicit" in msg and "relu" in msg and "abs" in msg


def test_every_entry_has_motivation():
    strategies_tf._ensure_tables()
    for table in (strategies_tf._LAYER_TABLE, strategies_tf._FN_TABLE):
        for key, (strat, why) in table.items():
            assert strat in ("relu", "abs")
            assert len(why) > 20
