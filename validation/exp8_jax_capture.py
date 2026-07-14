"""EXP8 (E5): cattura Flax via capture_intermediates — invarianza bitwise.

Protocollo: MLP flax Dense(32)+relu -> Dense(16)+tanh -> Dense(2);
apply() con e senza capture_intermediates sugli stessi parametri e input.

Gate:
  G1  output bitwise-identici con/senza cattura, SENZA jit (deve passare)
  G2  output bitwise-identici con/senza cattura, SOTTO jax.jit
      (atteso PASS; contingenza sotto se fallisce)
  G3  gradienti dei parametri bitwise-identici con/senza cattura nel
      value_and_grad (la collection 'intermediates' non entra nel grad path)
  G4  struttura intermediates: path 'Dense_0'/'Dense_1'/'Dense_2' presenti,
      output PRE-attivazione (Dense_0 contiene negativi anche se segue relu)

Contingenza G2 (non silenziosa): se G2 fallisce, stampare il massimo scarto
assoluto, declassare il gate a informativo SOLO cambiando la riga del check in
`check("G2 (informativo, jit non bitwise)", True, f"max |d|={d}")` e
registrare in D9 che sotto jit la garanzia e' allclose, non bitwise.
"""
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

failures = []


def check(name, cond, detail):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}: {detail}")
    if not cond:
        failures.append(name)


class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Dense(32)(x))
        x = jnp.tanh(nn.Dense(16)(x))
        return nn.Dense(2)(x)


model = MLP()
key = jax.random.PRNGKey(7)
x = jax.random.normal(jax.random.PRNGKey(0), (64, 20))
params = model.init(key, x)

# --- G1: no jit
y_plain = model.apply(params, x)
y_cap, mut = model.apply(params, x, capture_intermediates=True,
                         mutable=["intermediates"])
check("G1 outputs bitwise (no jit)",
      bool(jnp.array_equal(y_plain, y_cap)), "identical")

# --- G2: under jit
f_plain = jax.jit(lambda p, xx: model.apply(p, xx))
f_cap = jax.jit(lambda p, xx: model.apply(
    p, xx, capture_intermediates=True, mutable=["intermediates"])[0])
yj_plain = f_plain(params, x)
yj_cap = f_cap(params, x)
d = float(jnp.max(jnp.abs(yj_plain - yj_cap)))
check("G2 outputs bitwise (jit)",
      bool(jnp.array_equal(yj_plain, yj_cap)), f"max |d| = {d:.2e}")

# --- G3: gradients
y_target = jax.random.normal(jax.random.PRNGKey(1), (64, 2))


def loss_plain(p):
    return jnp.mean((model.apply(p, x) - y_target) ** 2)


def loss_cap(p):
    y, _ = model.apply(p, x, capture_intermediates=True,
                       mutable=["intermediates"])
    return jnp.mean((y - y_target) ** 2)


g_plain = jax.grad(loss_plain)(params)
g_cap = jax.grad(loss_cap)(params)
flat_p, _ = jax.tree_util.tree_flatten(g_plain)
flat_c, _ = jax.tree_util.tree_flatten(g_cap)
same = all(bool(jnp.array_equal(a, b)) for a, b in zip(flat_p, flat_c))
check("G3 gradients bitwise", same, "capture does not touch grad path")

# --- G4: structure + pre-activation semantics
inter = mut["intermediates"]
names = sorted(k for k in inter if k != "__call__")
check("G4 paths", names == ["Dense_0", "Dense_1", "Dense_2"], f"{names}")
d0 = np.asarray(inter["Dense_0"]["__call__"][0])
check("G4 pre-activation (Dense_0 has negatives)",
      bool((d0 < 0).any()),
      "captured BEFORE relu -> strategy_map must be explicit (D8)")

print("\n" + ("ALL PASS" if not failures else f"FAILURES: {failures}"))
raise SystemExit(0 if not failures else 1)
