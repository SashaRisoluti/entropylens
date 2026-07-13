"""EXP7 (E3): cattura TF via modello ausiliario — invarianza per costruzione.

Protocollo: MLP funzionale Dense(32,relu) -> Dense(16,tanh) -> Dense(2);
modello ausiliario keras.Model(model.inputs[0], {name: layer.output}).

Gate:
  G1  output del modello ausiliario == ricalcolo manuale strato per strato
      (bitwise, eager)
  G2  output del modello utente identico prima/dopo la creazione dell'aux e
      dopo observe() (bitwise) — B1 per costruzione, congelato
  G3  pesi dopo 3 step di training identici bitwise con/senza observe() tra
      gli step (stesso seed) — B3 per costruzione, congelato
  G4  l'aux condivide le variabili (stessi oggetti, nessuna copia)
  G5  (ancora negativa attesa) con BatchNorm: observe(training=False) !=
      attivazioni del passo di training — documenta la semantica di
      osservazione inference-mode (D8)
  G6  funziona anche su Sequential (model.inputs esiste dopo build)
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras

failures = []


def check(name, cond, detail):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}: {detail}")
    if not cond:
        failures.append(name)


def make_mlp():
    keras.utils.set_random_seed(7)
    inp = keras.Input((20,))
    h1 = keras.layers.Dense(32, activation="relu", name="d1")(inp)
    h2 = keras.layers.Dense(16, activation="tanh", name="d2")(h1)
    out = keras.layers.Dense(2, name="d3")(h2)
    return keras.Model(inp, out)


rng = np.random.default_rng(0)
x = rng.standard_normal((64, 20)).astype(np.float32)

# --- G1 + G4
# Keras 3: model.inputs e' una LISTA anche a input singolo; si passa
# inputs[0] all'aux per evitare lo structure-warning ad ogni chiamata
model = make_mlp()
aux = keras.Model(model.inputs[0],
                  {l.name: l.output for l in model.layers
                   if not isinstance(l, keras.layers.InputLayer)})
outs = aux(x, training=False)
h = x
for name in ("d1", "d2", "d3"):
    h = model.get_layer(name)(h)
    check(f"G1 aux == manual ({name})",
          bool(tf.reduce_all(tf.equal(outs[name], h))),
          "bitwise equal" )
check("G4 shared variables",
      aux.get_layer("d1").kernel is model.get_layer("d1").kernel,
      "same variable object, no copy")

# --- G2
y0 = model(x, training=False)
_ = aux(x, training=False)          # observe
y1 = model(x, training=False)
check("G2 user output invariant", bool(tf.reduce_all(tf.equal(y0, y1))),
      "bitwise equal before/after observe")

# --- G3: 3 training steps with/without observation between steps
def train(observe):
    m = make_mlp()
    a = keras.Model(m.inputs[0], {l.name: l.output for l in m.layers
                                  if not isinstance(l, keras.layers.InputLayer)})
    opt = keras.optimizers.SGD(0.1)
    xb = tf.constant(x)
    yb = tf.constant(rng2.integers(0, 2, size=(64,)))
    for _ in range(3):
        with tf.GradientTape() as tape:
            logits = m(xb, training=True)
            loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(yb, logits))
        grads = tape.gradient(loss, m.trainable_variables)
        opt.apply_gradients(zip(grads, m.trainable_variables))
        if observe:
            _ = a(xb, training=False)
    return [v.numpy() for v in m.trainable_variables]


rng2 = np.random.default_rng(1)
w_plain = train(False)
rng2 = np.random.default_rng(1)
w_obs = train(True)
same = all(np.array_equal(a, b) for a, b in zip(w_plain, w_obs))
check("G3 weights bitwise invariant", same,
      "3 SGD steps, observe() between steps changes nothing")

# --- G5: BatchNorm observation semantics (expected DIFFERENCE)
keras.utils.set_random_seed(7)
inp = keras.Input((20,))
hbn = keras.layers.Dense(32, activation="relu", name="d1")(inp)
hbn = keras.layers.BatchNormalization(name="bn")(hbn)
outbn = keras.layers.Dense(2, name="d3")(hbn)
mbn = keras.Model(inp, outbn)
abn = keras.Model(mbn.inputs[0], {"bn": mbn.get_layer("bn").output})
train_act = None
# training-pass activation: call bn layer chain with training=True
t1 = mbn.get_layer("d1")(x)
train_act = mbn.get_layer("bn")(t1, training=True)
obs_act = abn(x, training=False)["bn"]
check("G5 BN: observe != training pass (expected)",
      not bool(tf.reduce_all(tf.equal(train_act, obs_act))),
      "inference-mode observation documented in D8")

# --- G6: Sequential
keras.utils.set_random_seed(7)
seq = keras.Sequential([keras.Input((20,)),
                        keras.layers.Dense(8, activation="relu", name="s1"),
                        keras.layers.Dense(2, name="s2")])
aux_seq = keras.Model(seq.inputs[0], {l.name: l.output for l in seq.layers})
o = aux_seq(x, training=False)
check("G6 Sequential capture", set(o) == {"s1", "s2"} and o["s1"].shape == (64, 8),
      f"captured {sorted(o)}")

print("\n" + ("ALL PASS" if not failures else f"FAILURES: {failures}"))
raise SystemExit(0 if not failures else 1)
