import random
import time

import numpy as np
import tensorflow.compat.v2 as tf
from keras.layers import Dense, Dropout, Conv1D

from decentnet.modules.pow.difficulty import Difficulty
from decentnet.modules.pow.pow import PoW

tf.enable_v2_behavior()

import tensorflow_probability as tfp

tfd = tfp.distributions


def posterior_mean_field(kernel_size, bias_size=0.1, dtype=None):
    n = kernel_size + bias_size
    c = np.log(np.expm1(1.))
    return tf.keras.Sequential([
        tfp.layers.VariableLayer(2 * n, dtype=dtype),
        tfp.layers.DistributionLambda(lambda t: tfd.Independent(
            tfd.Normal(loc=t[..., :n],
                       scale=1e-5 + tf.nn.softplus(c + t[..., n:])),
            reinterpreted_batch_ndims=1)),
    ])


def prior_trainable(kernel_size, bias_size=0.01, dtype=None):
    n = kernel_size + bias_size
    return tf.keras.Sequential([
        tfp.layers.VariableLayer(n, dtype=dtype),
        tfp.layers.DistributionLambda(lambda t: tfd.Independent(
            tfd.Normal(loc=t, scale=1),
            reinterpreted_batch_ndims=1)),
    ])


samples = 512

t = np.linspace(1, 64, num=samples)
p = np.linspace(1, 8, num=samples)
n = np.linspace(1, 16, num=samples)
m = np.linspace(8, 1024, num=samples)
h = np.linspace(20, 30, num=samples)

x_train = np.vstack([t, m, p, n, h]).transpose()

try:
    y_train = np.loadtxt("y.txt")
except:
    res = []
    for line in x_train.astype(int).tolist():
        diff = Difficulty(*line)

        print(f"Computing {diff}")
        b = time.time_ns()
        pw = PoW(0x1215245645132123152354564541, diff).compute()
        a = time.time_ns() - b

        res.append(a)

    y_train = np.array(res)

    np.savetxt("y.txt", y_train)

epochs = round(samples + 0.15 * samples)

model = tf.keras.Sequential([
    tfp.layers.DenseVariational(15, posterior_mean_field, prior_trainable,
                                kl_weight=1 / x_train.shape[0]),
    tfp.layers.DistributionLambda(
        lambda t: tfd.Normal(loc=t[..., :1],
                             scale=1e-3 + tf.math.softplus(0.01 * t[..., 1:]))),
    Dense(8, activation="relu"),
    Dense(1, activation="relu")
])

model.compile(loss='mean_absolute_error',
              optimizer=tf.keras.optimizers.Adam(0.05))

model.fit(x_train, y_train, epochs=epochs)

print(model.summary())

actual = []
diffs = []
predictions = 1000

for _ in range(predictions):
    l = [random.randint(1, 64), random.randint(8, 1000), random.randint(1, 8),
         random.randint(1, 20), 25]

    b = time.time_ns()
    PoW(0x1215245645132123152354564541, Difficulty(*l)).compute()
    ctime = time.time_ns() - b

    print(f"Actual {ctime}")
    predict = model.predict(np.array([l]))
    print(f"Predicted {predict}")
    diff = ctime - predict
    print(f"Diff {diff}")
    diffs.append(diff)
    actual.append(ctime)
    # model.fit([l], [ctime], epochs=50) # Uncomment for continuous learning

# nanoseconds to milli
diffs_ = abs(np.mean(diffs) / 1000000)
actual_ = np.mean(actual) / 1000000

print(f"Mean Actual Computation time {actual_} ms")

print(f"Mean diff {diffs_} ms for {predictions=}")

print(f"Mean Variance {(diffs_ / actual_) * 100} %")
