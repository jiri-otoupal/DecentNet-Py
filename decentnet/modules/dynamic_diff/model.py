import time

import numpy as np
import tensorflow.compat.v2 as tf
from keras.layers import Dense, Dropout, Conv1D

from decentnet.modules.pow.difficulty import Difficulty
from decentnet.modules.pow.pow import PoW

tf.enable_v2_behavior()

import tensorflow_probability as tfp

tfd = tfp.distributions


def posterior_mean_field(kernel_size, bias_size=0.5, dtype=None):
    n = kernel_size + bias_size
    c = np.log(np.expm1(1.))
    return tf.keras.Sequential([
        tfp.layers.VariableLayer(2 * n, dtype=dtype),
        tfp.layers.DistributionLambda(lambda t: tfd.Independent(
            tfd.Normal(loc=t[..., :n],
                       scale=1e-5 + tf.nn.softplus(c + t[..., n:])),
            reinterpreted_batch_ndims=1)),
    ])


def prior_trainable(kernel_size, bias_size=0, dtype=None):
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
        b = time.time()
        pw = PoW(0x1215245645132123152354564541, diff).compute()
        a = round((time.time() - b) * 1000)

        res.append(a)

    y_train = np.array(res)

    np.savetxt("y.txt", y_train)

epochs = round(samples + 0.15 * samples)

model = tf.keras.Sequential([
    tfp.layers.DenseVariational(5, posterior_mean_field, prior_trainable,
                                kl_weight=1 / x_train.shape[0]),
    tfp.layers.DistributionLambda(
        lambda t: tfd.Normal(loc=t[..., :1],
                             scale=1e-3 + tf.math.softplus(0.1 * t[..., 1:]))),
    Dense(1, activation="gelu")
])

model.compile(loss='mean_absolute_error',
              optimizer=tf.keras.optimizers.Adam(0.001),
              metrics=[tf.keras.metrics.Accuracy()])

model.fit(x_train, y_train, epochs=epochs)

print(model.summary())

l = [32, 64, 1, 64, 25]

b = time.time()
PoW(0x1215245645132123152354564541, Difficulty(*l)).compute()
ctime = round((time.time() - b) * 1000)

print(f"Actual {ctime}")
predict = model.predict(np.array([l]))
print(f"Predicted {predict}")
print(f"Diff {ctime - predict}")
