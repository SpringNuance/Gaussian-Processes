import gpflow
import tensorflow as tf
from gpflow.base import Module
from gpflow.mean_functions import Zero
from gpflow.models.model import MeanAndVariance

from layer_initializations import init_layers_linear


class DeepGPBase(Module):

    def __init__(self, likelihood, layers, num_samples=1, **kwargs):
        super().__init__(name="DeepGPBase")
        self.num_samples = num_samples
        self.likelihood = likelihood
        self.layers = layers

    def propagate(self, X, full_cov=False, num_samples=1, zs=None):
        sX = tf.tile(tf.expand_dims(X, 0), [num_samples, 1, 1])
        Fs, Fmeans, Fvars = [], [], []
        F = sX
        zs = zs or [None, ] * len(self.layers)
        for layer, z in zip(self.layers, zs):
            F, Fmean, Fvar = layer.sample_from_conditional(F, z=z, full_cov=full_cov)

            Fs.append(F)
            Fmeans.append(Fmean)
            Fvars.append(Fvar)
        return Fs, Fmeans, Fvars

    def predict_f(self, predict_at, num_samples, full_cov=False) -> MeanAndVariance:
        Fs, Fmeans, Fvars = self.propagate(predict_at, full_cov=full_cov,
                                           num_samples=num_samples)
        return Fmeans[-1], Fvars[-1]

    def predict_all_layers(self, predict_at, num_samples, full_cov=False):
        return self.propagate(predict_at, full_cov=full_cov,
                              num_samples=num_samples)

    def predict_y(self, predict_at, num_samples):
        Fmean, Fvar = self.predict_f(predict_at, num_samples=num_samples,
                                     full_cov=False)
        return self.likelihood.predict_mean_and_var(Fmean, Fvar)

    def predict_log_density(self, data, num_samples):
        Fmean, Fvar = self.predict_f(data[0], num_samples=num_samples,
                                     full_cov=False)
        l = self.likelihood.predict_density(Fmean, Fvar, data[1])
        log_num_samples = tf.math.log(tf.cast(self.num_samples, gpflow.base.default_float()))
        return tf.reduce_logsumexp(l - log_num_samples, axis=0)


class DeepGP(DeepGPBase):
    """
    This is the Doubly-Stochastic Deep GP, with linear/identity mean functions at each layer.

    The key reference is

    ::
      @inproceedings{salimbeni2017doubly,
        title={Doubly Stochastic Variational Inference for Deep Gaussian Processes},
        author={Salimbeni, Hugh and Deisenroth, Marc},
        booktitle={NIPS},
        year={2017}
      }

    """
    def __init__(self, X, Y, Z, kernels, layer_sizes, likelihood,
                 num_outputs=None, mean_function=Zero(), whiten=False,
                 num_samples=1):
        layers = init_layers_linear(X, Y, Z, kernels, layer_sizes,
                                    mean_function=mean_function,
                                    num_outputs=num_outputs,
                                    whiten=whiten)
        super().__init__(likelihood, layers, num_samples)
        self.num_data = X.shape[0]
