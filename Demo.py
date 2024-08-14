"""
Variational Autoencoder (VAE) and Hamiltonian Variational Autoencoder (HVAE) implementation.

This module contains PyTorch implementations of VAE and HVAE models for unsupervised learning
and generative modeling tasks.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as dist
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt

class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) model.

    This class implements a VAE with convolutional layers for encoding and decoding.
    It can be used for unsupervised learning tasks on image data.
    """

    def __init__(self, args, avg_logit):
        """
        Initialize the VAE model.

        Args:
            args: An object containing model hyperparameters.
            avg_logit (float): The average logit value for initialization.
        """

        super(VAE, self).__init__()
        self.z_dim = args.z_dim
        self.avg_logit = avg_logit
        self._set_net_params()

    def get_elbo(self, x, args):
        """
        Compute the Evidence Lower BOund (ELBO) for the given input.

        Args:
            x (torch.Tensor): Input data.
            args: An object containing model hyperparameters.

        Returns:
            torch.Tensor: The computed ELBO.
        """

        # Encode input data x to obtain approximate posterior q(z|x)
        q_mu, q_sigma = self._inf_network(x)
        q_z = dist.Normal(q_mu, q_sigma)

        # Sample z from the approximate posterior q(z|x)
        # (Reparameterization trick to sample z from q(z|x))
        z = q_z.rsample()
        p_x_given_z_logits = self._gen_work(z)
        p_z = dist.Normal(torch.zeros_like(q_mu), torch.ones_like(q_sigma))

        # Compute KL divergence and expected log-likelihood
        kl = dist.kl_divergence(q_z, p_z).sum(dim=1)
        expected_log_likelihood = self._bernoulli_log_likelihood(x, p_x_given_z_logits)
        elbo = expected_log_likelihood - kl

        return elbo

    def get_nll(self, x, args):
        """
        Compute the Negative Log-Likelihood (NLL) for the given input.

        Args:
            x (torch.Tensor): Input data.
            args: An object containing model hyperparameters.

        Returns:
            torch.Tensor: The computed NLL.
        """
        # Encode input data x to obtain approximate posterior q(z|x)
        q_mu, q_sigma = self._inf_network(x)
        q_z = dist.Normal(q_mu, q_sigma)

        # Sample z from the approximate posterior q(z|x)
        # (Reparameterization trick to sample z from q(z|x))
        z = q_z.rsample()
        p_x_given_z_logits = self._gen_work(z)
        expected_log_likelihood = self._bernoulli_log_likelihood(x, p_x_given_z_logits)

        # Compute prior minus generator term
        prior_minus_gen = (-0.5 * (z**2).sum(1)
                           + q_sigma.log().sum(1)
                           + 0.5 * (((z - q_mu) / q_sigma)**2).sum(1))
        nll_samples = expected_log_likelihood + prior_minus_gen

        # Reshape and compute the log-sum-exp for importance sampling
        nll_samples_reshaped = nll_samples.view(args.n_IS, args.n_batch_test)
        nll_lse = torch.logsumexp(nll_samples_reshaped, dim=0)
        nll = np.log(args.n_IS) - nll_lse.mean()

        return nll




















