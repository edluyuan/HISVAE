import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import numpy as np


class VAE(nn.Module):
    def __init__(self, args, avg_logit):
        super(VAE, self).__init__()
        self.z_dim = args.z_dim
        self.avg_logit = avg_logit
        self._set_net_params()

    def get_elbo(self, x, args):
        q_mu, q_sigma = self._inf_network(x)
        q_z = dist.Normal(q_mu, q_sigma)

        z = q_z.rsample()
        p_x_given_z_logits = self._gen_network(z)
        p_z = dist.Normal(torch.zeros_like(z), torch.ones_like(z))

        kl = dist.kl_divergence(q_z, p_z).sum(1)
        expected_log_likelihood = self._bernoulli_log_likelihood(x, p_x_given_z_logits)
        elbo = (expected_log_likelihood - kl).mean()

        return elbo

    def get_nll(self, x, args):
        q_mu, q_sigma = self._inf_network(x)
        q_z = dist.Normal(q_mu, q_sigma)

        z = q_z.rsample()
        p_x_given_z_logits = self._gen_network(z)
        expected_log_likelihood = self._bernoulli_log_likelihood(x, p_x_given_z_logits)

        prior_minus_gen = (-0.5 * (z ** 2).sum(1)
                           + q_sigma.log().sum(1)
                           + 0.5 * (((z - q_mu) / q_sigma) ** 2).sum(1))
        nll_samples = expected_log_likelihood + prior_minus_gen

        nll_samples_reshaped = nll_samples.view(args.n_IS, args.n_batch_test)
        nll_lse = torch.logsumexp(nll_samples_reshaped, dim=0)
        nll = np.log(args.n_IS) - nll_lse.mean()

        return nll

    def get_samples(self, args):
        z_0 = torch.randn(args.n_gen_samples, self.z_dim, device=self.fc_mu.weight.device)
        logits = self._gen_network(z_0)
        samples = torch.sigmoid(logits)
        return samples

    def _set_net_params(self):
        # Inference network
        self.conv1 = nn.Conv2d(1, 16, 5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(16, 32, 5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(32, 32, 5, stride=1, padding=2)
        self.fc1 = nn.Linear(32 * 7 * 7, 450)
        self.fc_mu = nn.Linear(450, self.z_dim)
        self.fc_sigma = nn.Linear(450, self.z_dim)

        # Generative network
        self.fc_g1 = nn.Linear(self.z_dim, 450)
        self.fc_g2 = nn.Linear(450, 32 * 7 * 7)
        self.deconv1 = nn.ConvTranspose2d(32, 32, 5, stride=1, padding=2)
        self.deconv2 = nn.ConvTranspose2d(32, 16, 5, stride=2, padding=2, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(16, 1, 5, stride=2, padding=2, output_padding=1)

    def _inf_network(self, x):
        h1 = F.softplus(self.conv1(x))
        h2 = F.softplus(self.conv2(h1))
        h3 = F.softplus(self.conv3(h2))
        h3_flat = h3.view(h3.size(0), -1)
        h4 = F.softplus(self.fc1(h3_flat))
        mu = self.fc_mu(h4)
        sigma = F.softplus(self.fc_sigma(h4))
        return mu, sigma

    def _gen_network(self, z):
        h1 = F.softplus(self.fc_g1(z))
        h2 = F.softplus(self.fc_g2(h1))
        h2_2d = h2.view(h2.size(0), 32, 7, 7)
        h3 = F.softplus(self.deconv1(h2_2d))
        h4 = F.softplus(self.deconv2(h3))
        logits = self.deconv3(h4)
        return logits

    def _bernoulli_log_likelihood(self, x, logits):
        return -F.binary_cross_entropy_with_logits(logits, x, reduction='none').sum([1, 2, 3])


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import numpy as np


class HVAE(VAE):
    def __init__(self, args, avg_logit):
        super().__init__(args, avg_logit)
        self.K = args.K
        self.args = args  # Store args for later use
        self._init_hvae_params()

    def _init_hvae_params(self):
        init_lf = self.args.init_lf * np.ones(self.z_dim)
        init_lf_reparam = np.log(init_lf / (self.args.max_lf - init_lf))

        if self.args.vary_eps == 'true':
            init_lf_reparam = np.tile(init_lf_reparam, (self.K, 1))

        self.lf_reparam = nn.Parameter(torch.tensor(init_lf_reparam, dtype=torch.float32))
        self.temp_method = self.args.temp_method

        if self.temp_method == 'free':
            init_alphas = self.args.init_alpha * np.ones(self.K)
            init_alphas_reparam = np.log(init_alphas / (1 - init_alphas))
            self.alphas_reparam = nn.Parameter(torch.tensor(init_alphas_reparam, dtype=torch.float32))
        elif self.temp_method == 'fixed':
            init_T_0 = self.args.init_T_0
            init_T_0_reparam = np.log(init_T_0 - 1)
            self.T_0_reparam = nn.Parameter(torch.tensor(init_T_0_reparam, dtype=torch.float32))
        elif self.temp_method == 'none':
            self.register_buffer('T_0', torch.tensor(1., dtype=torch.float32))
            self.register_buffer('alphas', torch.ones(self.K, dtype=torch.float32))
        else:
            raise ValueError(f'Tempering method {self.temp_method} not supported')

    def get_elbo(self, x, args):
        q_mu, q_sigma = self._inf_network(x)

        z_0 = q_mu + q_sigma * torch.randn_like(q_mu)
        p_0 = torch.sqrt(self.T_0) * torch.randn_like(q_mu)

        z_K, p_K = self._his(z_0, p_0, x, args)

        p_x_given_zK_logits = self._gen_network(z_K)
        expected_log_likelihood = self._bernoulli_log_likelihood(x, p_x_given_zK_logits)

        log_prob_zK = -0.5 * (z_K ** 2).sum(1)
        log_prob_pK = -0.5 * (p_K ** 2).sum(1)
        sum_log_sigma = q_sigma.log().sum(1)

        neg_kl_term = log_prob_zK + log_prob_pK + sum_log_sigma + self.z_dim
        elbo = (expected_log_likelihood + neg_kl_term).mean()

        return elbo

    def get_nll(self, x, args):
        q_mu, q_sigma = self._inf_network(x)

        z_0 = q_mu + q_sigma * torch.randn_like(q_mu)
        p_0 = torch.sqrt(self.T_0) * torch.randn_like(q_mu)

        z_K, p_K = self._his(z_0, p_0, x, args)

        p_x_given_zK_logits = self._gen_network(z_K)
        expected_log_likelihood = self._bernoulli_log_likelihood(x, p_x_given_zK_logits)

        log_prob_zK = -0.5 * (z_K ** 2).sum(1)
        log_prob_pK = -0.5 * (p_K ** 2).sum(1)
        sum_log_sigma = q_sigma.log().sum(1)

        log_prob_z0 = -0.5 * (((z_0 - q_mu) / q_sigma) ** 2).sum(1)
        log_prob_p0 = -0.5 / self.T_0 * (p_0 ** 2).sum(1)

        nll_samples = (expected_log_likelihood + log_prob_zK + log_prob_pK
                       + sum_log_sigma - log_prob_z0 - log_prob_p0)

        nll_samples_reshaped = nll_samples.view(args.n_IS, args.n_batch_test)
        nll_lse = torch.logsumexp(nll_samples_reshaped, dim=0)
        nll = np.log(args.n_IS) - nll_lse.mean()

        return nll

    def _his(self, z_0, p_0, x, args):
        z = z_0
        p = p_0

        for k in range(1, self.K + 1):
            if args.vary_eps == 'true':
                lf_eps = self.lf_eps[k - 1, :]
            else:
                lf_eps = self.lf_eps

            p_half = p - 0.5 * lf_eps * self._dU_dz(z, x)
            z = z + lf_eps * p_half
            p_temp = p_half - 0.5 * lf_eps * self._dU_dz(z, x)

            p = self.alphas[k - 1] * p_temp

        return z, p

    def _dU_dz(self, z, x):
        z.requires_grad_(True)
        net_out = self._gen_network(z)
        U = F.softplus(net_out).sum((1, 2, 3)) - (x * net_out).sum((1, 2, 3))
        grad_U = torch.autograd.grad(U.sum(), z)[0] + z
        return grad_U

    @property
    def lf_eps(self):
        return torch.sigmoid(self.lf_reparam) * self.args.max_lf

    @property
    def alphas(self):
        if self.temp_method == 'free':
            return torch.sigmoid(self.alphas_reparam)
        elif self.temp_method == 'fixed':
            T_0 = 1 + torch.exp(self.T_0_reparam)
            k_vec = torch.arange(1, self.K + 1, dtype=torch.float32, device=self.lf_reparam.device)
            k_m_1_vec = torch.arange(0, self.K, dtype=torch.float32, device=self.lf_reparam.device)
            temp_sched = (1 - T_0) * k_vec ** 2 / self.K ** 2 + T_0
            temp_sched_m_1 = (1 - T_0) * k_m_1_vec ** 2 / self.K ** 2 + T_0
            return torch.sqrt(temp_sched / temp_sched_m_1)
        else:
            return self.alphas

    @property
    def T_0(self):
        if self.temp_method == 'free':
            return torch.prod(self.alphas) ** (-2)
        elif self.temp_method == 'fixed':
            return 1 + torch.exp(self.T_0_reparam)
        else:
            return self.T_0