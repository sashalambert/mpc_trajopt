__author__ = "Alexander Lambert"
__license__ = "MIT"

import torch
import torch.distributions as dist
import scipy
from abc import ABC, abstractmethod
import numpy as np
from scipy.stats import multivariate_normal


def straight_line_trajb(
        batch_size,
        start_confb,
        goal_confb,
        traj_time,
        num_steps,
        dof,
):
    # batch_size = start_confb.shape[0]
    th_initb = torch.zeros((batch_size, int(num_steps+1), 2*dof))
    avg_velb = (goal_confb - start_confb) / traj_time*1.0
    for i in range( int(num_steps) +1):
        th_initb[:, i, 0:dof] = start_confb[:, 0:dof] * (num_steps - i) * 1.0 / num_steps * 1.0 \
                              + goal_confb[:, 0:dof] * i * 1.0/num_steps * 1.0
    th_initb[:, :, dof:] = avg_velb.unsqueeze(1)
    return th_initb


class MP_Prior:

    def __init__(
            self,
            num_steps,
            dt,
            state_dim,
            dof,
            K_s_inv,
            K_gp_inv,
            start_state,
            tensor_args,
            K_g_inv=None,
            goal_state=None,
            mean=None,
            prior_type='const_velocity',
            use_numpy=False,
    ):
        """
        Motion-Planning prior.

        reference: "Continuous-time Gaussian process motion planning via
        probabilistic inference", Mukadam et al. (IJRR 2018)

        Parameters
        ----------
        num_steps : int
            Planning horizon length (not including start state).
        dt :  float
            Time-step size.
        state_dim : int
            State state_dimension.
        K_s_inv : Tensor
            Start-state inverse covariance. Shape: [state_dim, state_dim]
        K_gp_inv :  Tensor
            Gaussian-process single-step inverse covariance i.e. 'Q_inv'.
            Assumed constant, meaning homoscedastic noise with constant step-size.
            Shape: [2 * state_dim, 2 * state_dim]
        start_state : Tensor
            Shape: [state_dim]
        (Optional) K_g_inv : Tensor
            Goal-state inverse covariance. Shape: [state_dim, state_dim]
        (Optional) goal_state :  Tensor
            Shape: [state_dim]
        (Optional) dof : int
            Degrees of freedom.
        """

        self.state_dim = state_dim
        self.dof = dof
        self.num_steps = num_steps
        self.M = state_dim * (num_steps + 1)
        # self.num_steps = num_steps - 1
        # self.M = state_dim * num_steps
        self.tensor_args = tensor_args
        self.use_numpy = use_numpy

        self.goal_directed = (goal_state is not None)
        if self.goal_directed:
            assert K_g_inv is not None
            # self.M += state_dim

        # Mean can be provided
        if mean is not None:
            Sigma_inv = self.get_const_vel_covariance(
                dt,
                K_s_inv,
                K_gp_inv,
                K_g_inv,
            )
        elif prior_type == 'const_velocity':

            mean = self.get_const_vel_mean(
                start_state,
                goal_state,
                dt,
                num_steps,
                dof,
            )
            Sigma_inv = self.get_const_vel_covariance(
                dt,
                K_s_inv,
                K_gp_inv,
                K_g_inv,
            )
        else:
            raise NotImplementedError

        self.mean = mean.flatten()
        # self.Sigma_inv = Sigma_inv
        self.Sigma_inv = Sigma_inv + torch.eye(Sigma_inv.shape[0], **tensor_args) * 1.e-3
        self.update_dist(self.mean, self.Sigma_inv)

    def update_dist(
            self,
            mean,
            Sigma_inv,
    ):
        # Create Multi-variate Normal Distribution
        if self.use_numpy:
            mean_np = mean.cpu().numpy()
            Sigma_inv_np = Sigma_inv.cpu().numpy()
            Sigma_np = np.linalg.inv(Sigma_inv_np)
            self.dist = multivariate_normal(mean_np, Sigma_np)
        else:
            self.dist = dist.MultivariateNormal(
                mean,
                precision_matrix=Sigma_inv,
                # covariance_matrix=torch.inverse(Sigma_inv),
                # scale_tril=torch.cholesky(torch.inverse(Sigma_inv))
            )

    def get_mean(self, reshape=True):
        if reshape:
            return self.mean.clone().detach().reshape(
                1, self.num_steps + 1, self.state_dim,
            )
        else:
            self.mean.clone().detach()

    def set_mean(self, mean_new):
        assert mean_new.shape == self.mean.shape
        self.mean = mean_new.clone().detach()
        self.update_dist(self.mean, self.Sigma_inv)

    def set_Sigma_inv(self, Sigma_inv_new):
        assert Sigma_inv_new.shape == self.Sigma_inv.shape
        self.Sigma_inv = Sigma_inv_new.clone().detach()
        self.update_dist(self.mean, self.Sigma_inv)

    def const_vel_trajectory(
        self,
        start_state,
        goal_state,
        dt,
        num_steps,
        dof,
    ):
        state_traj = torch.zeros(num_steps + 1, 2 * dof, **self.tensor_args)
        mean_vel = (goal_state[:dof] - start_state[:dof]) / (num_steps * dt)
        for i in range(num_steps + 1):
            state_traj[i, :dof] = start_state[:dof] * (num_steps - i) * 1. / num_steps \
                                  + goal_state[:dof] * i * 1./num_steps
        state_traj[:, dof:] = mean_vel.unsqueeze(0)
        return state_traj

    def get_const_vel_mean(
        self,
        start_state,
        goal_state,
        dt,
        num_steps,
        dof,
    ):

        # Make mean goal-directed if goal_state is provided.
        if self.goal_directed:
            return self.const_vel_trajectory(
                start_state,
                goal_state,
                dt,
                num_steps,
                dof
            )
        else:
            return start_state.repeat(num_steps + 1, 1)

    def get_const_vel_covariance(
        self,
        dt,
        K_s_inv,
        K_gp_inv,
        K_g_inv,
        precision_matrix=True,
    ):
        # Transition matrix
        Phi = torch.eye(self.state_dim, **self.tensor_args)
        Phi[:self.dof, self.dof:] = torch.eye(self.dof, **self.tensor_args) * dt
        diag_Phis = Phi
        for _ in range(self.num_steps - 1):
            diag_Phis = torch.block_diag(diag_Phis, Phi)

        A = torch.eye(self.M, **self.tensor_args)
        A[self.state_dim:, :-self.state_dim] += -1. * diag_Phis
        if self.goal_directed:
            b = torch.zeros(self.state_dim, self.M,  **self.tensor_args)
            b[:, -self.state_dim:] = torch.eye(self.state_dim,  **self.tensor_args)
            A = torch.cat((A, b))

        Q_inv = K_s_inv
        for _ in range(self.num_steps):
            Q_inv = torch.block_diag(Q_inv, K_gp_inv).to(**self.tensor_args)
        if self.goal_directed:
            Q_inv = torch.block_diag(Q_inv, K_g_inv).to(**self.tensor_args)

        K_inv = A.t() @ Q_inv @ A
        if precision_matrix:
            return K_inv
        else:
            return torch.inverse(K_inv)

    def sample(self, num_samples):
        if self.use_numpy:
            samples = self.dist.rvs(num_samples).view(
                num_samples, self.num_steps + 1, self.state_dim,
            )
            samples = torch.as_tensor(samples, **self.tensor_args)
            return samples
        else:
            return self.dist.sample((num_samples,)).view(
                num_samples, self.num_steps + 1, self.state_dim,
            )

    def log_prob(self, x):
        return self.dist.log_prob(x)

    def save_dist(self, save_path='/tmp/dist.pt'):
        params = {
            'mean': self.mean,
            'Sigma_inv': self.Sigma_inv,
        }
        torch.save(params, save_path)
