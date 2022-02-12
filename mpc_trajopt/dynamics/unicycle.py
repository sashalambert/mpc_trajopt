__author__ = "Alexander Lambert"
__license__ = "MIT"


import numpy as np
import torch
from .point_particle import PointParticle


class UnicycleSystem(PointParticle):
    def __init__(
            self,
            dt=0.2,
            c_weights=None,
            state_dim=3,
            control_dim=2,
            **kwargs,
    ):

        if c_weights is None:
            self._c_weights = {
                'pos': 10.,
                'theta': 10.,
                'obst': 10.,
                'vel': 10.,
                'theta_dot': 10.,
                'pos_T': 10.,
                'theta_T': 10.,
                'vel_T': 0.,
                'theta_dot_T': 0.,
            }
        else:
            self._c_weights = c_weights

        super().__init__(
            state_dim=state_dim,
            control_dim=control_dim,
            dt=dt,
            c_weights=c_weights,
            **kwargs,
        )
        self.mean = []
        self.std = []

    def dynamics(
            self,
            state,
            u_s,
            use_crn=False,
    ):
        """
        Simple Unicyclce model.
         State : [x, y, theta]
         Controls: [vel, w]
         Dyn:
            dx = vel * cos(theta)
            dy = vel * sin(theta)
            dtheta = w
        """
        num_ctrl_samples, num_state_samples = state.size(0), state.size(1)
        ctrl_dim = u_s.size(-1)

        # clamp controls
        u = u_s.clone()
        # u[..., 0] = u_s[..., 0].clamp(min=-self.max_control[0], max=self.max_control[0])
        # u[..., 1] = u_s[..., 1].clamp(min=-self.max_control[1], max=self.max_control[1])

        # Noise in control channel
        if not self.deterministic:
            if use_crn:
                noise = self.dyn_std * torch.randn(
                    num_state_samples,
                    ctrl_dim,
                )
            else:
                noise = self.dyn_std * torch.randn(
                    num_ctrl_samples,
                    num_state_samples,
                    ctrl_dim,
                )
            u = u[...] + noise

        theta = state[..., 2].unsqueeze(-1)
        vel = u[..., 0].unsqueeze(-1)
        w = u[..., 1].unsqueeze(-1)
        dstate = torch.cat((
            vel * torch.cos(theta),
            vel * torch.sin(theta),
            w,
        ), dim=-1
        )

        state_next = state + dstate * self.dt
        return state_next

    def traj_cost(
            self,
            X_in, U_in,
            observation=None,
    ):
        """
        Implements quadratic trajectory cost.
        Args
        ----
        X_in : Tensor
            State trajectories, of shape
                [steps+1, num_ctrl_samples, num_state_samples, state dim]
        U_in : Tensor
            Control trajectories, of shape
                [steps+1, num_ctrl_samples, num_state_samples, control dim]
        observation : dict (optional)
            Contains goal_state and map
        Returns
        -------
        cost : Tensor
            Trajectory costs, of shape [num_ctrl_samples, num_state_samples]
        """


        (rollout_steps,
         num_ctrl_samples,
         num_state_samples,
         state_dim) = X_in.shape

        batch_size = num_ctrl_samples * num_state_samples

        # New shape: X: [batch, steps+1, particles, state_dim]
        X = X_in.view(
            -1,
            num_ctrl_samples * num_state_samples,
            state_dim,
        ).transpose(0, 1)

        U = U_in.view(
            -1,
            num_ctrl_samples * num_state_samples,
            self.control_dim,
        ).transpose(0, 1)

        if observation is not None:
            goal_state = observation['goal_state']
            cost_func = observation['cost_func']
        else:
            goal_state = None
            cost_func = None

        # Discount factors
        discount_seq = torch.cumprod(torch.ones(rollout_steps, **self.tensor_args) * self.discount, dim=0)
        discount_seq /= self.discount  # start with weight one

        # Collision costs
        if cost_func is not None:
            pos = X[:, :, :2]
            energy_cost = cost_func(pos.view(-1, 2))
            energy_cost = energy_cost.view(batch_size, rollout_steps).sum(1)
            energy_cost *= self._c_weights['e_cost']
        else:
            energy_cost = 0.

        # Distance to goal
        if goal_state is not None:
            dX = goal_state - X
        else:
            dX = torch.zeros_like(X)

        dX_final = dX[:, -1, :]

        # Discounted Costs
        pos_cost = (dX[..., 0:2]**2 * self._c_weights['pos']).sum(-1) * discount_seq.unsqueeze(0)
        theta_cost = (dX[..., 2]**2 * self._c_weights['theta']) * discount_seq.unsqueeze(0)
        vel_cost = (U[..., 0]**2 * self._c_weights['vel']) * discount_seq.unsqueeze(0)
        theta_dot_cost = (U[..., 1]**2 * self._c_weights['theta_dot']) * discount_seq.unsqueeze(0)
        terminal_pos_cost = (dX_final[..., 0:2]**2 * self._c_weights['pos_T']).sum(-1) * discount_seq[-1].unsqueeze(0)
        terminal_vel_cost = (U[..., 0]**2 * self._c_weights['vel_T']).sum(-1) * discount_seq[-1].unsqueeze(0)
        terminal_theta_dot_cost = (U[..., 1]**2 * self._c_weights['theta_dot_T']).sum(-1) * discount_seq[-1].unsqueeze(0)

        # Sum along trajectory timesteps
        pos_cost = pos_cost.sum(1)
        theta_cost = theta_cost.sum(1)
        vel_cost = vel_cost.sum(1)
        theta_dot_cost = theta_dot_cost.sum(1)

        if self.verbose:
            print('pos_cost: {:5.4f}'.format(pos_cost.mean().detach().cpu().numpy()))
            print('theta_cost: {:5.4f}'.format(theta_cost.mean().detach().cpu().numpy()))
            print('terminal_cost: {:5.4f}'.format(terminal_pos_cost.mean().detach().cpu().numpy()))
            if map is not None:
                print('energy_cost: {:5.4f}'.format(energy_cost.mean().detach().cpu().numpy()))
            print('')

        costs = (pos_cost + theta_cost + vel_cost + theta_dot_cost + energy_cost +
                 terminal_pos_cost + terminal_vel_cost + terminal_theta_dot_cost).unsqueeze(1)

        return costs.view(num_ctrl_samples, num_state_samples)