__author__ = "Alexander Lambert"
__license__ = "MIT"


import numpy as np
import torch


class PointParticle:
    def __init__(
            self,
            dt=0.01,
            dyn_std=np.zeros(4, ),
            c_weights=None,
            discount=1.0,
            control_type='velocity',
            start_state=None,
            goal_state=None,
            verbose=False,
            deterministic=True,
            device=None,
            ctrl_min: list = None,
            ctrl_max: list = None,
            control_dim=2,
            state_dim=2,
            rollout_steps=None,
            tensor_args=None,
    ):

        if tensor_args is None:
            tensor_args = {'device': torch.device('cuda'), 'dtype': torch.float32}
        self.tensor_args = tensor_args

        self.control_dim = control_dim
        if control_type == 'velocity':
            self.state_dim = state_dim
        elif control_type == 'acceleration':
            self.state_dim = state_dim * 2
        else:
            raise IOError('control_type "{}" not recognized'.format(control_type))
        if c_weights is None:
            self._c_weights = {
                'pos': 10.,
                'vel': 10.,
                'ctrl': 0.,
                'e_cost': 10.,
                'obst': 100.,
                'pos_T': 10.,
                'vel_T': 0.,
            }
        else:
            self._c_weights = c_weights

        assert len(ctrl_min) == self.control_dim
        assert len(ctrl_max) == self.control_dim
        self.ctrl_min = torch.tensor(ctrl_min).to(**tensor_args)
        self.ctrl_max = torch.tensor(ctrl_max).to(**tensor_args)

        self.discount_seq = self._get_discount_seq(discount, rollout_steps)

        if start_state is not None:
            self.start_state = torch.from_numpy(start_state).to(**self.tensor_args)
        else:
            self.start_state = torch.zeros(self.state_dim, **self.tensor_args)
        self.state = self.start_state.clone()
        if goal_state is not None:
            if isinstance(goal_state, np.ndarray):
                self.goal_state = torch.from_numpy(goal_state).to(**self.tensor_args)
            elif isinstance(goal_state, torch.Tensor):
                self.goal_state = goal_state
            else:
                raise IOError
        else:
            self.goal_state = torch.zeros(self.state_dim, **self.tensor_args)
        self.rollout_steps = rollout_steps
        self.device = device
        self.dt = dt
        self.dyn_std = torch.from_numpy(dyn_std).to(**self.tensor_args)
        self.control_type = control_type
        self.discount = discount
        self.verbose = verbose
        self.deterministic = deterministic

    @property
    def state(self):
        return self._state.clone().detach()

    @state.setter
    def state(self, value):
        self._state = value

    def reset(self):
        self.state = self.start_state.clone()
        cost, _ = self.traj_cost(
            self.state.reshape(1, 1, 1, -1),
            torch.zeros(1, 1, 1, self.control_dim),
        )
        return self.state, cost

    def step(self, action):
        state = self.state.reshape(1, 1, -1)
        action = action.reshape(1, 1, -1)
        self.state = self.dynamics(state, action)

        state = state.reshape(1, 1, 1, -1)
        action = action.reshape(1, 1, 1, -1)
        cost, _ = self.traj_cost(state, action)

        return self.state.squeeze(), cost.squeeze()

    def dynamics(
            self,
            x,
            u_s,
            use_crn=False,
    ):
        num_ctrl_samples, num_state_samples = x.size(0), x.size(1)
        ctrl_dim = u_s.size(-1)

        # clamp controls
        u = u_s.clamp(min=self.ctrl_min, max=self.ctrl_max)

        if self.deterministic:
            xdot = torch.cat(
                (x[...,2:], u),
                dim=2,
            )
        else:
            # Noise in control channel
            if use_crn:
                noise = self.dyn_std * torch.randn(
                    num_state_samples,
                    ctrl_dim,
                    **self.tensor_args,
                )
            else:
                noise = self.dyn_std * torch.randn(
                    num_ctrl_samples,
                    num_state_samples,
                    ctrl_dim,
                    **self.tensor_args,
                )

            u_s = u[...] + noise
            xdot = torch.cat(
                (x[..., 2:], u_s),
                dim=2,
            )
        x_next = x + xdot * self.dt
        return x_next

    def render(self, state=None, ):
        pass

    def _get_discount_seq(self, discount, rollout_steps):
        # Discount factors
        discount_seq = torch.cumprod(
            torch.ones(
                rollout_steps,
                device=self.tensor_args['device'],
                dtype=self.tensor_args['dtype'],
            ) * discount,
            dim=0,
        )
        discount_seq /= discount  # start with weight one
        return discount_seq

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

        Returns
        -------
        cost : Tensor
            Trajectory costs, of shape [num_ctrl_samples, num_state_samples]
        """

        rollout_steps, num_ctrl_samples, num_state_samples, state_dim = X_in.shape

        batch_size = num_ctrl_samples * num_state_samples
        # New shape: X: [batch, steps+1, particles, state_dim]
        X = X_in.view(
            -1,
            batch_size,
            state_dim,
        ).transpose(0, 1)

        U = U_in.view(
            -1,
            batch_size,
            self.control_dim,
        ).transpose(0, 1)

        if observation is not None:
            goal_state = observation['goal_state']
            cost_func = observation['cost_func']
            obst_map = observation['obst_map']
        else:
            goal_state = self.goal_state
            cost_func = None
            obst_map = None

        if cost_func is not None:
            energy_cost = cost_func(X.view(-1, 2))
            energy_cost = energy_cost.view(batch_size, rollout_steps).sum(1)
            energy_cost *= self._c_weights['e_cost']
        else:
            energy_cost = 0.

        if obst_map is not None:
            obst_cost = obst_map.get_collisions(X[..., 0:2])
            obst_cost = obst_cost * self._c_weights['obst'] * self.discount_seq.unsqueeze(0)
        else:
            obst_cost = 0.

        # Distance to goal
        dX = X - goal_state[..., :2]
        dX_final = dX[:, -1, :]

        # Discounted Costs
        pos_cost = (dX[..., 0:2]**2 * self._c_weights['pos']).sum(-1) * self.discount_seq.unsqueeze(0)
        vel_cost = (dX[..., 2:4]**2 * self._c_weights['vel']).sum(-1) * self.discount_seq.unsqueeze(0)
        control_cost = (U**2 * self._c_weights['ctrl']).sum(-1) * self.discount_seq.unsqueeze(0)
        terminal_cost = (dX_final**2 * self._c_weights['pos_T']).sum(-1) * self.discount_seq[-1].unsqueeze(0)

        # Sum along trajectory timesteps
        pos_cost = pos_cost.sum(1)
        vel_cost = vel_cost.sum(1)
        control_cost = control_cost.sum(1)
        if obst_map is not None:
            obst_cost = obst_cost.sum(1)

        if self.verbose:
            print('pos_cost: {:5.4f}'.format(pos_cost.mean().detach().cpu().numpy()))
            print('vel_cost: {:5.4f}'.format(vel_cost.mean().detach().cpu().numpy()))
            print('control_cost: {:5.4f}'.format(control_cost.mean().detach().cpu().numpy()))
            print('terminal_cost: {:5.4f}'.format(terminal_cost.mean().detach().cpu().numpy()))
            if cost_func is not None:
                print('energy_cost: {:5.4f}'.format(energy_cost.mean().detach().cpu().numpy()))
            if obst_map is not None:
                print('obst_cost: {:5.4f}'.format(obst_cost.mean().detach().cpu().numpy()))
            print('')

        costs = (pos_cost + vel_cost + control_cost + terminal_cost + obst_cost + energy_cost ).unsqueeze(1)

        return costs.view(num_ctrl_samples, num_state_samples)
