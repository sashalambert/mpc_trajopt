__author__ = "Alexander Lambert"
__license__ = "MIT"

import torch
# from .control_priors import diag_Cov, const_ctrl_Cov
from mpc.obstacle_map.map_generator import generate_obstacle_map
from mpc.mp_priors import MP_Prior
from mpc.factors.gp_factor import GPFactor
from mpc.factors.unary_factor import UnaryFactor
import matplotlib.pyplot as plt
import time


class GPPI:

    def __init__(
            self,
            num_samples,
            traj_len,
            opt_iters,
            dt=None,
            n_dof=None,
            step_size=1.,
            temp=1.,
            start_state=None,
            goal_state=None,
            sigma_start=None,
            sigma_goal=None,
            sigma_gp=None,
            w_gp=None,
            w_obst=None,
            tensor_args=None,
    ):
        if tensor_args is None:
            tensor_args = {'device': torch.device('cuda'), 'dtype': torch.float32}
        self.tensor_args = tensor_args

        self.n_dof = n_dof
        self.d_state_opt = 2 * self.n_dof
        self.dt = dt

        self.traj_len = traj_len
        self.num_samples = num_samples
        self.opt_iters = opt_iters
        self.step_size = step_size
        self.temp = temp
        self.sigma_start = sigma_start
        self.sigma_goal = sigma_goal
        self.sigma_gp = sigma_gp
        self.w_gp = w_gp
        self.w_obst = w_obst
        self.start_state = start_state # position + velocity
        self.goal_state = goal_state # position + velocity

        self._mean = None
        self._weights = None
        self._sample_dist = None

        self.reset(start_state, goal_state)

    def get_prior_dist(
            self,
            state_init,
            goal_state=None,
    ):

        self.start_prior = UnaryFactor(
            self.d_state_opt,
            self.sigma_start,
            self.start_state,
            self.tensor_args,
        )

        self.goal_prior = UnaryFactor(
            self.d_state_opt,
            self.sigma_goal,
            self.goal_state,
            self.tensor_args,
        )

        self.gp_prior = GPFactor(
            self.n_dof,
            self.sigma_gp,
            self.dt,
            self.traj_len - 1,
            self.tensor_args,
        )

        return MP_Prior(
            self.traj_len - 1,
            self.dt,
            2 * self.n_dof,
            self.n_dof,
            self.start_prior.K,
            self.gp_prior.Q_inv[0],
            state_init,
            self.tensor_args,
            K_g_inv=self.goal_prior.K,
            goal_state=goal_state,
            prior_type='const_velocity',
        )

    def reset(
            self,
            start_state=None,
            goal_state=None,
    ):

        if start_state is not None:
            self.start_state = start_state.clone()

        if goal_state is not None:
            self.goal_state = goal_state.clone()

        # Will set mean to either the start state, or a const-vel traj to goal.
        self._sample_dist = self.get_prior_dist(self.start_state, self.goal_state)
        self._mean = self._sample_dist.get_mean().squeeze(0)
        self.Sigma_inv = self._sample_dist.Sigma_inv

        self.state_particles = self._sample_dist.sample(self.num_samples).to(**self.tensor_args)

    def _get_costs(self, observation):

        th = self.state_particles

        # Start prior
        err_p = self.start_prior.get_error(th[:, [0]], calc_jacobian=False)
        w_mat = self.start_prior.K
        start_costs = err_p @ w_mat.unsqueeze(0) @ err_p.transpose(1, 2)
        start_costs = start_costs.squeeze()

        # Goal prior
        err_g = self.goal_prior.get_error(th[:, [-1]], calc_jacobian=False)
        w_mat = self.goal_prior.K
        goal_costs = err_g @ w_mat.unsqueeze(0) @ err_g.transpose(1, 2)
        goal_costs = goal_costs.squeeze()

        # GP prior
        err_gp = self.gp_prior.get_error(th, calc_jacobian=False)
        w_mat = self.gp_prior.Q_inv[0] # repeated Q_inv
        w_mat = w_mat.reshape(1, 1, self.d_state_opt, self.d_state_opt)
        gp_costs = err_gp.transpose(2, 3) @ w_mat @ err_gp
        gp_costs = gp_costs.sum(1)
        gp_costs = gp_costs.squeeze()
        gp_costs *= self.w_gp

        costs = start_costs + goal_costs + gp_costs

        # Obstacle cost
        if 'obst_map' in observation:
            obst_cost = observation['obst_map'].get_collisions(th[..., 0:2]) * self.w_obst
            obst_cost = obst_cost.sum(1)
            costs += obst_cost

        return costs.reshape(self.num_samples,)

    def sample_and_eval(self, observation):

        # TODO: update prior covariance with new goal location

        # Sample state-trajectory particles
        self.state_particles = self._sample_dist.sample(self.num_samples).to(
            **self.tensor_args)

        # Evaluate costs
        costs = self._get_costs(observation)

        # Add cost from importance-sampling ratio
        V = self.state_particles.view(self.num_samples, -1)
        U = self._mean.view(-1, 1)
        costs += self.temp * (V @ self.Sigma_inv @ U).squeeze()

        position_seq = self.state_particles[..., :self.n_dof]
        velocity_seq = self.state_particles[..., -self.n_dof:]

        return (
            velocity_seq,
            position_seq,
            costs,
        )

    def _update_distribution(self, costs, traj_particles):

        self._weights = torch.softmax( -costs / self.temp, dim=0)
        self._weights = self._weights.reshape(-1, 1, 1)

        self._mean.add_(
            self.step_size * (
                self._weights * (traj_particles - self._mean.unsqueeze(0))
            ).sum(0)
        )
        self._sample_dist.set_mean(self._mean.flatten())

    def optimize(
            self,
            observation={'state': None},
            opt_iters=None,
    ):

        if opt_iters is None:
            opt_iters = self.opt_iters

        for opt_step in range(opt_iters):

            with torch.no_grad():
                (control_samples,
                 state_trajectories,
                 costs,) = self.sample_and_eval(observation)

                self._update_distribution(costs, self.state_particles)

        self._recent_control_samples = control_samples
        self._recent_state_trajectories = state_trajectories
        self._recent_weights = self._weights

        return (
            state_trajectories,
            control_samples,
            costs,
        )

    def _get_traj(self, mode='best'):
        if mode == 'best':
            particle_ind = self._weights.argmax()
            traj = self.state_particles[particle_ind].clone()
        elif mode == 'mean':
            traj = self._mean.clone()
        else:
            raise ValueError('Unidentified sampling mode in get_next_action')
        return traj

    def get_recent_samples(self):
        return (
            self._recent_control_samples.detach().clone(),
            self._recent_state_trajectories.detach().clone(),
            self._recent_weights.detach().clone(),
        )


if __name__ == "__main__":

    device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    tensor_args = {'device': device, 'dtype': torch.float32}

    start_q = torch.Tensor([-8, -8]).to(**tensor_args)
    goal_q = torch.Tensor([8, 8]).to(**tensor_args)
    start_state = torch.cat((start_q, torch.zeros(2, **tensor_args)))
    goal_state = torch.cat((goal_q, torch.zeros(2, **tensor_args)))

    ## Planner - 2D point particle dynamics
    gppi_params = dict(
        num_samples=64,
        traj_len=64,
        dt=0.01,
        n_dof=2,
        opt_iters=1,
        temp=1.,
        start_state=start_state,
        goal_state=goal_state,
        step_size=1.,
        sigma_start=0.001,
        sigma_goal=0.001,
        sigma_gp=5.,
        w_gp=100.,
        w_obst=1.e9,
        tensor_args=tensor_args,
    )
    planner = GPPI(**gppi_params)

    ## Obstacle map
    obst_list = [(0, 0, 4, 6)]
    # obst_list = []
    cell_size = 0.1
    map_dim = [20, 20]

    obst_map = generate_obstacle_map(
        map_dim, obst_list, cell_size,
        map_type='direct',
        random_gen=True,
        num_obst=10,
        rand_xy_limits=[[-8, 8], [-8, 8]],
        rand_shape=[2, 2],
        tensor_args=tensor_args,
    )[0]

    obs = {
        'state': start_state,
        'goal_state': goal_state,
        'cost_func': None,
        'obst_map': obst_map
    }

    opt_iters = 500
    # opt_iters = 100

    traj_history = []
    for i in range(opt_iters):
        print(i)
        time_start = time.time()
        planner.optimize(obs)
        print(time.time() - time_start)
        controls, trajectories, weights = planner.get_recent_samples()
        traj_history.append(trajectories)

    ## Plotting
    import numpy as np
    x = np.linspace(-10, 10, 200)
    y = np.linspace(-10, 10, 200)
    grid = np.asarray(np.meshgrid(x, y)).transpose(1, 2, 0).reshape((-1, 2))

    for iter, trajs in enumerate(traj_history):

        if iter == opt_iters - 1:
        # if iter < 10:
            fig = plt.figure()
            ax = fig.gca()
            cs = ax.contourf(x, y, obst_map.map, 20)
            cbar = fig.colorbar(cs, ax=ax)

            trajs = trajs.cpu().numpy()
            for i in range(trajs.shape[0]):
                ax.plot(trajs[i, :, 0], trajs[i, :, 1], 'r', alpha=0.5)
            mean_traj = trajs.mean(0)
            ax.plot(mean_traj[:, 0], mean_traj[:, 1], 'b')
            plt.show()
            plt.close('all')
