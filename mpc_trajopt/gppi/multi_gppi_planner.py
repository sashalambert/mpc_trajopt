__author__ = "Alexander Lambert"
__license__ = "MIT"

import torch
# from .control_priors import diag_Cov, const_ctrl_Cov
from mpc.obstacle_map.map_generator import generate_obstacle_map
from mpc.gppi.mp_priors_multi import Multi_MP_Prior
from mpc.factors.gp_factor import GPFactor
from mpc.factors.unary_factor import UnaryFactor
import matplotlib.pyplot as plt
import time
import random


class MultiGPPI:

    def __init__(
            self,
            num_particles_per_goal,
            num_samples,
            traj_len,
            opt_iters,
            dt=None,
            n_dof=None,
            step_size=1.,
            temp=1.,
            start_state=None,
            multi_goal_states=None,
            sigma_start=None,
            sigma_start_init=None,
            sigma_start_sample=None,
            sigma_goal=None,
            sigma_goal_init=None,
            sigma_goal_sample=None,
            sigma_gp=None,
            sigma_gp_init=None,
            sigma_gp_sample=None,
            w_obst=None,
            seed=0,
            tensor_args=None,
    ):
        if tensor_args is None:
            tensor_args = {'device': torch.device('cuda'), 'dtype': torch.float32}
        self.tensor_args = tensor_args

        torch.manual_seed(seed)

        self.n_dof = n_dof
        self.d_state_opt = 2 * self.n_dof
        self.dt = dt

        self.traj_len = traj_len
        self.num_goals = multi_goal_states.shape[0]
        self.num_particles_per_goal = num_particles_per_goal
        self.num_particles = num_particles_per_goal * self.num_goals
        self.num_samples = num_samples
        self.opt_iters = opt_iters
        self.step_size = step_size
        self.temp = temp
        self.sigma_start = sigma_start
        self.sigma_start_init = sigma_start_init
        self.sigma_start_sample = sigma_start_sample
        self.sigma_goal = sigma_goal
        self.sigma_goal_init = sigma_goal_init
        self.sigma_goal_sample = sigma_goal_sample
        self.sigma_gp = sigma_gp
        self.sigma_gp_init = sigma_gp_init
        self.sigma_gp_sample = sigma_gp_sample
        self.w_obst = w_obst
        self.start_states = start_state # position + velocity

        assert multi_goal_states.dim() == 2
        assert multi_goal_states.shape[0] == self.num_goals
        self.multi_goal_states = multi_goal_states # position + velocity
        # self.goal_state = goal_states[0]

        self._mean = None
        self._weights = None
        self._sample_dist = None

        self.set_prior_factors()
        self.reset(start_state, multi_goal_states)

    def set_prior_factors(self):

        #========= Cost factors ===============
        self.start_prior = UnaryFactor(
            self.d_state_opt,
            self.sigma_start,
            self.start_states,
            self.tensor_args,
        )

        self.gp_prior = GPFactor(
            self.n_dof,
            self.sigma_gp,
            self.dt,
            self.traj_len - 1,
            self.tensor_args,
        )

        self.multi_goal_prior = []
        for i in range(self.num_goals):
            self.multi_goal_prior.append(
                UnaryFactor(
                    self.d_state_opt,
                    self.sigma_goal,   # Assume same goal Cov. for now
                    self.multi_goal_states[i],
                    self.tensor_args,
                )
            )

        #========= Initialization factors ===============
        self.start_prior_init = UnaryFactor(
            self.d_state_opt,
            self.sigma_start_init,
            self.start_states,
            self.tensor_args,
        )

        self.gp_prior_init = GPFactor(
            self.n_dof,
            self.sigma_gp_init,
            self.dt,
            self.traj_len - 1,
            self.tensor_args,
        )

        self.multi_goal_prior_init = []
        for i in range(self.num_goals):
            self.multi_goal_prior_init.append(
                UnaryFactor(
                    self.d_state_opt,
                    self.sigma_goal_init,    # Assume same goal Cov. for now
                    self.multi_goal_states[i],
                    self.tensor_args,
                )
            )

        #========= Sampling factors ===============
        self.start_prior_sample = UnaryFactor(
            self.d_state_opt,
            self.sigma_start_sample,
            self.start_states,
            self.tensor_args,
        )

        self.gp_prior_sample = GPFactor(
            self.n_dof,
            self.sigma_gp_sample,
            self.dt,
            self.traj_len - 1,
            self.tensor_args,
        )

        self.multi_goal_prior_sample = []
        for i in range(self.num_goals):
            self.multi_goal_prior_sample.append(
                UnaryFactor(
                    self.d_state_opt,
                    self.sigma_goal_sample,   # Assume same goal Cov. for now
                    self.multi_goal_states[i],
                    self.tensor_args,
                )
            )

    def get_prior_dist(
            self,
            gp_prior_init,
            state_init,
            goal_states=None,
    ):

        return Multi_MP_Prior(
            self.traj_len - 1,
            self.dt,
            2 * self.n_dof,
            self.n_dof,
            self.start_prior_sample.K,
            gp_prior_init.Q_inv[0],
            state_init,
            K_g_inv=self.multi_goal_prior_init[0].K,  # Assume same goal Cov. for now
            goal_states=goal_states,
            tensor_args=self.tensor_args,
        )

    def get_sampling_dist(
            self,
            particle_means,
            gp_prior_sample,
            state_init,
    ):

        return Multi_MP_Prior(
            self.traj_len - 1,
            self.dt,
            2 * self.n_dof,
            self.n_dof,
            self.start_prior_sample.K,
            gp_prior_sample.Q_inv[0],
            state_init,
            K_g_inv=self.multi_goal_prior_sample[0].K,  # Assume same goal Cov. for now
            means=particle_means,
            tensor_args=self.tensor_args,
        )

    def reset(
            self,
            start_state=None,
            multi_goal_states=None,
    ):

        if start_state is not None:
            self.start_state = start_state.clone()

        if multi_goal_states is not None:
            self.multi_goal_states = multi_goal_states.clone()

        # Initialization particles from prior distribution
        self._init_dist = self.get_prior_dist(
            self.gp_prior_init,
            self.start_state,
            self.multi_goal_states,
        )
        # self._init_dist = self.get_prior_dist(self.gp_prior_init, self.start_state,  torch.zeros(self.d_state_opt, **tensor_args))
        self.particle_means = self._init_dist.sample(self.num_particles_per_goal).to(**self.tensor_args)
        self.particle_means = self.particle_means.flatten(0, 1)

        # Sampling distributions
        self._sample_dist = self.get_sampling_dist(
            self.particle_means,
            self.gp_prior_sample,
            self.start_state,
        )
        self.Sigma_inv = self._sample_dist.Sigma_inv

        self.state_samples = self._sample_dist.sample(self.num_samples).to(**self.tensor_args)

    def _get_costs(self, observation):

        th = self.state_samples.reshape(-1, self.traj_len, self.d_state_opt)

        # Start prior
        err_p = self.start_prior.get_error(th[:, [0]], calc_jacobian=False)
        w_mat = self.start_prior.K
        start_costs = err_p @ w_mat.unsqueeze(0) @ err_p.transpose(1, 2)
        start_costs = start_costs.squeeze()

        # GP prior
        err_gp = self.gp_prior.get_error(th, calc_jacobian=False)
        w_mat = self.gp_prior.Q_inv[0] # repeated Q_inv
        w_mat = w_mat.reshape(1, 1, self.d_state_opt, self.d_state_opt)
        gp_costs = err_gp.transpose(2, 3) @ w_mat @ err_gp
        gp_costs = gp_costs.sum(1)
        gp_costs = gp_costs.squeeze()

        costs = start_costs + gp_costs

        # Obstacle cost
        if 'obst_map' in observation:
            obst_cost = observation['obst_map'].get_collisions(th[..., 0:2]) * self.w_obst
            obst_cost = obst_cost.sum(1)
            costs += obst_cost

        # Goal prior
        th = th.reshape(self.num_goals, self.num_particles_per_goal * self.num_samples, self.traj_len, self.d_state_opt)
        costs = costs.reshape(self.num_goals, self.num_particles_per_goal * self.num_samples)
        for i in range(self.num_goals):
            err_g = self.multi_goal_prior[i].get_error(th[i, :, [-1]], calc_jacobian=False)
            w_mat = self.multi_goal_prior[i].K
            goal_costs = err_g @ w_mat.unsqueeze(0) @ err_g.transpose(1, 2)
            goal_costs = goal_costs.squeeze()
            costs[i] += goal_costs

        costs = costs.reshape(self.num_particles, self.num_samples)

        # Add cost from importance-sampling ratio
        # V  = self.state_samples.view(-1, self.num_samples, self.traj_len * self.d_state_opt)  # flatten trajectories
        # U = self.particle_means.view(-1, 1, self.traj_len * self.d_state_opt)
        # costs += self.temp * (V @ self.Sigma_inv @ U.transpose(1, 2)).squeeze(2)

        return costs

    def sample_and_eval(self, observation):

        # TODO: update prior covariance with new goal location

        # Sample state-trajectory particles
        self.state_samples = self._sample_dist.sample(self.num_samples).to(
            **self.tensor_args)

        # Evaluate costs
        costs = self._get_costs(observation)

        position_seq = self.state_samples[..., :self.n_dof]
        velocity_seq = self.state_samples[..., -self.n_dof:]

        position_seq_mean = self.particle_means[..., :self.n_dof].clone()
        velocity_seq_mean = self.particle_means[..., -self.n_dof:].clone()

        return (
            velocity_seq,
            position_seq,
            velocity_seq_mean,
            position_seq_mean,
            costs,
        )

    def _update_distribution(self, costs, traj_samples):

        self._weights = torch.softmax( -costs / self.temp, dim=1)
        self._weights = self._weights.reshape(-1, self.num_samples, 1, 1)

        self.particle_means.add_(
            self.step_size * (
                self._weights * (traj_samples - self.particle_means.unsqueeze(1))
            ).sum(1)
        )
        self._sample_dist.set_mean(self.particle_means.view(self.num_particles, -1))

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
                 control_particles,
                 state_particles,
                 costs,) = self.sample_and_eval(observation)

                self._update_distribution(costs, self.state_samples)

        self._recent_control_samples = control_samples
        self._recent_control_particles = control_particles
        self._recent_state_trajectories = state_trajectories
        self._recent_state_particles = state_particles
        self._recent_weights = self._weights

        return (
            state_trajectories,
            control_samples,
            costs,
        )

    def _get_traj(self, mode='best'):
        if mode == 'best':
            #TODO: Fix for multi-particles
            particle_ind = self._weights.argmax()
            traj = self.state_samples[particle_ind].clone()
        elif mode == 'mean':
            traj = self._mean.clone()
        else:
            raise ValueError('Unidentified sampling mode in get_next_action')
        return traj

    def get_recent_samples(self):
        return (
            self._recent_control_samples.detach().clone(),
            self._recent_control_particles.detach().clone(),
            self._recent_state_trajectories.detach().clone(),
            self._recent_state_particles.detach().clone(),
            self._recent_weights.detach().clone(),
        )


if __name__ == "__main__":

    device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    tensor_args = {'device': device, 'dtype': torch.float32}

    start_q = torch.Tensor([-9, -9]).to(**tensor_args)
    start_state = torch.cat((start_q, torch.zeros(2, **tensor_args)))
    # goal_q = torch.Tensor([9, 9]).to(**tensor_args)
    # goal_state = torch.cat((goal_q, torch.zeros(2, **tensor_args)))
    # num_goals = 3
    # multi_goal_states = goal_state.repeat(num_goals, 1)

    multi_goal_states = torch.tensor([
        [9, 9, 0., 0.],
        [9, -3, 0., 0.],
        [-3, 9, 0., 0.],
    ]).to(**tensor_args)

    num_particles = 16
    num_particles_per_goal = 5

    seed = 11

    ## Planner - 2D point particle dynamics
    gppi_params = dict(
        num_particles_per_goal=num_particles_per_goal,
        num_samples=128,
        traj_len=64,
        dt=0.02,
        n_dof=2,
        opt_iters=1, # Keep this 1 for visualization
        temp=1.,
        start_state=start_state,
        multi_goal_states=multi_goal_states,
        step_size=0.5,
        sigma_start=0.001,
        sigma_goal=0.001,
        sigma_gp=0.1,
        sigma_start_init=0.001,
        sigma_goal_init=0.001,
        sigma_gp_init=50.,
        sigma_start_sample=0.001,
        sigma_goal_sample=0.1,
        sigma_gp_sample=5,
        w_obst=1.e9,
        seed=seed,
        tensor_args=tensor_args,
    )
    planner = MultiGPPI(**gppi_params)

    ## Obstacle map
    # obst_list = [(0, 0, 4, 6)]
    obst_list = []
    cell_size = 0.1
    map_dim = [20, 20]

    obst_params = dict(
        map_dim=map_dim,
        obst_list=obst_list,
        cell_size=cell_size,
        map_type='direct',
        random_gen=True,
        num_obst=10,
        rand_xy_limits=[[-7.5, 7.5], [-7.5, 7.5]],
        rand_shape=[2, 2],
        seed=seed,
        tensor_args=tensor_args,
    )
    # For obst. generation
    random.seed(seed)
    obst_map = generate_obstacle_map(**obst_params)[0]

    obs = {
        'state': start_state,
        'goal_states': multi_goal_states,
        'cost_func': None,
        'obst_map': obst_map
    }

    #---------------------------------------------------------------------------
    # Optimize
    # opt_iters = 500
    opt_iters = 1000

    traj_history = []
    for i in range(opt_iters + 1):
        print(i)
        time_start = time.time()
        planner.optimize(obs)
        # print(time.time() - time_start)
        controls, _, trajectories, _, weights = planner.get_recent_samples()
        traj_history.append(trajectories)
    #---------------------------------------------------------------------------
    # Plotting

    import numpy as np
    x = np.linspace(-10, 10, 200)
    y = np.linspace(-10, 10, 200)

    for iter, trajs in enumerate(traj_history):

        if iter % 250 == 0:
        # if iter % 25 == 0:
        # if iter % 10 == 0:
            fig = plt.figure()
            ax = fig.gca()
            cs = ax.contourf(x, y, obst_map.map, 20)
            cbar = fig.colorbar(cs, ax=ax)

            trajs = trajs.cpu().numpy()
            mean_trajs = trajs.mean(1)
            for i in range(trajs.shape[0]):
                for j in range(trajs.shape[1]):
                    ax.plot(trajs[i, j, :, 0], trajs[i, j, :, 1], 'r', alpha=0.15)
            for i in range(trajs.shape[0]):
                ax.plot(mean_trajs[i, :, 0], mean_trajs[i, :, 1], 'b')
            plt.show()
            plt.close('all')
