__author__ = "Alexander Lambert"
__license__ = "MIT"

import torch
from mpc.gppi.mp_priors_multi import Multi_MP_Prior
from mpc.factors.gp_factor import GPFactor
from mpc.factors.unary_factor import UnaryFactor

import copy


class MultiGPPI:

    def __init__(
            self,
            num_particles_per_goal,
            num_samples,
            traj_len,
            opt_iters,
            dt=None,
            n_dof=None,
            num_goals=None,
            step_size=1.,
            temp=1.,
            init_sigmas=None,
            sampling_sigmas=None,
            cost_func=None,
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

        self.cost_func = cost_func
        self.traj_len = traj_len
        self.num_goals = num_goals
        self.num_particles_per_goal = num_particles_per_goal
        self.num_particles = num_particles_per_goal * self.num_goals
        self.num_samples = num_samples
        self.opt_iters = opt_iters
        self.step_size = step_size
        self.temp = temp
        self.init_sigmas = init_sigmas
        self.sampling_sigmas = sampling_sigmas
        self.start_state = None # position + velocity
        self.multi_goal_states = None # position + velocity

        self._mean = None
        self._weights = None
        self._sample_dist = None

    def set_prior_factors(self):

        #========= Initialization factors ===============
        self.start_prior_init = UnaryFactor(
            self.d_state_opt,
            self.init_sigmas['sigma_start_init'],
            self.start_state,
            self.tensor_args,
        )

        self.gp_prior_init = GPFactor(
            self.n_dof,
            self.init_sigmas['sigma_gp_init'],
            self.dt,
            self.traj_len - 1,
            self.tensor_args,
        )

        self.multi_goal_prior_init = []
        for i in range(self.num_goals):
            self.multi_goal_prior_init.append(
                UnaryFactor(
                    self.d_state_opt,
                    self.init_sigmas['sigma_goal_init'],    # Assume same goal Cov. for now
                    self.multi_goal_states[i],
                    self.tensor_args,
                )
            )

        #========= Sampling factors ===============
        self.start_prior_sample = UnaryFactor(
            self.d_state_opt,
            self.sampling_sigmas['sigma_start_sample'],
            self.start_state,
            self.tensor_args,
        )

        self.gp_prior_sample = GPFactor(
            self.n_dof,
            self.sampling_sigmas['sigma_gp_sample'],
            self.dt,
            self.traj_len - 1,
            self.tensor_args,
        )

        self.multi_goal_prior_sample = []
        for i in range(self.num_goals):
            self.multi_goal_prior_sample.append(
                UnaryFactor(
                    self.d_state_opt,
                    self.sampling_sigmas['sigma_goal_sample'],   # Assume same goal Cov. for now
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
            cost_func=None,
    ):

        # Assume single start state provided
        assert start_state.dim() == 1
        assert start_state.shape[0] == self.n_dof * 2 # Pos + Vel

        assert multi_goal_states.dim() == 2
        assert multi_goal_states.shape[0] == self.num_goals
        assert multi_goal_states.shape[1] == self.n_dof * 2 # Pos + Vel

        self.start_state = start_state.clone()
        self.multi_goal_states = multi_goal_states.clone()

        if cost_func is not None:
            self.cost_func = cost_func

        # Set cost factors
        self.set_prior_factors()

        # Initialization particles from prior distribution
        self._init_dist = self.get_prior_dist(
            self.gp_prior_init,
            self.start_state,
            self.multi_goal_states,
        )

        self.particle_means = self._init_dist.sample(self.num_particles_per_goal).to(**self.tensor_args)
        self.particle_means = self.particle_means.flatten(0, 1)

        # Sampling distributions
        self._sample_dist = self.get_sampling_dist(
            self.particle_means,
            self.gp_prior_sample,
            self.start_state,
        )
        self.Sigma_inv = self._sample_dist.Sigma_inv

        self.traj_samples = self._sample_dist.sample(self.num_samples).to(**self.tensor_args)

    def _get_costs(self, th, observation):

        # Get factored prior costs
        costs = self.cost_func(th, observation)

        # Add cost from importance-sampling ratio
        # V  = self.traj_samples.view(-1, self.num_samples, self.traj_len * self.d_state_opt)  # flatten trajectories
        # U = self.particle_means.view(-1, 1, self.traj_len * self.d_state_opt)
        # costs += self.temp * (V @ self.Sigma_inv @ U.transpose(1, 2)).squeeze(2)

        return costs

    def sample_and_eval(self, observation):

        # TODO: update prior covariance with new goal location

        # Sample state-trajectory particles
        self.traj_samples = self._sample_dist.sample(self.num_samples).to(
            **self.tensor_args)

        # Evaluate costs
        costs = self._get_costs(self.traj_samples, observation)

        position_seq = self.traj_samples[..., :self.n_dof]
        velocity_seq = self.traj_samples[..., -self.n_dof:]

        position_seq_mean = self.particle_means[..., :self.n_dof].clone()
        velocity_seq_mean = self.particle_means[..., -self.n_dof:].clone()

        return (
            position_seq_mean,
            velocity_seq_mean,
            position_seq,
            velocity_seq,
            costs,
        )

    def _update_distribution(self, costs, traj_samples):

        costs = costs.reshape(self.num_particles, self.num_samples)

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
                (state_particles,
                 control_particles,
                 state_samples,
                 control_samples,
                 costs,) = self.sample_and_eval(observation)

                self._update_distribution(costs, self.traj_samples)

        # Save solutions. Index per-goal.
        self._recent_state_particles = state_particles
        self._recent_control_particles = control_particles
        self._recent_state_samples = state_samples
        self._recent_control_samples = control_samples
        self._recent_weights = self._weights
        self._recent_observation = copy.deepcopy(observation)

        return (
            state_samples,
            control_samples,
            costs,
        )

    def get_best_trajs(self):
        state_trajs = self._recent_state_particles
        control_trajs = self._recent_control_particles
        full_trajs = torch.cat((state_trajs, control_trajs,), dim=-1)

        obs = self._recent_observation
        costs = self._get_costs(full_trajs, obs)

        state_trajs = state_trajs.view(self.num_goals, self.num_particles_per_goal, self.traj_len, self.n_dof)
        control_trajs = control_trajs.view(self.num_goals, self.num_particles_per_goal, self.traj_len, self.n_dof)
        costs = costs.view(self.num_goals, self.num_particles_per_goal)

        best_idxs = costs.argmin(dim=1)
        best_idxs = best_idxs.view(-1, 1, 1, 1).repeat(1, 1, state_trajs.shape[2], state_trajs.shape[3])

        best_state_trajs = state_trajs.gather(1, best_idxs).flatten(0, 1)
        best_control_trajs = control_trajs.gather(1, best_idxs).flatten(0, 1)

        return best_state_trajs, best_control_trajs

    def get_recent_samples(self):
        return (
            self._recent_state_particles.detach().clone(),
            self._recent_control_particles.detach().clone(),
            self._recent_state_samples.detach().clone(),
            self._recent_control_samples.detach().clone(),
            self._recent_weights.detach().clone(),
        )
