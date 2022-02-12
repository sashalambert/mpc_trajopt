__author__ = "Alexander Lambert"
__license__ = "MIT"

import torch
# from .control_priors import diag_Cov, const_ctrl_Cov
from mpc.obstacle_map.map_generator import generate_obstacle_map
from mpc.obstacle_map.sdf_map import SDFMap
from mpc.gppi.mp_priors_multi import Multi_MP_Prior
from mpc.factors.gp_factor import GPFactor
from mpc.factors.unary_factor import UnaryFactor
from mpc.factors.obstacle_factor import ObstacleFactor
import matplotlib.pyplot as plt
import time
import random


class MultiGPMP:

    def __init__(
            self,
            num_particles_per_goal,
            traj_len,
            opt_iters,
            dt=None,
            n_dof=None,
            step_size=1.,
            temp=1.,
            start_state=None,
            multi_goal_states=None,
            sigma_obst=None,
            sigma_start=None,
            sigma_start_init=None,
            sigma_start_sample=None,
            sigma_gp=None,
            sigma_gp_init=None,
            sigma_gp_sample=None,
            sigma_goal=None,
            sigma_goal_init=None,
            sigma_goal_sample=None,
            seed=0,
            solver_params=None,
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
        self.opt_iters = opt_iters
        self.step_size = step_size
        self.temp = temp
        self.sigma_obst = sigma_obst
        self.sigma_start = sigma_start
        self.sigma_start_init = sigma_start_init
        self.sigma_start_sample = sigma_start_sample
        self.sigma_goal = sigma_goal
        self.sigma_goal_init = sigma_goal_init
        self.sigma_goal_sample = sigma_goal_sample
        self.sigma_gp = sigma_gp
        self.sigma_gp_init = sigma_gp_init
        self.sigma_gp_sample = sigma_gp_sample
        self.start_states = start_state # position + velocity
        self.solver_params = solver_params
        assert multi_goal_states.dim() == 2
        assert multi_goal_states.shape[0] == self.num_goals
        self.multi_goal_states = multi_goal_states # position + velocity
        # self.goal_state = goal_states[0]

        self._mean = None
        self._weights = None
        self._dist = None

        self.N = self.d_state_opt * self.traj_len # flattened particle dimension
        self.M = None
        self.set_prior_factors()
        self.reset(start_state, multi_goal_states)

    def set_prior_factors(self):
        self.M = 0 # First dimension of the graph jacobian

        #========= Main factors ===============
        self.start_prior = UnaryFactor(
            self.d_state_opt,
            self.sigma_start,
            self.start_states,
            self.tensor_args,
        )
        self.M += self.d_state_opt

        self.gp_prior = GPFactor(
            self.n_dof,
            self.sigma_gp,
            self.dt,
            self.traj_len - 1,
            self.tensor_args,
        )
        self.M += self.d_state_opt * (self.traj_len - 1)

        self.multi_goal_prior = []
        for i in range(self.num_goals):
            self.multi_goal_prior.append(
                UnaryFactor(
                    self.d_state_opt,
                    self.sigma_goal,
                    self.multi_goal_states[i],
                    self.tensor_args,
                )
            )
        self.M += self.d_state_opt

        self.obs_factor = ObstacleFactor(
            self.n_dof,
            self.sigma_obst,
            self.traj_len-1,
        )
        self.M += self.traj_len - 1

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
                    self.sigma_goal_init,
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
                    self.sigma_goal_sample,
                    self.multi_goal_states[i],
                    self.tensor_args,
                )
            )

    def get_dist(
            self,
            start_prior,
            gp_prior,
            goal_prior,
            state_init,
            particle_means=None,
            goal_states=None,
    ):

        return Multi_MP_Prior(
            self.traj_len - 1,
            self.dt,
            2 * self.n_dof,
            self.n_dof,
            start_prior.K,
            gp_prior.Q_inv[0],
            state_init,
            K_g_inv=goal_prior.K,  # Assume same goal Cov. for now
            means=particle_means,
            goal_states=goal_states,
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

        self.set_prior_factors()

        # Initialization particles from prior distribution
        self._init_dist = self.get_dist(
            self.start_prior_init,
            self.gp_prior_init,
            self.multi_goal_prior_init[0],
            self.start_state,
            goal_states=self.multi_goal_states,
        )
        # self._init_dist = self.get_prior_dist(self.gp_prior_init, self.start_state,  torch.zeros(self.d_state_opt, **tensor_args))
        self.particle_means = self._init_dist.sample(self.num_particles_per_goal).to(**self.tensor_args)
        self.particle_means = self.particle_means.flatten(0, 1)

        # Main distributions
        self._dist = self.get_dist(
            self.start_prior,
            self.gp_prior,
            self.multi_goal_prior[0],
            self.start_state,
            particle_means=self.particle_means,
        )
        self.Sigma_inv = self._dist.Sigma_inv

        self._sample_dist = self.get_dist(
            self.start_prior_sample,
            self.gp_prior_sample,
            self.multi_goal_prior_sample[0],
            self.start_state,
            particle_means=self.particle_means,
        )

    def optimize(
            self,
            observation={'state': None},
            opt_iters=None,
    ):

        if opt_iters is None:
            opt_iters = self.opt_iters

        for opt_step in range(opt_iters):
            b, K = self._step(observation)

        self.costs = self._get_costs(b, K)

        position_seq_mean = self.particle_means[..., :self.n_dof].clone()
        velocity_seq_mean = self.particle_means[..., -self.n_dof:].clone()
        costs = self.costs.clone()

        self._recent_control_particles = velocity_seq_mean
        self._recent_state_trajectories = position_seq_mean
        # self._recent_weights = self._weights

        return (
            velocity_seq_mean,
            position_seq_mean,
            costs,
        )

    def _step(self, observation):

        A, b, K = self.get_linear_system(self.particle_means, observation)

        J_t_J, g = self._get_grad_terms(
            A, b, K,
            delta=self.solver_params['delta'],
            trust_region=self.solver_params['trust_region'],
        )
        d_theta = self.get_torch_solve(
            J_t_J, g,
            method=self.solver_params['method'],
        )

        d_theta = d_theta.view(
                self.num_particles,
                self.traj_len,
                self.d_state_opt,
            )

        # self.state_particles.grad = -1. * d_theta
        self.particle_means = self.particle_means + self.step_size * d_theta

        # self.state_particles[..., :self.n_dofs] = self.state_particles[..., :self.n_dofs].clamp(
        #     min=self.joint_limits[:, 0] + self.jt_lim_offset,
        #     max=self.joint_limits[:, 1] - self.jt_lim_offset,
        # )

        return b, K

    def get_linear_system(self, x, observation):
        b = x.shape[0]
        temp_A = torch.zeros(b, self.M, self.N, **self.tensor_args)
        temp_b = torch.zeros(b, self.M, 1, **self.tensor_args)
        temp_K = torch.zeros(b, self.M, self.M, **self.tensor_args)

        # Start prior factor
        err_p, H_p = self.start_prior.get_error(x[:, [0]])
        temp_A[:, :self.d_state_opt, :self.d_state_opt] = H_p
        temp_b[:, :self.d_state_opt] = err_p
        temp_K[:, :self.d_state_opt, :self.d_state_opt] = self.start_prior.K

        # GP factors
        err_gp, H1_gp, H2_gp = self.gp_prior.get_error(x)
        for i in range(self.traj_len - 1):
            temp_A[:, (i+1)*self.d_state_opt:(i+2)*self.d_state_opt, i*self.d_state_opt:(i+1)*self.d_state_opt] = H1_gp[[i]]
            temp_A[:, (i+1)*self.d_state_opt:(i+2)*self.d_state_opt, (i+1)*self.d_state_opt:(i+2)*self.d_state_opt] = H2_gp[[i]]
            temp_b[:, (i+1)*self.d_state_opt:(i+2)*self.d_state_opt] = err_gp[:, i]
            temp_K[:, (i+1)*self.d_state_opt:(i+2)*self.d_state_opt, (i+1)*self.d_state_opt:(i+2)*self.d_state_opt] = self.gp_prior.Q_inv[[i]]
        offset = self.d_state_opt * self.traj_len

        # Goal prior
        x = x.reshape(self.num_goals, self.num_particles_per_goal, self.traj_len, self.d_state_opt)
        npg = self.num_particles_per_goal
        for i in range(self.num_goals):
            err_g, H_g = self.multi_goal_prior[i].get_error(x[i, :, [-1]])
            temp_A[i*npg: (i+1)*npg, offset:offset + self.d_state_opt, -self.d_state_opt: ] = H_g
            temp_b[i*npg: (i+1)*npg, offset:offset + self.d_state_opt] = err_g
            temp_K[i*npg: (i+1)*npg, offset:offset + self.d_state_opt, offset:offset + self.d_state_opt] = self.multi_goal_prior[i].K
        x = x.reshape(-1, self.traj_len, self.d_state_opt)
        offset = offset + self.d_state_opt

        if 'obst_map' in observation:
            x.requires_grad = True
            # no factor on first (current) time
            err_obst, H_obst = self.obs_factor.get_error(
                x[:, 1:, :self.n_dof],
                observation['obst_map'],
                observation['obst_locs'],
                observation['standardizer'],
            )
            for i in range(self.traj_len - 1):
                temp_A[:, offset+i, (i+1)*self.d_state_opt:(i+1)*self.d_state_opt + self.n_dof] = H_obst[:, i]
                temp_b[:, offset+i, :] = err_obst[:, [i]]
                temp_K[:, offset+i, offset+i] = self.obs_factor.K
            x.requires_grad = False
            offset = offset + self.traj_len - 1

        return temp_A, temp_b, temp_K

    def _get_grad_terms(
            self,
            A, b, K,
            delta=0.,
            trust_region=False,
    ):
        I = torch.eye(self.N, self.N, **self.tensor_args)
        A_t_K = A.transpose(1, 2) @ K
        A_t_A = A_t_K @ A
        if not trust_region:
            J_t_J = A_t_A + delta*I
        else:
            J_t_J = A_t_A + delta * I * torch.diagonal(A_t_A, dim1=1, dim2=2).unsqueeze(-1)
            # Since hessian will be averaged over particles, add diagonal matrix of the mean.
            # diag_A_t_A = A_t_A.mean(0) * I
            # J_t_J = A_t_A + delta * diag_A_t_A
        g = A_t_K @ b
        return J_t_J, g

    def get_torch_solve(
        self,
        A, b,
        method,
    ):
        if method == 'inverse':
            return torch.linalg.solve(A, b)
        elif method == 'cholesky':
            l = torch.linalg.cholesky(A)
            z = torch.triangular_solve(b, l, transpose=False, upper=False)[0]
            return torch.triangular_solve(z, l, transpose=True, upper=False)[0]
        else:
            raise NotImplementedError

    def _get_costs(self, errors, w_mat):
        costs = errors.transpose(1, 2) @ w_mat.unsqueeze(0) @ errors
        return costs.reshape(self.num_particles,)

    def sample_trajectories(self, num_samples_per_particle):
        self._sample_dist.set_mean(self.particle_means.view(self.num_particles, -1))
        self.state_samples = self._sample_dist.sample(num_samples_per_particle).to(
            **self.tensor_args)
        position_seq = self.state_samples[..., :self.n_dof]
        velocity_seq = self.state_samples[..., -self.n_dof:]
        return (
            position_seq,
            velocity_seq,
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

    num_particles_per_goal = 5

    seed = 11

    ## Planner - 2D point particle dynamics
    gpmp_params = dict(
        num_particles_per_goal=num_particles_per_goal,
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
        sigma_obst=0.003,
        sigma_start_init=0.001,
        sigma_goal_init=0.001,
        sigma_gp_init=50.,
        sigma_start_sample=0.001,
        sigma_goal_sample=0.001,
        sigma_gp_sample=2.5,
        seed=seed,
        solver_params={
            'delta': 0.,
            'trust_region': True,
            'method': 'cholesky',
        },
        tensor_args=tensor_args,
    )
    planner = MultiGPMP(**gpmp_params)

    ## Obstacle map
    # obst_list = [(0, 0, 4, 6)]
    obst_list = []
    cell_size = 0.1
    map_dim = [20, 20]

    # obst_params = dict(
    #     map_dim=map_dim,
    #     obst_list=obst_list,
    #     cell_size=cell_size,
    #     map_type='direct',
    #     random_gen=True,
    #     num_obst=10,
    #     rand_xy_limits=[[-7.5, 7.5], [-7.5, 7.5]],
    #     rand_shape=[2, 2],
    #     seed=seed,
    #     tensor_args=tensor_args,
    # )
    #
    # # For obst. generation
    # random.seed(seed)
    # obst_map = generate_obstacle_map(**obst_params)[0]

    obst_params = dict(
        centers_list=[
            [5, 0.],
            [0., 5],
            [-5, 0.],
        ],
        radii_list=[
            [2.5],
            [2.5],
            [2.5],
        ],
        buff_list=[
            [0.],
            [0.],
            [0.],
        ],
        # obst_preset=None,
        obst_preset='5x5',
        obst_spacing=5,
        obst_size=1.5,
        obst_buff=0.1,
    )
    obst_map = SDFMap(
        obst_params,
        tensor_args=tensor_args,
    )
    obs = {
        'state': start_state,
        'goal_states': multi_goal_states,
        'cost_func': None,
        'obst_map': obst_map
    }

    #---------------------------------------------------------------------------
    # Optimize
    opt_iters = 50
    # opt_iters = 500
    # opt_iters = 1000

    traj_history = []
    trajectories = planner.particle_means[..., :2].clone()
    traj_history.append(trajectories)
    for i in range(opt_iters + 1):
        print(i)
        time_start = time.time()
        controls, trajectories, costs = planner.optimize(obs)
        # print(time.time() - time_start)
        traj_history.append(trajectories)

    traj_samples, _ = planner.sample_trajectories(16)
    traj_samples = traj_samples.cpu().numpy()
    #---------------------------------------------------------------------------
    # Plotting

    import numpy as np
    res = 200
    x = np.linspace(-10, 10, res)
    y = np.linspace(-10, 10, res)
    X, Y = np.meshgrid(x,y)
    grid = torch.from_numpy(np.stack((X, Y), axis=-1))
    Z = obst_map.get_sdf(grid.reshape(-1, 2).to(**tensor_args)).reshape(res, res).detach().cpu().numpy()

    for iter, trajs in enumerate(traj_history):

        if iter % 25 == 0:
        # if iter % 100 == 0:
            fig = plt.figure()
            ax = fig.gca()
            cs = ax.contourf(X, Y, Z, 20)
            cbar = fig.colorbar(cs, ax=ax)

            trajs = trajs.cpu().numpy()
            for i in range(trajs.shape[0]):
                ax.plot(trajs[i, :, 0], trajs[i, :, 1], 'b')
            plt.show()
            plt.close('all')

    trajs = traj_history[-1]
    fig = plt.figure()
    ax = fig.gca()
    cs = ax.contourf(X, Y, Z, 20)
    cbar = fig.colorbar(cs, ax=ax)
    trajs = trajs.cpu().numpy()
    for i in range(traj_samples.shape[0]):
        for j in range(traj_samples.shape[1]):
            ax.plot(traj_samples[i, j, :, 0], traj_samples[i, j, :, 1], 'r', alpha=0.15)
    for i in range(trajs.shape[0]):
        ax.plot(trajs[i, :, 0], trajs[i, :, 1], 'b')
    plt.show()
    plt.close('all')

