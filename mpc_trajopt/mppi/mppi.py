__author__ = "Alexander Lambert"
__license__ = "MIT"

import torch
from mpc.control_priors import get_multivar_gaussian_prior
# from .control_priors import diag_Cov, const_ctrl_Cov
from mpc.dynamics.point_particle import PointParticle
from mpc.obstacle_map.map_generator import generate_obstacle_map
import matplotlib.pyplot as plt
from matplotlib import cm


class MPPI:

    def __init__(
            self,
            system,
            num_ctrl_samples,
            rollout_steps,
            opt_iters,
            control_std=None,
            step_size=1.,
            temp=1.,
            Cov_prior_type='indep_ctrl',
            tensor_args=None,
            **kwargs,
    ):
        if tensor_args is None:
            tensor_args = {'device': torch.device('cuda'), 'dtype': torch.float32}
        self.tensor_args = tensor_args

        self.system = system
        self.state_dim = system.state_dim
        self.control_dim = system.control_dim
        self.rollout_steps = rollout_steps
        self.num_ctrl_samples = num_ctrl_samples
        self.opt_iters = opt_iters

        self.step_size = step_size
        self.temp = temp
        self._mean = torch.zeros(
            self.rollout_steps,
            self.control_dim,
            **self.tensor_args,
        )
        self.control_std = control_std
        self.Cov_prior_type = Cov_prior_type
        self.weights = None

        self.ctrl_dist = get_multivar_gaussian_prior(
                control_std,
                rollout_steps,
                self.control_dim,
                Cov_type=Cov_prior_type,
                mu_init=self._mean,
                tensor_args=self.tensor_args,
            )

        Cov_inv = []
        for i in range(self.control_dim):
            Cov_inv.append(self.ctrl_dist.Cov[..., i].inverse())
        self.Cov_inv = torch.stack(Cov_inv)

    def reset(self):
        self._mean = torch.zeros(
            self.rollout_steps,
            self.control_dim,
            **self.tensor_args,
        )
        self.update_ctrl_dist()

    def update_ctrl_dist(self):
        # Update mean
        self.ctrl_dist.update_means(self._mean)

    def update_controller(self, costs, U_sampled):
        weights = torch.softmax(
            -costs / self.temp,
            dim=0,
        )
        self.weights = weights.clone()

        weights = weights.reshape(-1, 1, 1)
        self._mean.add_(
            self.step_size * (
                weights * (U_sampled - self._mean.unsqueeze(0))
            ).sum(0)
        )

        self.update_ctrl_dist()

    def sample_and_eval(self, observation):
        state_trajectories = torch.empty(
            self.num_ctrl_samples,
            self.rollout_steps, # self.rollout_steps + 1,
            self.state_dim,
            **self.tensor_args,
        )

        # Sample control sequences
        control_samples = self.ctrl_dist.sample(
            self.num_ctrl_samples,
            tensor_args=self.tensor_args,
        )

        # Roll-out dynamics
        state_trajectories[:, 0] = observation['state']
        for i in range(self.rollout_steps - 1):
            state_trajectories[:, i+1] = self.system.dynamics(
                state_trajectories[:, i].unsqueeze(1),
                control_samples[:, i].unsqueeze(1),
            ).squeeze(1)

        # Evaluate costs
        costs = self.system.traj_cost(
            state_trajectories.transpose(0, 1).unsqueeze(2),
            control_samples.transpose(0, 1).unsqueeze(2),
            observation,
        )

        # Add cost from importance-sampling ratio
        for i in range(self.control_dim):
            V = control_samples[..., i]
            U = self._mean[..., i]
            costs += self.temp * (V @ self.Cov_inv[i] @ U).reshape(-1, 1)

        return (
            control_samples,
            state_trajectories,
            costs,
        )

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

                self.update_controller(costs, control_samples)
                self._mean = self._mean.detach()

        self._recent_control_samples = control_samples
        self._recent_state_trajectories = state_trajectories
        self._recent_weights = self.weights

        return (
            state_trajectories,
            control_samples,
            costs,
        )

    def pop(self):
        action = self._mean[0, :].clone().detach()
        self.shift()
        return action

    def shift(self):
        self._mean = self._mean.roll(shifts=-1, dims=-1)
        self._mean[-1:] = 0.

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

    ## Point Particle dynamics
    # mppi_params = dict(
    #     num_ctrl_samples=64,
    #     rollout_steps=64,
    #     # control_std=[1., 1.],
    #     control_std=[2.5, 2.5],
    #     temp=1.,
    #     opt_iters=1,
    #     step_size=1.,
    #     Cov_prior_type='indep_ctrl',
    #     tensor_args=tensor_args,
    # )

    mppi_params = dict(
        num_ctrl_samples=64,
        rollout_steps=64,
        control_std=[0.15, 0.15],
        temp=1.,
        opt_iters=1,
        step_size=1.,
        Cov_prior_type='const_ctrl',
        tensor_args=tensor_args,
    )

    system_params = dict(
        rollout_steps=mppi_params['rollout_steps'],
        ctrl_min=[-100, -100],
        ctrl_max=[100, 100],
        verbose=True,
        discount=1.,
        dt=0.01,
        c_weights={
            'pos': 0.,
            'vel': 0.,
            'ctrl': 0.,
            'e_cost': 1.,
            'obst': 1.e9,
            'pos_T': 1.e3,
            'vel_T': 0.,
        },
        tensor_args=tensor_args,
    )
    system = PointParticle(**system_params)
    controller = MPPI(system, **mppi_params)

    start_state = torch.Tensor([-8, -8]).to(**tensor_args)
    goal_state = torch.Tensor([8, 8]).to(**tensor_args)

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
    )

    obs = {
        'state': start_state,
        'goal_state': goal_state,
        'cost_func': None,
        'obst_map': obst_map
    }

    traj_history = []
    for i in range(250):
        print(i)
        controller.optimize(obs)
        controls, trajectories, weights = controller.get_recent_samples()
        traj_history.append(trajectories)

    ## Plotting
    import numpy as np
    x = np.linspace(-10, 10, 200)
    y = np.linspace(-10, 10, 200)
    grid = np.asarray(np.meshgrid(x, y)).transpose(1, 2, 0).reshape((-1, 2))

    for iter, trajs in enumerate(traj_history):

        if iter == 249:
        # if iter > 25:
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
