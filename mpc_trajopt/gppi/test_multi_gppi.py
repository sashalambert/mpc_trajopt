import torch
from multi_gppi_planner import MultiGPPI
from cost_functions import (
    CostComposite,
    CostObst2D,
    CostFactoredMultiGoal,
)
from mpc.obstacle_map.map_generator import generate_obstacle_map
import matplotlib.pyplot as plt
import time
import random

if __name__ == "__main__":

    device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    tensor_args = {'device': device, 'dtype': torch.float32}

    # Start / Goal positions and velocities
    start_q = torch.Tensor([-9, -9]).to(**tensor_args)
    start_state = torch.cat((start_q, torch.zeros(2, **tensor_args)))
    multi_goal_states = torch.tensor([
        [9, 9, 0., 0.],
        [9, -3, 0., 0.],
        [-3, 9, 0., 0.],
    ]).to(**tensor_args)
    num_goals = 3
    num_particles = 16
    num_particles_per_goal = 5
    traj_len = 64
    seed = 11
    n_dof = 2
    dim = n_dof * 2
    dt = 0.02

    #-------------------------------- Cost func. ---------------------------------

    # Factored Cost params
    cost_sigmas = dict(
        sigma_start=0.001,
        sigma_goal=0.001,
        sigma_gp=0.1,
    )
    obst_cost_weight = 1.e9

    # Construct cost function
    cost_func_list = []

    cost_prior_multigoal = CostFactoredMultiGoal(
        n_dof, start_state, multi_goal_states, traj_len, dt,
        sigma_params=cost_sigmas, tensor_args=tensor_args,
    )
    cost_func_list += [cost_prior_multigoal.eval]

    cost_obst_2D = CostObst2D(traj_len, weight_obst=obst_cost_weight, tensor_args=tensor_args)
    cost_func_list += [cost_obst_2D.eval]

    cost_composite = CostComposite(cost_func_list, traj_len, dim, tensor_args=tensor_args)
    cost_func = cost_composite.eval

    #--------------------------- Planner init. ----------------------------------

    # Intializing particle trajectories
    init_sigmas = dict(
        sigma_start_init=0.001,
        sigma_goal_init=0.001,
        sigma_gp_init=50.,
    )

    # Sampling from particle trajectory dists
    sampling_sigmas = dict(
        sigma_start_sample=0.001,
        sigma_goal_sample=0.1,
        sigma_gp_sample=5,
    )

    ## Planner - 2D point particle dynamics
    gppi_params = dict(
        num_particles_per_goal=5,
        num_samples=128,
        traj_len=traj_len,
        dt=dt,
        n_dof=n_dof,
        num_goals=num_goals,
        opt_iters=1, # Keep this 1 for visualization
        temp=1.,
        step_size=0.5,
        seed=0,
        init_sigmas=init_sigmas,
        sampling_sigmas=sampling_sigmas,
        cost_func=cost_func,
        tensor_args=tensor_args,
    )
    planner = MultiGPPI(**gppi_params)
    planner.reset(start_state, multi_goal_states)

    #------------------------------ Obstacles ---------------------------------

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

    #---------------------------------- Run ------------------------------------

    # opt_iters = 500
    opt_iters = 1000

    traj_history = []
    best_traj_history = []
    for i in range(opt_iters + 1):
        print(i)
        time_start = time.time()
        planner.optimize(obs)
        # print(time.time() - time_start)

        (state_trajs_mean,
         control_state_trajs_mean,
         state_trajs,
         control_trajs,
         weights,) = planner.get_recent_samples()

        best_state_trajs, best_control_trajs = planner.get_best_trajs()

        traj_history.append(state_trajs)
        best_traj_history.append(best_state_trajs)

    #------------------------------- Visualize ------------------------------------

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

            best_trajs = best_traj_history[iter].cpu().numpy()
            for i in range(best_trajs.shape[0]):
                ax.plot(best_trajs[i, :, 0], best_trajs[i, :, 1], 'g')

            plt.show()
            plt.close('all')
