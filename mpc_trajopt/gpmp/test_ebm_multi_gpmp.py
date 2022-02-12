import time

import matplotlib.pyplot as plt
import torch

from ebm_il.ebm_models.conditional_ebms import cond_MLP
from ebm_il.utils.generic import load_model
from mpc.obstacle_map.sdf_map import SDFMap
from multi_gpmp import MultiGPMP
from ebm_il.utils.generic import Standardizer


def load_ebm(ebm_params, device):
    model_class = ebm_params.pop('model_type')
    filepath = ebm_params.pop('filepath')
    ebm = model_class(**ebm_params)
    load_model(filepath, ebm)
    ebm = ebm.to(device)
    return ebm

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

    num_particles_per_goal = 7

    seed = 0

    #---------------------------------------------------------------------------

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

    USE_EBM_OBST = True

    #---------------------------------------------------------------------------
    # Learned EBM Obstacle map

    if USE_EBM_OBST:

        # obst_ebm_weights = './params_three_obst_ebm.pt'
        #
        # obst_ebm_params = dict(
        #     model_type=cond_MLP,
        #     filepath=obst_ebm_weights,
        #     input_dim=2+3*2,
        #     hidden_dim=512,
        #     hidden_depth=3,
        #     dropout=False,
        #     batch_norm=False,
        #     use_obst_vec=True,
        # )
        #
        # obst_map = load_ebm(obst_ebm_params, device)
        # # Dataset obst_locs range: [-7.5, 7.5]
        #
        # obst_locs = torch.tensor([
        #     [0., 0],
        #     [-6., -2],
        #     [5., 5.],
        # ], **tensor_args)
        #
        # # Random obstacle locs
        # # Note: training data does not include overlapping obstacles
        # torch.manual_seed(1)
        # obst_locs = torch.rand(3, 2).to(**tensor_args) * 15 - 7.5

        obst_ebm_weights = './params_five_obst_ebm.pt'

        obst_ebm_params = dict(
            model_type=cond_MLP,
            filepath=obst_ebm_weights,
            input_dim=2+5*2,
            hidden_dim=512,
            hidden_depth=3,
            dropout=False,
            batch_norm=False,
            use_obst_vec=True,
        )

        obst_map = load_ebm(obst_ebm_params, device)
        # Dataset obst_locs range: [-7.5, 7.5]
        obst_locs = torch.tensor([
            [0., 0],
            [-5., -5],
            [-5., 5.],
            [5., -5.],
            [5., 5.],
        ], **tensor_args)

        #Random obstacle locs
        #Note: training data does not include overlapping obstacles
        # torch.manual_seed(15)
        # obst_locs = torch.rand(5, 2).to(**tensor_args) * 15 - 7.5

        # Dataset dem. range: [-10, 10]
        standardizer = Standardizer(dim=2, min_x=-10., max_x=10)

    #---------------------------------------------------------------------------
    else:
        # SDF Obstacle map
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

        # Context vector only needed for EBM model
        obst_locs = None
        standardizer = None

    #---------------------------------------------------------------------------
    # Planning env params

    obs = {
        'state': start_state,
        'goal_states': multi_goal_states,
        'obst_map': obst_map,
        'obst_locs': obst_locs,
        'standardizer': standardizer,
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
    grid = torch.from_numpy(np.stack((X, Y), axis=-1)).to(**tensor_args)

    if isinstance(obst_map, cond_MLP):
        with torch.no_grad():
            grid = grid.reshape(-1, 2)
            obst_locs = obst_locs.repeat(grid.size(0), 1, 1)

            grid_in = standardizer.normalize(grid)
            obst_locs_in = standardizer.normalize(obst_locs)
            e_grid = obst_map(grid_in, None, obst_locs_in, None)
        Z = e_grid.cpu().numpy().reshape((res, res))

    elif isinstance(obst_map, SDFMap):
        Z = obst_map.get_sdf(grid.reshape(-1, 2).to(**tensor_args)).reshape(res, res).detach().cpu().numpy()
    else:
        raise TypeError

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