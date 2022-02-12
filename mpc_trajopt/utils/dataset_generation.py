import torch
from torch.utils.data import TensorDataset
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

import ebm_il
import yaml
from mpc.obstacle_map.map_generator import generate_obstacle_map
from mpc.gppi.multi_gppi_planner import MultiGPPI

root_path = Path(ebm_il.__path__[0]).resolve().parent
exp_name = 'test_multi_obst2'
data_path = root_path / 'data' / 'planar_nav' / exp_name
data_path.mkdir(parents=True, exist_ok=True)

def generate_dataset(
    N=100,  # num envs instance
    start_q=[-9, -9],
    goal_q=[9, 9],
    cell_size=0.1,
    map_dim=[20, 20],
    opt_iters=500,
    seed=1,
    gppi_params=None,
    obst_map_params=None,
    show=False
):
    device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')
    tensor_args = {'device': device, 'dtype': torch.float32}
    start_q = torch.Tensor(start_q).to(**tensor_args)
    goal_q = torch.Tensor(goal_q).to(**tensor_args)
    start_state = torch.cat((start_q, torch.zeros(2, **tensor_args)))
    goal_state = torch.cat((goal_q, torch.zeros(2, **tensor_args)))
    if gppi_params is None:  # use default setting
        gppi_params = dict(
            num_particles=5,
            num_samples=128,
            traj_len=64,
            dt=0.02,
            n_dof=2,
            opt_iters=1,
            temp=1.,
            start_state=start_state,
            goal_state=goal_state,
            step_size=0.5,
            sigma_start=0.001,
            sigma_goal=0.01,
            sigma_gp=5.,
            sigma_goal_init=0.01,
            sigma_gp_init=50.,
            w_gp=1.e6,
            w_obst=1.e9,
            seed=seed,
            tensor_args=tensor_args,
        )
    if obst_map_params is None:
        obst_map_params = dict(
            map_type='direct',
            random_gen=True,
            num_obst=10,
            rand_xy_limits=[[-7.5, 7.5], [-7.5, 7.5]],
            rand_shape=[2, 2],
            seed=seed,
            tensor_args=tensor_args,
        )
    planner = MultiGPPI(**gppi_params)
    trajs = []
    maps = torch.zeros(0, int(map_dim[0]/cell_size), int(map_dim[1]/cell_size)).to(**tensor_args)
    # starts, goals = [], []  # may store random start/goal later if needed
    # for plotting
    x = np.linspace(-10, 10, 200)
    y = np.linspace(-10, 10, 200)
    for _ in tqdm(range(N)):
        planner.reset(start_state, goal_state)
        obst_map = generate_obstacle_map(map_dim, [], cell_size, **obst_map_params)
        obs = {
            'state': start_state,
            'goal_state': goal_state,
            'cost_func': None,
            'obst_map': obst_map
        }
        for i in range(opt_iters):
            planner.optimize(obs)
        controls, trajectories, weights = planner.get_recent_samples()
        mean_trajs = trajectories.mean(1)
        collisions = 1 - obst_map.get_collisions(mean_trajs)
        free_trajs = mean_trajs[collisions.all(dim=1), :, :].cpu().numpy()
        trajs.append(free_trajs)
        maps = torch.cat((maps, obst_map.convert_map().unsqueeze(0)), dim=0)
        if show:
            fig = plt.figure()
            ax = fig.gca()
            cs = ax.contourf(x, y, obst_map.map, 20)
            fig.colorbar(cs, ax=ax)
            for i in range(free_trajs.shape[0]):
                ax.plot(free_trajs[i, :, 0], free_trajs[i, :, 1], 'r', alpha=0.15)
            plt.show()
            plt.close('all')
    trajs = torch.Tensor(trajs)  # this may output a warning of different size array
    dataset = TensorDataset(trajs, maps)
    # Write params to config
    gppi_params.pop('start_state')
    gppi_params.pop('goal_state')
    gppi_params.pop('tensor_args')
    obst_map_params.pop('tensor_args')
    exp_params = {'gppi_params': gppi_params, 'obst_params': obst_map_params}
    with open(str(data_path / 'config.yaml'), 'w') as outfile:
        yaml.dump(exp_params, outfile, sort_keys=False)
    # dave dataset
    npart = gppi_params['num_particles']
    nobst = obst_map_params['num_obst']
    filename = 'npart_{}_nobst_{}'.format(npart, nobst)
    torch.save(dataset, data_path / filename + '.pt')


if __name__ == "__main__":
    generate_dataset(N=2, opt_iters=1000, show=True)
