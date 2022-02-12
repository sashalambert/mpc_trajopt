import os
from pathlib import Path
import argparse
import torch

# Data
from ebm_il.data_models import LASA
# Model
from ebm_il.ebm_models import NaiveEnergyModel
from mpc.mppi.mppi import MPPI
from mpc.dynamics.point_particle import PointParticle
from mpc.dynamics.unicycle import UnicycleSystem
from ebm_il.samplers.mpc_sampler import MPCSampler
from mpc.vis_mppi import plot_mppi
from ebm_il.utils import to_numpy
import matplotlib.pyplot as plt
from ebm_il.utils.generic import load_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="Sshape")
    # parser.add_argument('--dataset', default="CShape")
    parser.add_argument('--save_path', default="test_output/lasa")
    parser.add_argument('--z_dim', type=int, default=2)
    parser.add_argument('--dim', type=int, default=512)
    parser.add_argument('--energy_model_iters', type=int, default=1)
    parser.add_argument('--generator_iters', type=int, default=10)
    parser.add_argument('--mcmc_iters', type=int, default=0)
    parser.add_argument('--lamda1', type=float, default=.1)
    ## Lambda 2 avoids the overfitting to the data and to be a little bit smoother ##
    parser.add_argument('--lamda2', type=float, default=0)
    parser.add_argument('--alpha', type=float, default=.01)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_samples', type=int, default=256)
    parser.add_argument('--iters', type=int, default=20000)
    parser.add_argument('--n_points', type=int, default=1600)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--save_interval', type=int, default=100)
    parser.add_argument('--vis_freq', type=int, default=100)
    parser.add_argument('--dyn_model', type=str, default='point_particle', choices=['point_particle', 'unicycle'])
    args = parser.parse_args()
    return args

args = parse_args()
args.save_path = os.path.join(args.save_path, args.dataset)

# time_now = time.strftime("%Y_%m_%d_%H_%M_%S")
# root = Path(os.path.join(args.save_path, time_now))
root = Path(os.path.join(args.save_path))

## Create Saving Directories ##
if root.exists():
    os.system('rm -rf %s' % str(root))

os.makedirs(str(root))
os.system('mkdir -p %s' % str(root / 'models'))
os.system('mkdir -p %s' % str(root / 'images'))

## GPU/ CPU ##
device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')
#device = 'cpu'

tensor_args = {'device': device, 'dtype': torch.float32}

if __name__ == '__main__':
    ## Dataset ##
    dim = 2
    data = LASA(filename=args.dataset, device=device)
    dataset = data.dataset

    ## EBM model ##
    #ebm = SimpleEBM(dim=2, hidden_dim=256).to(device)
    ebm = NaiveEnergyModel(args.dim).to(device)
    if args.dataset == 'Sshape':
        filenum = 5200
        filepath = '../examples/ebm_training_examples/lasa/logs/Sshape/2021_10_15_17_58_43/models/chkpt_{:05d}.pt'.format(filenum)
    elif args.dataset == 'CShape':
        filenum = 15300
        filepath = '../examples/ebm_training_examples/logs/lasa/CShape/2021_10_19_18_09_16/models/chkpt_{:05d}.pt'.format(filenum)
    else:
        raise IOError
    _ = load_model(filepath, ebm)

    # start_state = data.init_position.mean(axis=0)
    # goal_state = data.trajs_normal[:, -1, :].mean(axis=0)
    goal_state = data.init_position.mean(axis=0)
    start_state = data.trajs_normal[:, -1, :].mean(axis=0)
    start_state = torch.from_numpy(start_state).to(**tensor_args)
    goal_state = torch.from_numpy(goal_state).to(**tensor_args)

    if args.dyn_model == 'point_particle':
        ## Point Particle dynamics##
        mppi_params = dict(
            num_ctrl_samples=32,
            # rollout_steps=data.trj_length,
            rollout_steps=32,
            control_std=[1., 1.],
            temp=1.,
            opt_iters=1,
            step_size=1.,
            Cov_prior_type='indep_ctrl',
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
                'pos_T': 1.e3,
                'vel_T': 0.,
            },
            tensor_args=tensor_args,
        )
        system = PointParticle(**system_params)

    elif args.dyn_model == 'unicycle':
        # Unicycle dynamics: append heading value.
        start_state = torch.cat((start_state, torch.Tensor([1.52]).to(**tensor_args)))
        goal_state = torch.cat((goal_state, torch.Tensor([0]).to(**tensor_args)))

        mppi_params = dict(
            num_ctrl_samples=32,
            # rollout_steps=data.trj_length,
            rollout_steps=32,
            control_std=[0.5, 4.],
            temp=1.,
            opt_iters=1,
            step_size=1.,
            Cov_prior_type='indep_ctrl',
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
                    # 'pos': 1.e1,
                    'pos': 0.,
                    'theta': 0.,
                    'vel': 0.,
                    'theta_dot': 0.,
                    'ctrl': 0.,
                    # 'e_cost': 0.,
                    'e_cost': 1.e3,
                    'pos_T': 1e5,
                    'theta_T': 0.,
                    'vel_T': 0.,
                    'theta_dot_T': 0.,
            },
            tensor_args=tensor_args,
        )
        system = UnicycleSystem(**system_params)
    else:
        raise IOError

    controller = MPPI(system, **mppi_params)
    generator = MPCSampler(controller, system)

    def vis_fn(model, itr, save_path, trajs=None):
        mppi_fig = plot_mppi(
            model,
            minx=to_numpy(dataset.min[0])-0.5, maxx=to_numpy(dataset.max[0])+0.5,
            miny=to_numpy(dataset.min[1])-0.5, maxy=to_numpy(dataset.max[1])+0.5,
            device=device,
            iter=itr,
            # e_lim=[-1., 10.],
            e_lim=None,
            trajs=trajs,
        )
        mppi_fig.savefig(save_path / 'mppi_{:05d}.png'.format(itr))
        plt.close('all')

    traj_history = []
    for _ in range(250):
        _, trajectories, _ = generator.get_traj_samples(ebm, start_state, goal_state)
        traj_history.append(trajectories)

    for iter, trajs in enumerate(traj_history):
        vis_fn(ebm, iter, root / 'images', trajs=trajs.cpu().numpy())
