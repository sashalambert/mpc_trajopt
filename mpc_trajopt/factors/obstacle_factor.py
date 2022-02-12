__author__ = "Alexander Lambert"
__license__ = "MIT"


import numpy as np
import torch
from ebm_il.ebm_models.conditional_ebms import cond_MLP
from mpc.obstacle_map.sdf_map import SDFMap

class ObstacleFactor:

    def __init__(
            self,
            dof,
            sigma,
            num_factors,
    ):
        self.sigma = sigma
        self.num_factors = num_factors
        self.dof = dof
        self.K = 1. / (sigma**2)

    def get_error(
            self,
            x_traj,
            obst_func,
            obst_locs=None,
            standardizer=None,
            calc_jacobian=True,
    ):
        batch, horizon = x_traj.shape[0], x_traj.shape[1]
        # x_traj = x_traj.clone().detach()
        # x_traj.requires_grad = True
        states = x_traj[:, :, :2]
        if isinstance(obst_func, cond_MLP):
            obst_func.eval()
            states = standardizer.normalize(states)
            obst_locs = standardizer.normalize(obst_locs)
            error = obst_func(states, None, obst_locs.repeat(states.size(0), 1, 1), None)
            error = error.squeeze(-1)
        elif isinstance(obst_func, SDFMap):
            error = obst_func.get_collisions(states)
        else:
            raise TypeError('Obstacle function type not currently supported.')

        if calc_jacobian:
            H = -1. * torch.autograd.grad(error.sum(), x_traj)[0]
            error = error.detach()
            error.requires_grad = False
            return error, H.reshape(batch, horizon, self.dof)
        else:
            return error

