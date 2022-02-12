__author__ = "Alexander Lambert"
__license__ = "MIT"

import numpy as np
import torch
import matplotlib.pyplot as plt


class SDFMap:

    def __init__(
            self,
            obst_params,
            obst_type='rbf',
            obst_preset=None,
            tensor_args=None,
    ):

        self.tensor_args = tensor_args
        self.make_obst(obst_params)
        self.num_obst = self.centers.shape[0]
        self.obst_type = obst_type

    def make_obst(self, obst_params):
        obst_preset = obst_params['obst_preset']
        if obst_preset is None:
            pass
        elif obst_preset == '5x5':
            s = obst_params['obst_spacing']
            sigma = obst_params['obst_size']
            buff = obst_params['obst_buff']
            obst_params = dict(
                centers_list=[
                    [-2*s, 2*s] ,  [-s, 2*s],  [0., 2*s],  [s, 2*s], [2*s, 2*s],
                    [-2*s, s]   , [-s, s],  [0., s],  [s, s], [2*s, s],
                    [-2*s, 0.]  , [-s, 0.], [0., 0.], [s, 0.], [2*s, 0.],
                    [-2*s, -s]  , [-s, -s], [0., -s], [s, -s], [2*s, -s],
                    [-2*s, -2*s], [-s, -2*s], [0., -2*s], [s, -2*s], [2*s, -2*s],
                ],
                radii_list=[
                    [sigma],
                ] * 25,
                buff_list=[
                    [buff],
                ] * 25
            )
        else:
            raise NotImplementedError

        self.centers = torch.from_numpy(
            np.array(
                obst_params['centers_list'],
            ),
        ).to(**self.tensor_args)
        self.radii = torch.from_numpy(
            np.array(
                obst_params['radii_list'],
            ),
        ).to(**self.tensor_args)
        self.buff = torch.from_numpy(
            np.array(
                obst_params['buff_list'],
            ),
        ).to(**self.tensor_args)
        assert self.centers.shape[0] == self.radii.shape[0]

    def get_collisions(self, X, requires_grad=True):
        """
        Checks for collision in a batch of trajectories using the generated
        occupancy grid (i.e. obstacle map), and
        returns sum of collision costs for the entire batch.

        Parameters
        ----------
        X : tensor
         2D position trajectories. Shape: [batch, horizon, 2]

        Returns
        -------
        collision_probs: tensor
            Probability of collision. Shape: [batch, horizon]

        """
        assert X.shape[-1] == 2
        batch, horizon, _ = X.shape
        if self.obst_type == 'sdf':
            costs = self.get_sdf(X.reshape(-1, 2), requires_grad)
        elif self.obst_type == 'rbf':
            costs = self.get_rbf(X.reshape(-1, 2), requires_grad)
        elif self.obst_type == 'mahalanobis':
            costs = self.get_mahalanobis(X.reshape(-1, 2), requires_grad)
        else:
            raise NotImplementedError
        return costs.reshape(batch, horizon)

    def get_sdf(self, X, requires_grad=False):
        """
        Parameters
        ----------
        X : tensor
         2D position trajectories. Shape: [batch,  2]

        Returns
        -------

        """
        batch = X.shape[0]
        sdf = torch.zeros(batch, requires_grad=requires_grad, **self.tensor_args)
        for ob in range(self.num_obst):
            c_xy = self.centers[ob]
            rad = self.radii[ob]
            buff = self.buff[ob]
            diff_sq = (X - c_xy)**2
            dists = (diff_sq.sum(-1)).sqrt()
            sdf_ob = (rad + buff - dists).clamp(min=0.)
            sdf = sdf + sdf_ob
        return sdf

    def get_rbf(self, X, requires_grad):
        """
        Parameters
        ----------
        X : tensor
         2D position trajectories. Shape: [batch,  2]

        Returns
        -------

        """
        batch = X.shape[0]
        costs = torch.zeros(batch, requires_grad=requires_grad, **self.tensor_args)
        for ob in range(self.num_obst):
            c_xy = self.centers[ob]
            rad = self.radii[ob]
            diff_sq = ((X - c_xy)**2).sum(-1)
            rbf_ob = torch.exp(-0.5 * diff_sq / rad**2)
            costs = costs + rbf_ob
        return costs

    def get_mahalanobis(self, X, requires_grad):
        """
        Parameters
        ----------
        X : tensor
         2D position trajectories. Shape: [batch,  2]

        Returns
        -------

        """
        batch = X.shape[0]
        costs = torch.zeros(batch, requires_grad=requires_grad, **self.tensor_args)
        for ob in range(self.num_obst):
            c_xy = self.centers[ob]
            rad = self.radii[ob]
            diff_sq = ((X - c_xy)**2).sum(-1)
            cost_ob = (diff_sq / rad**2).sqrt()  # Identity matrix
            costs = costs + cost_ob
        return costs

    def plot_map(
            self,
            xy_lim=(-10, 10),
            res=100,
            # save_path="/tmp/sdf_obst_map.png",
            save_path=None,
    ):

        ax = plt.gca()
        x = np.linspace(xy_lim[0], xy_lim[1], res)
        y = np.linspace(xy_lim[0], xy_lim[1], res)
        X, Y = np.meshgrid(x,y)
        grid = torch.from_numpy(np.stack((X, Y), axis=-1)).to(**self.tensor_args)
        Z = self.get_sdf(grid.reshape(-1, 2)).reshape(res, res).detach().cpu().numpy()
        # Z = self.get_rbf(grid.reshape(-1, 2), False).reshape(res, res).detach().numpy()
        # Z = self.get_mahalanobis(grid.reshape(-1, 2)).reshape(res, res).detach().numpy()

        # plt.imshow(Z, cmap='viridis', origin='lower')
        cs = plt.contourf(X, Y, Z, 10, cmap=plt.cm.binary, origin='lower')
        plt.colorbar()
        ax.xlim = xy_lim
        ax.ylim = xy_lim

        # if save_path is not None:
        #     plt.savefig(save_path)
        plt.show()

if __name__ == "__main__":

    device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    tensor_args = {'device': device, 'dtype': torch.float32}

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
        obst_preset=None,
        # obst_preset='5x5',
        obst_spacing=2.5,
        obst_size=5.,
        obst_buff=0.1,
    )

    map = SDFMap(
        obst_params,
        tensor_args=tensor_args,
    )

    map.plot_map(
        xy_lim=(-10, 10),
    )