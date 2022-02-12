import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_mppi(
        netE,
        n_points=500,
        beta=1.,
        device='cpu',
        minx=-2,
        maxx=2,
        miny=-2,
        maxy=2,
        iter: int = None,
        e_lim: list = None,
        trajs: np.ndarray = None,
        show_mean = True,
):
    x = np.linspace(minx, maxx, n_points)
    y = np.linspace(miny, maxy, n_points)

    grid = np.asarray(np.meshgrid(x, y)).transpose(1, 2, 0).reshape((-1, 2))

    with torch.no_grad():
        grid = torch.from_numpy(grid).float().to(device)
        e_grid = netE(grid) * beta

    e_grid = e_grid.cpu().numpy().reshape((n_points, n_points))

    # Plot 2-D state trajectories
    fig = plt.Figure()
    ax = fig.add_subplot(111)
    cs = ax.contourf(x, y, e_grid, 100)
    # cs = ax.contourf(x, y, e_grid, 100, vmin=e_lim[0], vmax=e_lim[1])
    for i in range(trajs.shape[0]):
        ax.plot(trajs[i, :, 0], trajs[i, :, 1], 'r', alpha=0.5)
    if show_mean:
        mean_traj = trajs.mean(0)
        ax.plot(mean_traj[:, 0], mean_traj[:, 1], 'b')

    cbar = fig.colorbar(cs, ax=ax)
    # ax.axis('equal')
    ax.set_aspect('equal')
    # plt.tight_layout(pad=0., w_pad=0., h_pad=0.)
    # plt.show()

    return fig
