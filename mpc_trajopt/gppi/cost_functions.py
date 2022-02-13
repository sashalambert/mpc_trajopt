import torch
from mpc.factors.gp_factor import GPFactor
from mpc.factors.unary_factor import UnaryFactor
from mpc.obstacle_map.obs_map import ObstacleMap


class CostComposite:

    def __init__(
        self,
        cost_func_list,
        traj_len,
        dim,
        tensor_args=None
    ):
        self.dim = dim
        self.traj_len = traj_len
        self.cost_func_list = cost_func_list
        self.tensor_args = tensor_args

    def eval(self, trajs, observation):
        trajs = trajs.reshape(-1, self.traj_len, self.dim)
        costs = torch.zeros(trajs.shape[0], **self.tensor_args)

        for cost_func in self.cost_func_list:
            costs += cost_func(trajs, observation)

        # # Prior cost
        # costs = self.cost_prior_func(trajs, observation)
        #
        # # Obstacle cost
        # costs += self.cost_2D_obst(trajs, observation)

        return costs

class CostFactoredMultiGoal:

    def __init__(
        self,
        n_dof,
        start_state,
        goal_states,
        traj_len,
        dt,
        sigma_params=None,
        tensor_args=None,
    ):
        self.n_dof = n_dof
        self.dim = n_dof * 2 # Pos + Vel
        self.start_state = start_state
        self.goal_states = goal_states
        self.num_goals = goal_states.shape[0]
        self.traj_len = traj_len
        self.dt = dt

        self.sigma_start = sigma_params['sigma_start']
        self.sigma_goal = sigma_params['sigma_goal']
        self.sigma_gp = sigma_params['sigma_gp']
        self.tensor_args = tensor_args

        self.set_cost_factors()

    def set_cost_factors(self):

        #========= Cost factors ===============
        self.start_prior = UnaryFactor(
            self.dim,
            self.sigma_start,
            self.start_state,
            self.tensor_args,
        )

        self.gp_prior = GPFactor(
            self.n_dof,
            self.sigma_gp,
            self.dt,
            self.traj_len - 1,
            self.tensor_args,
        )

        self.multi_goal_prior = []
        for i in range(self.num_goals):
            self.multi_goal_prior.append(
                UnaryFactor(
                    self.dim,
                    self.sigma_goal,   # Assume same goal Cov.
                    self.goal_states[i],
                    self.tensor_args,
                )
            )

    def eval(self, trajs, observation):

        trajs = trajs.reshape(-1, self.traj_len, self.dim)

        # Start cost
        err_p = self.start_prior.get_error(trajs[:, [0]], calc_jacobian=False)
        w_mat = self.start_prior.K
        start_costs = err_p @ w_mat.unsqueeze(0) @ err_p.transpose(1, 2)
        start_costs = start_costs.squeeze()

        # GP cost
        err_gp = self.gp_prior.get_error(trajs, calc_jacobian=False)
        w_mat = self.gp_prior.Q_inv[0] # repeated Q_inv
        w_mat = w_mat.reshape(1, 1, self.dim, self.dim)
        gp_costs = err_gp.transpose(2, 3) @ w_mat @ err_gp
        gp_costs = gp_costs.sum(1)
        gp_costs = gp_costs.squeeze()

        costs = start_costs + gp_costs

        # Multi-Goal cost
        trajs = trajs.reshape(self.num_goals, -1, self.traj_len, self.dim)
        costs = costs.reshape(self.num_goals, -1)
        for i in range(self.num_goals):
            err_g = self.multi_goal_prior[i].get_error(trajs[i, :, [-1]], calc_jacobian=False)
            w_mat = self.multi_goal_prior[i].K
            goal_costs = err_g @ w_mat.unsqueeze(0) @ err_g.transpose(1, 2)
            goal_costs = goal_costs.squeeze()
            costs[i] += goal_costs
        costs = costs.reshape(-1)

        return costs


class CostObst2D:

    def __init__(
        self,
        traj_len,
        weight_obst,
        tensor_args=None,
    ):
        self.dim = 4  # 2D Pos + Vel
        self.traj_len = traj_len
        self.weight_obst = weight_obst
        self.tensor_args = tensor_args

    def eval(self, trajs, observation):
        """ """
        assert trajs.shape[-1] == self.dim
        trajs = trajs.reshape(-1, self.traj_len, self.dim)

        # Obstacle cost
        obst_cost = torch.zeros(trajs.shape[0], **self.tensor_args)
        if 'obst_map' in observation:
            obst_map = observation['obst_map']
            assert isinstance(obst_map, ObstacleMap)
            obst_cost = obst_map.get_collisions(trajs[..., :2]) * self.weight_obst
            obst_cost = obst_cost.sum(1)

        return obst_cost
