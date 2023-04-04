import numpy as np
import math
def levy(n, m, beta):
    num = math.gamma(1 + beta) * math.sin(math.pi * beta / 2)
    den = math.gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2))
    sigma_u = (num / den) ** (1 / beta)
    u = np.random.normal(scale=sigma_u, size=(n, m))
    v = np.random.normal(size=(n, m))
    z = u / (np.abs(v) ** (1 / beta))
    return z

def initialization(search_agents_no, dim, ub, lb):
    boundary_no = 1
    if type(ub) == np.ndarray:
        boundary_no = ub.shape(0)
    if type(ub) == list:
        boundary_no = len(ub)
    if boundary_no == 1:
        positions = np.random.rand(search_agents_no, dim) * (ub - lb) + lb
    else:
        positions = np.zeros((search_agents_no, dim))
        for i in range(dim):
            ub_i = ub[i]
            lb_i = lb[i]
            positions[:, i] = np.random.rand(search_agents_no, ) * (ub_i - lb_i) + lb_i
    return positions

def F1(x):
    return np.sum(np.power(x, 2), axis=0)
def F2(x):
    return np.prod(np.abs(x), axis=0) + np.sum(np.abs(x), axis=0)
def F3(x):
    return np.max(np.abs(x), axis=0)


def get_functions_details(function_name):
    fobj = F1
    lb = -100
    ub = 100
    dim = 50
    if function_name == 'F1':
        fobj = F1
        lb = -100
        ub = 100
        dim = 50
    elif function_name == 'F2':
        fobj = F2
        lb = -10
        ub = 10
        dim = 50
    elif function_name == 'F3':
        fobj = F3
        lb = -100
        ub = 100
        dim = 50
    return [lb, ub, dim, fobj]

class Mpa():
    def __init__(self, search_agents_no=25, max_iter=500, lb=-100, ub=100, dim=50, fobj=F1, FADs=0.2, P=0.5):
        self.search_agents_no = search_agents_no
        self.max_iter = max_iter
        self.lb = lb
        self.ub = ub
        self.dim = dim
        self.fobj = fobj
        self.FADs = FADs
        self.P = P

    def opt(self):
        search_agents_no = self.search_agents_no
        max_iter = self.max_iter
        lb = self.lb
        ub = self.ub
        dim = self.dim
        fobj = self.fobj
        FADs = self.FADs
        P = self.P
        top_predator_pos = np.zeros((1, dim))
        top_predator_fit = np.inf
        convergence_curve = np.zeros((max_iter,))
        step_size = np.zeros((search_agents_no, dim))
        fitness = np.full((search_agents_no, 1), np.inf)
        prey = initialization(search_agents_no, dim, ub, lb)
        x_min = np.tile(np.ones(dim, ) * lb, (search_agents_no, 1))
        x_max = np.tile(np.ones(dim, ) * ub, (search_agents_no, 1))
        iter = 0
        cnt = 0
        while iter < max_iter:
            for i in range(prey.shape[0]):
                flag_ub = np.where(prey[i, :] > ub, 1, 0)
                flag_lb = np.where(prey[i, :] < lb, 1, 0)
                flag = flag_ub + flag_lb
                prey[i, :] = prey[i, :] * np.where(flag != 0, 0, 1) + ub * flag_ub + lb * flag_lb
                fitness[i, 0] = fobj(prey[i, :])
                if fitness[i, 0] < top_predator_fit:
                    cnt = cnt + 1
                    top_predator_fit = fitness[i, 0].copy()
                    top_predator_pos = prey[i, :].copy()
            if iter == 0:
                fit_old = fitness.copy()
                prey_old = prey.copy()

            inx = np.where(fit_old < fitness, 1, 0)
            indx = np.tile(inx, (1, dim))
            prey = indx * prey_old + np.where(indx != 0, 0, 1) * prey
            fitness = inx * fit_old + np.where(inx != 0, 0, 1) * fitness
            fit_old = fitness.copy()
            prey_old = prey.copy()

            elite = np.tile(top_predator_pos, (search_agents_no, 1))
            CF = (1 - iter / max_iter) ** (2 * iter / max_iter)

            RL = 0.05 * levy(search_agents_no, dim, 1.5)
            RB = np.random.randn(search_agents_no, dim)

            for i in range(prey.shape[0]):
                for j in range(prey.shape[1]):
                    R = np.random.rand()
                    if iter < max_iter / 3:
                        step_size[i, j] = RB[i, j] * (elite[i, j] - RB[i, j] * prey[i, j])
                        prey[i, j] = prey[i, j] + P * R * step_size[i, j]
                    elif max_iter / 3 <= iter < 2 * max_iter / 3:
                        if i > prey.shape[0] / 2:
                            step_size[i, j] = RB[i, j] * (RB[i, j] * elite[i, j] - prey[i, j])
                            prey[i, j] = elite[i, j] + P * CF * step_size[i, j]
                        else:
                            step_size[i, j] = RL[i, j] * (elite[i, j] - RL[i, j] * prey[i, j])
                            prey[i, j] = prey[i, j] + P * R * step_size[i, j]
                    else:
                        step_size[i, j] = RL[i, j] * (RL[i, j] * elite[i, j] - prey[i, j])
                        prey[i, j] = elite[i, j] + P * CF * step_size[i, j]

            for i in range(prey.shape[0]):
                flag_ub = np.where(prey[i, :] > ub, 1, 0)
                flag_lb = np.where(prey[i, :] < lb, 1, 0)
                flag = flag_ub + flag_lb
                prey[i, :] = prey[i, :] * np.where(flag != 0, 0, 1) + ub * flag_ub + lb * flag_lb

                fitness[i, 0] = fobj(prey[i, :])
                if fitness[i, 0] < top_predator_fit:
                    top_predator_fit = fitness[i, 0].copy()
                    top_predator_pos = prey[i, :].copy()

            if iter == 0:
                fit_old = fitness.copy()
                prey_old = prey.copy()

            inx = np.where(fit_old < fitness, 1, 0)
            indx = np.tile(inx, (1, dim))
            prey = indx * prey_old + np.where(indx != 0, 0, 1) * prey
            fitness = inx * fit_old + np.where(inx != 0, 0, 1) * fitness
            fit_old = fitness.copy()
            prey_old = prey.copy()

            if np.random.rand() < FADs:
                U = np.where(np.random.rand(search_agents_no, dim) < FADs, 1, 0)
                prey = prey + CF * ((x_min + np.random.rand(search_agents_no, dim) * (x_max - x_min)) * U)
            else:
                r = np.random.rand()
                Rs = prey.shape[0]
                step_size = (FADs * (1 - r) + r) * (
                            prey[np.random.permutation(Rs), :] - prey[np.random.permutation(Rs), :])
                prey = prey + step_size

            convergence_curve[iter] = top_predator_fit
            iter = iter + 1

        return [top_predator_fit, top_predator_pos, convergence_curve]

class QIMpa(Mpa):
    def __init__(self, search_agents_no=25, max_iter=500, lb=-100, ub=100, dim=50, fobj=F1, FADs=0.2, P=0.5):
        self.search_agents_no = search_agents_no
        self.max_iter = max_iter
        self.lb = lb
        self.ub = ub
        self.dim = dim
        self.fobj = fobj
        self.FADs = FADs
        self.P = P


    def opt(self):
        print("QI Mpa")
        search_agents_no = self.search_agents_no
        max_iter = self.max_iter
        lb = self.lb
        ub = self.ub
        dim = self.dim
        fobj = self.fobj
        FADs = self.FADs
        P = self.P

        top_predator_pos = np.zeros((1, dim))
        top_predator_fit = np.inf
        convergence_curve = np.zeros((max_iter,))
        step_size = np.zeros((search_agents_no, dim))
        fitness = np.full((search_agents_no, 1), np.inf)
        best_fitness = np.full((search_agents_no, 1), np.inf)
        prey = initialization(search_agents_no, dim, ub, lb)
        x_min = np.tile(np.ones(dim, ) * lb, (search_agents_no, 1))
        x_max = np.tile(np.ones(dim, ) * ub, (search_agents_no, 1))
        iter = 0
        FADs = 0.2
        P = 0.5
        cnt = 0
        while iter < max_iter:
            for i in range(prey.shape[0]):
                flag_ub = np.where(prey[i, :] > ub, 1, 0)
                flag_lb = np.where(prey[i, :] < lb, 1, 0)
                flag = flag_ub + flag_lb
                prey[i, :] = prey[i, :] * np.where(flag != 0, 0, 1) + ub * flag_ub + lb * flag_lb

                fitness[i, 0] = fobj(prey[i, :])
                if fitness[i, 0] < best_fitness[i, 0]:
                    best_fitness[i, 0] = fitness[i, 0]
                if fitness[i, 0] < top_predator_fit:
                    cnt = cnt + 1
                    top_predator_fit = fitness[i, 0].copy()
                    top_predator_pos = prey[i, :].copy()
            if iter == 0:
                fit_old = fitness.copy()
                prey_old = prey.copy()

            inx = np.where(fit_old < fitness, 1, 0)
            indx = np.tile(inx, (1, dim))
            prey = indx * prey_old + np.where(indx != 0, 0, 1) * prey
            fitness = inx * fit_old + np.where(inx != 0, 0, 1) * fitness
            fit_old = fitness.copy()
            prey_old = prey.copy()

            elite = np.tile(top_predator_pos, (search_agents_no, 1))
            CF = (1 - iter / max_iter) ** (2 * iter / max_iter)

            RL = 0.05 * levy(search_agents_no, dim, 1.5)
            RB = np.random.randn(search_agents_no, dim)

            for i in range(prey.shape[0]):
                for j in range(prey.shape[1]):
                    R = np.random.rand()
                    if iter < max_iter / 3:
                        step_size[i, j] = RB[i, j] * (elite[i, j] - RB[i, j] * prey[i, j])
                        prey[i, j] = prey[i, j] + P * R * step_size[i, j]
                    elif max_iter / 3 <= iter < 2 * max_iter / 3:
                        if i > prey.shape[0] / 2:
                            step_size[i, j] = RB[i, j] * (RB[i, j] * elite[i, j] - prey[i, j])
                            prey[i, j] = elite[i, j] + P * CF * step_size[i, j]
                        else:
                            rnd_i = np.random.randint(0, prey.shape[0] - 1)
                            rnd_k = np.random.randint(0, prey.shape[0] - 1)
                            numerator = (prey[rnd_i, j] ** 2 - top_predator_pos[j] ** 2) * best_fitness[rnd_k] + (
                                        top_predator_pos[j] ** 2 - prey[rnd_k, j] ** 2) * best_fitness[rnd_i] + (
                                                    prey[rnd_k, j] ** 2 - prey[rnd_i, j] ** 2) * top_predator_fit
                            denominator = (prey[rnd_i, j] - top_predator_pos[j]) * best_fitness[rnd_k] + (
                                        top_predator_pos[j] - prey[rnd_k, j]) * best_fitness[rnd_i] + (
                                                      prey[rnd_k, j] - prey[rnd_i, j]) * top_predator_fit + 1e-50
                            prey[i, j] = 0.5 * numerator / denominator
                            # step_size[i, j] = RL[i, j] * (elite[i, j] - RL[i, j] * prey[i, j])
                            # prey[i, j] = prey[i, j] + P * R * step_size[i, j]

                    else:
                        step_size[i, j] = RL[i, j] * (RL[i, j] * elite[i, j] - prey[i, j])
                        prey[i, j] = elite[i, j] + P * CF * step_size[i, j]

            for i in range(prey.shape[0]):
                flag_ub = np.where(prey[i, :] > ub, 1, 0)
                flag_lb = np.where(prey[i, :] < lb, 1, 0)
                flag = flag_ub + flag_lb
                prey[i, :] = prey[i, :] * np.where(flag != 0, 0, 1) + ub * flag_ub + lb * flag_lb

                fitness[i, 0] = fobj(prey[i, :])
                if fitness[i, 0] < best_fitness[i, 0]:
                    best_fitness[i, 0] = fitness[i, 0]
                if fitness[i, 0] < top_predator_fit:
                    top_predator_fit = fitness[i, 0].copy()
                    top_predator_pos = prey[i, :].copy()

            if iter == 0:
                fit_old = fitness.copy()
                prey_old = prey.copy()

            inx = np.where(fit_old < fitness, 1, 0)
            indx = np.tile(inx, (1, dim))
            prey = indx * prey_old + np.where(indx != 0, 0, 1) * prey
            fitness = inx * fit_old + np.where(inx != 0, 0, 1) * fitness
            fit_old = fitness.copy()
            prey_old = prey.copy()

            if np.random.rand() < FADs:
                U = np.where(np.random.rand(search_agents_no, dim) < FADs, 1, 0)
                prey = prey + CF * ((x_min + np.random.rand(search_agents_no, dim) * (x_max - x_min)) * U)
            else:
                r = np.random.rand()
                Rs = prey.shape[0]
                step_size = (FADs * (1 - r) + r) * (
                            prey[np.random.permutation(Rs), :] - prey[np.random.permutation(Rs), :])
                prey = prey + step_size

            convergence_curve[iter] = top_predator_fit
            iter = iter + 1

        return [top_predator_fit, top_predator_pos, convergence_curve]



