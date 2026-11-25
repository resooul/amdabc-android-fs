"""ABC-based optimizers used in the paper, built on the MEALPY Optimizer base class.

The adaptive coefficient of AMDABC follows the share of the FE budget used so
far when a FE budget is active (see PaperOptimizer._progress).

Paper names: AMDABC = MD3ABC, AMDABC1 = MD31ABC, AMDABC2 = MD32ABC, AMDABC3 = MD33ABC, ABC = OriginalABC2, GBABC = GBestGuidedABC2, GBC = GBC, ABCX = ABCX, DABC = DirectedABC, ABCVSS = ABCVSS
"""
import numpy as np
from .mealpy_core.optimizer import Optimizer


class PaperOptimizer(Optimizer):
    """Common base of the ABC-based optimizers."""

    def _progress(self, epoch):
        """Search progress in [0, 1]: FE-based under a FE budget, else epoch-based."""
        prog = getattr(self.problem, "progress", None)
        return prog() if callable(prog) else epoch / self.epoch

class MD3ABC(PaperOptimizer):

    def __init__(self, epoch: int = 10000, pop_size: int = 100, n_limits: int = 25, n_best: int = 3,
                 **kwargs: object) -> None:
        """
        Args:
            epoch: maximum number of iterations, default = 10000
            pop_size: number of population size = onlooker bees = employed bees, default = 100
            n_limits: Limit of trials before abandoning a food source, default=25
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.n_limits = self.validator.check_int("n_limits", n_limits, [1, 1000])
        self.n_best = self.validator.check_int("n_best", n_best, [1, 1000])
        self.is_parallelizable = False
        self.set_parameters(["epoch", "pop_size", "n_limits", "n_best"])
        self.sort_flag = False
        self.NS = int(np.ceil(self.pop_size / 2))
        self.pop_size = self.NS

    def initialize_variables(self):
        self.trials = np.zeros(self.pop_size)

    def evolve(self, epoch):

        alpha_max = 1.0
        alpha_min = 0.1
        alpha = 0.5 * (alpha_max - alpha_min) * (1 + np.cos(self._progress(epoch) * np.pi)) + alpha_min
        # alpha = (1 - (epoch + 1) / (self.epoch + 1)) ** (2 * (epoch + 1) / (self.epoch + 1))

        current_best = self.get_best_agent(self.pop, self.problem.minmax)
        # Employed bees phase
        for idx in range(0, self.NS):

            rdx = self.generator.choice(list(set(range(0, self.NS)) - {idx}))

            # Generate two random vectors for the two terms
            phi1 = self.generator.uniform(low=-1, high=1, size=self.problem.n_dims)
            phi2 = self.generator.uniform(low=-1, high=1, size=self.problem.n_dims)

            # Calculate the new position
            # Term 1: Original ABC differential search (exploration)
            term1 = alpha * phi1 * (self.pop[rdx].solution - self.pop[idx].solution)

            # Term 2: Global Best guided term (exploitation)
            term2 = alpha * phi2 * (current_best.solution - self.pop[idx].solution)

            pos_new = self.pop[idx].solution + term1 + term2
            # pos_new = np.clip(pos_new, self.problem.lb, self.problem.ub)
            pos_new = np.multiply(pos_new, (0.5 + self.generator.uniform(0, 1, self.problem.n_dims)))

            pos_new = self.correct_solution(pos_new)
            agent = self.generate_agent(pos_new)

            if self.compare_target(agent.target, self.pop[idx].target, self.problem.minmax):
                self.pop[idx] = agent
                self.trials[idx] = 0
            else:
                self.trials[idx] += 1

        # Onlooker bees phase
        for idx in range(0, self.NS):

            rdx = self.generator.choice(list(set(range(0, self.NS)) - {idx}))

            # Generate two random vectors for the two terms
            phi1 = self.generator.uniform(low=-1, high=1, size=self.problem.n_dims)
            phi2 = self.generator.uniform(low=-1, high=1, size=self.problem.n_dims)

            # Calculate the new position
            # Term 1: Original ABC differential search (exploration)
            term1 = alpha * phi1 * (self.pop[rdx].solution - self.pop[idx].solution)

            # Term 2: Global Best guided term (exploitation)
            term2 = alpha * phi2 * (current_best.solution - self.pop[idx].solution)

            pos_new = self.pop[idx].solution + term1 + term2
            pos_new = np.multiply(pos_new, (0.5 + self.generator.uniform(0, 1, self.problem.n_dims)))

            pos_new = self.correct_solution(pos_new)
            agent = self.generate_agent(pos_new)

            if self.compare_target(agent.target, self.pop[idx].target, self.problem.minmax):
                self.pop[idx] = agent
                self.trials[idx] = 0
            else:
                self.trials[idx] += 1

        # Scout bees phase
        # Check the number of trials for each employed bee and abandon the food source if the limit is exceeded
        best_stuck = np.where(self.trials >= self.n_limits)[0]
        id_best = self.get_index_best(self.pop, self.problem.minmax)
        for idx in best_stuck:
            if idx == id_best:
                # print("Best solution stuck")
                for idx2 in range(0, self.NS):
                    # global restart: re-initialize every food source
                    self.pop[idx2] = self.generate_agent()
                    self.trials[idx2] = 0
                break

        # Scout bees phase
        scout = np.where(self.trials >= self.n_limits)[0]
        for idx in scout:
            self.pop[idx] = self.generate_agent()
            self.trials[idx] = 0
            break


class MD31ABC(PaperOptimizer):

    def __init__(self, epoch: int = 10000, pop_size: int = 100, n_limits: int = 25, **kwargs: object) -> None:
        """
        Args:
            epoch: maximum number of iterations, default = 10000
            pop_size: number of population size = onlooker bees = employed bees, default = 100
            n_limits: Limit of trials before abandoning a food source, default=25
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.n_limits = self.validator.check_int("n_limits", n_limits, [1, 1000])
        self.is_parallelizable = False
        self.set_parameters(["epoch", "pop_size", "n_limits"])
        self.sort_flag = False

        self.NS = int(np.ceil(self.pop_size / 2))
        self.pop_size = self.NS

    def initialize_variables(self):
        self.trials = np.zeros(self.NS)
        self.pop_probability = np.ones(self.NS)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """

        alpha_max = 1.0
        alpha_min = 0.1
        alpha = 0.5 * (alpha_max - alpha_min) * (1 + np.cos(self._progress(epoch) * np.pi)) + alpha_min

        for idx in range(0, self.NS):
            # Choose a random employed bee to generate a new solution
            rdx = self.generator.choice(list(set(range(0, self.NS)) - {idx}))
            # Generate a new solution by the equation x_{ij} = x_{ij} + phi_{ij} * (x_{tj} - x_{ij})
            phi = self.generator.uniform(low=-1, high=1, size=self.problem.n_dims)
            p2c = self.generator.choice(list(set(range(0, self.problem.n_dims))))
            pos_new = np.copy(self.pop[idx].solution)
            pos_new[p2c] = self.pop[idx].solution[p2c] + alpha * phi[p2c] * (
                    self.pop[rdx].solution[p2c] - self.pop[idx].solution[p2c])
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_agent(pos_new)
            if self.compare_target(agent.target, self.pop[idx].target, self.problem.minmax):
                self.pop[idx] = agent
                self.trials[idx] = 0
            else:
                self.trials[idx] += 1

        # Calculate Probs
        if self.problem.minmax == "min":
            employed_fits = np.array([agent.target.fitness for agent in self.pop])
            max_fit = np.copy(max(employed_fits))
            for idx in range(0, self.NS):
                self.pop_probability[idx] = (0.9 * (self.pop[idx].target.fitness / max_fit)) + 0.1
        else:
            employed_fits = [agent.target.fitness for agent in self.pop]  # epsilon to avoid division by zero
            min_fit = np.copy(max(employed_fits))
            for idx in range(0, self.NS):
                self.pop_probability[idx] = (0.9 * (self.pop[idx].target.fitness / min_fit)) + 0.1

        # Onlooker bees phase
        # Calculate the probabilities of each employed bee
        employed_fits = np.array([agent.target.fitness for agent in self.pop])
        # probabilities = employed_fits / np.sum(employed_fits)
        for idx in range(0, self.NS):
            # Choose a random employed bee to generate a new solution
            rdx = self.generator.choice(list(set(range(0, self.NS)) - {idx}))
            # Generate a new solution by the equation x_{ij} = x_{ij} + phi_{ij} * (x_{tj} - x_{ij})
            p2c = self.generator.choice(list(set(range(0, self.problem.n_dims))))
            pos_new = np.copy(self.pop[idx].solution)
            pos_new[p2c] = self.pop[idx].solution[p2c] + alpha * phi[p2c] * (
                        self.pop[rdx].solution[p2c] - self.pop[idx].solution[p2c])
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_agent(pos_new)
            if self.compare_target(agent.target, self.pop[idx].target, self.problem.minmax):
                self.pop[idx] = agent
                self.trials[idx] = 0
            else:
                self.trials[idx] += 1
        # Scout bees phase
        # Check the number of trials for each employed bee and abandon the food source if the limit is exceeded
        abandoned = np.where(self.trials >= self.n_limits)[0]
        for idx in abandoned:
            self.pop[idx] = self.generate_agent()
            self.trials[idx] = 0
            break


class MD32ABC(PaperOptimizer):

    def __init__(self, epoch: int = 10000, pop_size: int = 100, n_limits: int = 25, **kwargs: object) -> None:
        """
        Args:
            epoch: maximum number of iterations, default = 10000
            pop_size: number of population size = onlooker bees = employed bees, default = 100
            n_limits: Limit of trials before abandoning a food source, default=25
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.n_limits = self.validator.check_int("n_limits", n_limits, [1, 1000])
        self.is_parallelizable = False
        self.set_parameters(["epoch", "pop_size", "n_limits"])
        self.sort_flag = False

        self.NS = int(np.ceil(self.pop_size / 2))
        self.pop_size = self.NS

    def initialize_variables(self):
        self.trials = np.zeros(self.NS)
        self.pop_probability = np.ones(self.NS)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """

        alpha = 1
        current_best = self.get_best_agent(self.pop, self.problem.minmax)
        # Employed bees phase
        for idx in range(0, self.NS):

            rdx = self.generator.choice(list(set(range(0, self.NS)) - {idx}))

            # Generate two random vectors for the two terms
            phi1 = self.generator.uniform(low=-1, high=1, size=self.problem.n_dims)
            phi2 = self.generator.uniform(low=-1, high=1, size=self.problem.n_dims)

            # Calculate the new position
            # Term 1: Original ABC differential search (exploration)
            term1 = alpha * phi1 * (self.pop[rdx].solution - self.pop[idx].solution)

            # Term 2: Global Best guided term (exploitation)
            term2 = alpha * phi2 * (current_best.solution - self.pop[idx].solution)

            pos_new = self.pop[idx].solution + term1 + term2
            # pos_new = np.clip(pos_new, self.problem.lb, self.problem.ub)
            pos_new = np.multiply(pos_new, (0.5 + self.generator.uniform(0, 1, self.problem.n_dims)))

            pos_new = self.correct_solution(pos_new)
            agent = self.generate_agent(pos_new)

            if self.compare_target(agent.target, self.pop[idx].target, self.problem.minmax):
                self.pop[idx] = agent
                self.trials[idx] = 0
            else:
                self.trials[idx] += 1

        # Onlooker bees phase
        for idx in range(0, self.NS):

            rdx = self.generator.choice(list(set(range(0, self.NS)) - {idx}))

            # Generate two random vectors for the two terms
            phi1 = self.generator.uniform(low=-1, high=1, size=self.problem.n_dims)
            phi2 = self.generator.uniform(low=-1, high=1, size=self.problem.n_dims)

            # Calculate the new position
            # Term 1: Original ABC differential search (exploration)
            term1 = alpha * phi1 * (self.pop[rdx].solution - self.pop[idx].solution)

            # Term 2: Global Best guided term (exploitation)
            term2 = alpha * phi2 * (current_best.solution - self.pop[idx].solution)

            pos_new = self.pop[idx].solution + term1 + term2
            pos_new = np.multiply(pos_new, (0.5 + self.generator.uniform(0, 1, self.problem.n_dims)))

            pos_new = self.correct_solution(pos_new)
            agent = self.generate_agent(pos_new)

            if self.compare_target(agent.target, self.pop[idx].target, self.problem.minmax):
                self.pop[idx] = agent
                self.trials[idx] = 0
            else:
                self.trials[idx] += 1
        # Scout bees phase
        # Check the number of trials for each employed bee and abandon the food source if the limit is exceeded
        abandoned = np.where(self.trials >= self.n_limits)[0]
        for idx in abandoned:
            self.pop[idx] = self.generate_agent()
            self.trials[idx] = 0
            break


class MD33ABC(PaperOptimizer):

    def __init__(self, epoch: int = 10000, pop_size: int = 100, n_limits: int = 25, **kwargs: object) -> None:
        """
        Args:
            epoch: maximum number of iterations, default = 10000
            pop_size: number of population size = onlooker bees = employed bees, default = 100
            n_limits: Limit of trials before abandoning a food source, default=25
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.n_limits = self.validator.check_int("n_limits", n_limits, [1, 1000])
        self.is_parallelizable = False
        self.set_parameters(["epoch", "pop_size", "n_limits"])
        self.sort_flag = False

        self.NS = int(np.ceil(self.pop_size / 2))
        self.pop_size = self.NS

    def initialize_variables(self):
        self.trials = np.zeros(self.NS)
        self.pop_probability = np.ones(self.NS)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        for idx in range(0, self.NS):
            # Choose a random employed bee to generate a new solution
            rdx = self.generator.choice(list(set(range(0, self.NS)) - {idx}))
            # Generate a new solution by the equation x_{ij} = x_{ij} + phi_{ij} * (x_{tj} - x_{ij})
            phi = self.generator.uniform(low=-1, high=1, size=self.problem.n_dims)
            p2c = self.generator.choice(list(set(range(0, self.problem.n_dims))))
            pos_new = np.copy(self.pop[idx].solution)
            pos_new[p2c] = self.pop[idx].solution[p2c] + phi[p2c] * (
                    self.pop[rdx].solution[p2c] - self.pop[idx].solution[p2c])
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_agent(pos_new)
            if self.compare_target(agent.target, self.pop[idx].target, self.problem.minmax):
                self.pop[idx] = agent
                self.trials[idx] = 0
            else:
                self.trials[idx] += 1

        # Calculate Probs
        if self.problem.minmax == "min":
            employed_fits = np.array([agent.target.fitness for agent in self.pop])
            max_fit = np.copy(max(employed_fits))
            for idx in range(0, self.NS):
                self.pop_probability[idx] = (0.9 * (self.pop[idx].target.fitness / max_fit)) + 0.1
        else:
            employed_fits = [agent.target.fitness for agent in self.pop]  # epsilon to avoid division by zero
            min_fit = np.copy(max(employed_fits))
            for idx in range(0, self.NS):
                self.pop_probability[idx] = (0.9 * (self.pop[idx].target.fitness / min_fit)) + 0.1

        # Onlooker bees phase
        # Calculate the probabilities of each employed bee
        employed_fits = np.array([agent.target.fitness for agent in self.pop])
        # probabilities = employed_fits / np.sum(employed_fits)
        for idx in range(0, self.NS):
            # Choose a random employed bee to generate a new solution
            rdx = self.generator.choice(list(set(range(0, self.NS)) - {idx}))
            # Generate a new solution by the equation x_{ij} = x_{ij} + phi_{ij} * (x_{tj} - x_{ij})
            p2c = self.generator.choice(list(set(range(0, self.problem.n_dims))))
            pos_new = np.copy(self.pop[idx].solution)
            pos_new[p2c] = self.pop[idx].solution[p2c] + phi[p2c] * (
                        self.pop[rdx].solution[p2c] - self.pop[idx].solution[p2c])
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_agent(pos_new)
            if self.compare_target(agent.target, self.pop[idx].target, self.problem.minmax):
                self.pop[idx] = agent
                self.trials[idx] = 0
            else:
                self.trials[idx] += 1

        # Scout bees phase
        # Check the number of trials for each employed bee and abandon the food source if the limit is exceeded
        best_stuck = np.where(self.trials >= self.n_limits)[0]
        id_best = self.get_index_best(self.pop, self.problem.minmax)
        for idx in best_stuck:
            if idx == id_best:
                # print("Best solution stuck")
                for idx2 in range(0, self.NS):
                    # global restart: re-initialize every food source
                    self.pop[idx2] = self.generate_agent()
                    self.trials[idx2] = 0
                break

        # Scout bees phase
        scout = np.where(self.trials >= self.n_limits)[0]
        for idx in scout:
            self.pop[idx] = self.generate_agent()
            self.trials[idx] = 0
            break


class OriginalABC2(PaperOptimizer):

    def __init__(
        self,
        epoch: int = 1000,
        pop_size: int = 100,
        n_limits: int = 25,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int(
            "pop_size", pop_size, [5, 10000]
        )
        self.n_limits = self.validator.check_int(
            "n_limits", n_limits, [1, 1000]
        )
        self.is_parallelizable = False
        self.set_parameters(["epoch", "pop_size", "n_limits"])
        self.sort_flag = False

        self.NS = int(np.ceil(self.pop_size / 2))
        self.pop_size = self.NS

    def initialize_variables(self):
        self.trials = np.zeros(self.NS)

    def evolve(self, epoch):
        # ==================== 1. EMPLOYED BEES PHASE ====================
        # Her işçi arı için 1 FFE -> Toplam: NS FFE
        for idx in range(self.NS):
            rdx = self.generator.choice(
                [i for i in range(self.NS) if i != idx]
            )
            p2c = self.generator.integers(0, self.problem.n_dims)
            phi = self.generator.uniform(-1.0, 1.0)

            pos_new = np.copy(self.pop[idx].solution)
            pos_new[p2c] = self.pop[idx].solution[p2c] + phi * (
                self.pop[rdx].solution[p2c] - self.pop[idx].solution[p2c]
            )

            pos_new = self.correct_solution(pos_new)
            agent = self.generate_agent(pos_new)

            if self.compare_target(
                agent.target, self.pop[idx].target, self.problem.minmax
            ):
                self.pop[idx] = agent
                self.trials[idx] = 0
            else:
                self.trials[idx] += 1

        # ==================== 2. CALCULATE PROBABILITIES ====================
        fits = np.array([agent.target.fitness for agent in self.pop])

        if self.problem.minmax == "min":
            raw_fits = np.where(fits >= 0, 1.0 / (1.0 + fits), 1.0 + np.abs(fits))
        else:
            raw_fits = np.where(fits >= 0, 1.0 + fits, 1.0 / (1.0 + np.abs(fits)))

        total_fit = np.sum(raw_fits)
        if total_fit == 0:
            probs = np.ones(self.NS) / self.NS
        else:
            probs = raw_fits / total_fit

        # ==================== 3. ONLOOKER BEES PHASE ====================
        # Rulet çarkı yöntemiyle (p=probs) tam olarak NS adet besin kaynağı seçilir.
        # Bu işlem %100 olarak NS adet FFE yapılmasını garanti eder.
        selected_indices = self.generator.choice(
            self.NS, size=self.NS, p=probs
        )

        for idx in selected_indices:
            rdx = self.generator.choice(
                [k for k in range(self.NS) if k != idx]
            )
            p2c = self.generator.integers(0, self.problem.n_dims)
            phi = self.generator.uniform(-1.0, 1.0)

            pos_new = np.copy(self.pop[idx].solution)
            pos_new[p2c] = self.pop[idx].solution[p2c] + phi * (
                self.pop[rdx].solution[p2c] - self.pop[idx].solution[p2c]
            )

            pos_new = self.correct_solution(pos_new)
            agent = self.generate_agent(pos_new)

            if self.compare_target(
                agent.target, self.pop[idx].target, self.problem.minmax
            ):
                self.pop[idx] = agent
                self.trials[idx] = 0
            else:
                self.trials[idx] += 1

        # ==================== 4. SCOUT BEES PHASE ====================
        # En çok başarısız olan tek çözümü terk et
        max_trial_idx = np.argmax(self.trials)
        if self.trials[max_trial_idx] >= self.n_limits:
            self.pop[max_trial_idx] = self.generate_agent()
            self.trials[max_trial_idx] = 0


class GBestGuidedABC2(PaperOptimizer):
    """
    The original version of: Artificial Bee Colony (ABC)

    Links:
        1. https://www.sciencedirect.com/topics/computer-science/artificial-bee-colony

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + n_limits (int): Limit of trials before abandoning a food source, default=25

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, ABC
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),
    >>>     "minmax": "min",
    >>>     "obj_func": objective_function
    >>> }
    >>>
    >>> model = ABC.OriginalABC(epoch=1000, pop_size=50, n_limits = 50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] B. Basturk, D. Karaboga, An artificial bee colony (ABC) algorithm for numeric function optimization,
    in: IEEE Swarm Intelligence Symposium 2006, May 12–14, Indianapolis, IN, USA, 2006.
    """
    def __init__(self, epoch: int = 10000, pop_size: int = 100, n_limits: int = 25, **kwargs: object) -> None:
        """
        Args:
            epoch: maximum number of iterations, default = 10000
            pop_size: number of population size = onlooker bees = employed bees, default = 100
            n_limits: Limit of trials before abandoning a food source, default=25
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.n_limits = self.validator.check_int("n_limits", n_limits, [1, 1000])
        self.is_parallelizable = False
        self.set_parameters(["epoch", "pop_size", "n_limits"])
        self.sort_flag = False

        self.NS = int(np.ceil(self.pop_size / 2))
        self.pop_size = self.NS

    def initialize_variables(self):
        self.trials = np.zeros(self.NS)
        self.pop_probability = np.ones(self.NS)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        for idx in range(0, self.NS):
            # Choose a random employed bee to generate a new solution
            rdx = self.generator.choice(list(set(range(0, self.NS)) - {idx}))
            # Generate a new solution by the equation x_{ij} = x_{ij} + phi_{ij} * (x_{tj} - x_{ij})
            phi = self.generator.uniform(low=-1, high=1, size=self.problem.n_dims)
            psi = self.generator.uniform(low=0, high=1.5, size=self.problem.n_dims)
            p2c = self.generator.choice(list(set(range(0, self.problem.n_dims))))
            pos_new = np.copy(self.pop[idx].solution)
            pos_new[p2c] = self.pop[idx].solution[p2c] + phi[p2c] * (
                    self.pop[idx].solution[p2c] - self.pop[rdx].solution[p2c])  + psi[p2c] * (
                    self.g_best.solution[p2c] - self.pop[idx].solution[p2c])
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_agent(pos_new)
            if self.compare_target(agent.target, self.pop[idx].target, self.problem.minmax):
                self.pop[idx] = agent
                self.trials[idx] = 0
            else:
                self.trials[idx] += 1

        # Calculate Probs
        if self.problem.minmax == "min":
            employed_fits = np.array([agent.target.fitness for agent in self.pop])
            max_fit = np.copy(max(employed_fits))
            for idx in range(0, self.NS):
                self.pop_probability[idx] = (0.9 * (self.pop[idx].target.fitness / max_fit)) + 0.1
        else:
            employed_fits = [agent.target.fitness for agent in self.pop]  # epsilon to avoid division by zero
            min_fit = np.copy(max(employed_fits))
            for idx in range(0, self.NS):
                self.pop_probability[idx] = (0.9 * (self.pop[idx].target.fitness / min_fit)) + 0.1

        # Onlooker bees phase
        # Calculate the probabilities of each employed bee
        employed_fits = np.array([agent.target.fitness for agent in self.pop])
        # probabilities = employed_fits / np.sum(employed_fits)
        for idx in range(0, self.NS):

            # Choose a random employed bee to generate a new solution
            rdx = self.generator.choice(list(set(range(0, self.NS)) - {idx}))
            # Generate a new solution by the equation x_{ij} = x_{ij} + phi_{ij} * (x_{tj} - x_{ij})
            phi = self.generator.uniform(low=-1, high=1, size=self.problem.n_dims)
            psi = self.generator.uniform(low=0, high=1.5, size=self.problem.n_dims)
            p2c = self.generator.choice(list(set(range(0, self.problem.n_dims))))
            pos_new = np.copy(self.pop[idx].solution)
            pos_new[p2c] = self.pop[idx].solution[p2c] + phi[p2c] * (
                    self.pop[idx].solution[p2c] - self.pop[rdx].solution[p2c]) + psi[p2c] * (
                                   self.g_best.solution[p2c] - self.pop[idx].solution[p2c])
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_agent(pos_new)
            if self.compare_target(agent.target, self.pop[idx].target, self.problem.minmax):
                self.pop[idx] = agent
                self.trials[idx] = 0
            else:
                self.trials[idx] += 1
        # Scout bees phase
        # Check the number of trials for each employed bee and abandon the food source if the limit is exceeded
        abandoned = np.where(self.trials >= self.n_limits)[0]
        for idx in abandoned:
            self.pop[idx] = self.generate_agent()
            self.trials[idx] = 0
            break


class GBC(PaperOptimizer):
    """
    The original version of: Genetic Bee Colony (GBC) algorithm.

    This is a hybrid algorithm that combines the Artificial Bee Colony (ABC) framework with operators from Genetic Algorithms (GA) to enhance performance, particularly for feature selection problems.

    ### Notes:
    * **Preprocessing**: The paper proposes a preprocessing step using mRMR (minimum Redundancy Maximum Relevance) to filter features. This step is external to the optimizer and is not implemented here. To replicate the paper's results, you should perform feature selection before using this optimizer.
    * **Onlooker Bee Phase**: This phase is significantly modified. Instead of the standard ABC update, it uses a uniform crossover operation. The two parents for the crossover are:
        1.  The best solution found so far (the "Queen Bee").
        2.  A new candidate solution generated from an employee bee selected via a roulette wheel.
    * **Scout Bee Phase**: This phase is enhanced to potentially replace up to two abandoned solutions per iteration, increasing the replacement rate.
        1.  **First Scout**: Replaces an abandoned solution with a new, randomly generated one (as in standard ABC).
        2.  **Second Scout**: Replaces a second abandoned solution by applying a mutation operator to the Queen Bee.
    * **Parameter Contradiction**: The source paper contains a contradiction for the Crossover Propriety Rate (CPR). [cite_start]The text specifies a value of 0.6 [cite: 294, 366][cite_start], while the experimental setup in Table 2 lists 0.8[cite: 350]. We have used **0.8** as the default value, as it reflects the tabulated experimental parameters.

    ### Hyper-parameters:
    * **n_limits** (int): The number of trials after which a solution is considered abandoned if it has not improved. Recommended range: [10, 100]. [cite_start]Default is 5, as used in the paper[cite: 350].
    * **cpr** (float): Crossover Propriety Rate, controlling the gene swapping in the onlooker phase. Recommended range: [0.7, 0.95]. [cite_start]Default is 0.8[cite: 350].
    * **mpr** (float): Mutation Propriety Rate for the second scout bee. Recommended range: [0.001, 0.05]. [cite_start]Default is 0.01[cite: 321, 350].

    ### Examples:
    ```python
    >>> import numpy as np
    >>> from mealpy import FloatVar
    >>> # from your_gbc_file import GBC  # Assuming you save the code in your_gbc_file.py
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),
    >>>     "minmax": "min",
    >>>     "obj_func": objective_function
    >>> }
    >>>
    >>> model = GBC(epoch=1000, pop_size=50, n_limits=5, cpr=0.8, mpr=0.01)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    ```

    ### References:
    [1] Alshamlan, H. M., Badr, G. H., & Alohali, Y. A. (2015). Genetic Bee Colony (GBC) algorithm:
    A new gene selection method for microarray cancer classification.
    Computational biology and chemistry, 56, 49-60.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, n_limits: int = 5, cpr: float = 0.8, mpr: float = 0.01,
                 **kwargs: object) -> None:
        """
        Args:
            epoch (int): Maximum number of iterations.
            pop_size (int): Number of solutions in the population (food sources).
            n_limits (int): Limit of trials before a solution is abandoned.
            cpr (float): Crossover Propriety Rate.
            mpr (float): Mutation Propriety Rate.
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.n_limits = self.validator.check_int("n_limits", n_limits, [1, 1000])
        self.cpr = self.validator.check_float("cpr", cpr, (0, 1.0))
        self.mpr = self.validator.check_float("mpr", mpr, (0, 1.0))

        self.set_parameters(["epoch", "pop_size", "n_limits", "cpr", "mpr"])
        self.sort_flag = False
        self.is_parallelizable = False

        self.NS = int(np.ceil(self.pop_size / 2))
        self.pop_size = self.NS

    def initialize_variables(self):
        """
        Initialize the trials counter for each solution.
        """
        self.trials = np.zeros(self.pop_size)

    def evolve(self, epoch):
        """
        The main evolutionary loop of the GBC algorithm.

        Args:
            epoch (int): The current iteration number.
        """
        # 🐝 Employee Bee Phase
        for i in range(self.pop_size):
            # Choose a random partner solution 'k'
            k = self.generator.choice(list(set(range(0, self.pop_size)) - {i}))

            # Generate a new candidate solution by perturbing a single dimension
            pos_new = self.pop[i].solution.copy()
            j = self.generator.integers(0, self.problem.n_dims)
            phi = self.generator.uniform(-1, 1)
            pos_new[j] += phi * (pos_new[j] - self.pop[k].solution[j])

            pos_new = self.correct_solution(pos_new)
            agent = self.generate_agent(pos_new)

            # Greedy selection: if the new solution is better, replace the old one
            if self.compare_target(agent.target, self.pop[i].target, self.problem.minmax):
                self.pop[i] = agent
                self.trials[i] = 0
            else:
                self.trials[i] += 1

        # 👀 Onlooker Bee Phase
        queen_bee = self.g_best  # The best solution found so far

        # Calculate probabilities for roulette wheel selection
        fitnesses = np.array([agent.target.fitness for agent in self.pop])
        if self.problem.minmax == "min":
            # For minimization, better solutions have lower fitness. Invert for probability.
            max_fit = np.max(fitnesses)
            probs = max_fit - fitnesses + 1e-9
        else:  # For maximization
            probs = fitnesses - np.min(fitnesses) + 1e-9

        total_prob = np.sum(probs)
        probs = probs / total_prob if total_prob != 0 else np.ones(self.pop_size) / self.pop_size

        for _ in range(self.pop_size):
            # Select an employee bee based on its fitness
            selected_idx = self.generator.choice(range(self.pop_size), p=probs)
            selected_bee = self.pop[selected_idx]

            # Create Parent 2 by generating a candidate solution from the selected bee
            k = self.generator.choice(list(set(range(0, self.pop_size)) - {selected_idx}))
            j = self.generator.integers(0, self.problem.n_dims)
            phi = self.generator.uniform(-1, 1)
            parent2_pos = selected_bee.solution.copy()
            parent2_pos[j] += phi * (parent2_pos[j] - self.pop[k].solution[j])
            parent2_pos = self.correct_solution(parent2_pos)

            # Parent 1 is the Queen Bee
            parent1_pos = queen_bee.solution

            # Perform Uniform Crossover
            offspring1_pos, offspring2_pos = np.zeros_like(parent1_pos), np.zeros_like(parent1_pos)
            for dim in range(self.problem.n_dims):
                if self.generator.random() < self.cpr:
                    offspring1_pos[dim], offspring2_pos[dim] = parent1_pos[dim], parent2_pos[dim]
                else:
                    offspring1_pos[dim], offspring2_pos[dim] = parent2_pos[dim], parent1_pos[dim]

            # Evaluate offspring and select the better one
            agent1 = self.generate_agent(self.correct_solution(offspring1_pos))
            agent2 = self.generate_agent(self.correct_solution(offspring2_pos))
            best_offspring = agent1 if self.compare_target(agent1.target, agent2.target,
                                                           self.problem.minmax) else agent2

            # Greedy selection against the original selected bee
            if self.compare_target(best_offspring.target, selected_bee.target, self.problem.minmax):
                self.pop[selected_idx] = best_offspring
                self.trials[selected_idx] = 0
            else:
                self.trials[selected_idx] += 1

        #  scouts Scout Bee Phase
        abandoned_indices = np.where(self.trials >= self.n_limits)[0]

        if len(abandoned_indices) > 0:
            # First Scout Bee: replaces the first abandoned solution with a random one
            idx_to_replace_1 = abandoned_indices[0]
            self.pop[idx_to_replace_1] = self.generate_agent()
            self.trials[idx_to_replace_1] = 0

            if len(abandoned_indices) > 1:
                # Second Scout Bee: replaces the second abandoned solution by mutating the Queen Bee
                idx_to_replace_2 = abandoned_indices[1]

                queen_bee_pos = self.g_best.solution
                # Select a random bee for the mutation formula
                rand_bee_idx = self.generator.choice(list(set(range(0, self.pop_size)) - {idx_to_replace_2}))
                rand_bee_pos = self.pop[rand_bee_idx].solution

                new_pos = queen_bee_pos.copy()
                for j in range(self.problem.n_dims):
                    if self.generator.random() < self.mpr:
                        R_ij = self.generator.uniform(-1, 1)
                        new_pos[j] += R_ij * (rand_bee_pos[j] - queen_bee_pos[j])

                self.pop[idx_to_replace_2] = self.generate_agent(self.correct_solution(new_pos))
                self.trials[idx_to_replace_2] = 0


class ABCX(PaperOptimizer):
    """
    The an improved version of: Artificial Bee Colony (ABCX)

    Links:
        1. https://doi.org/10.1007/s13042-020-01094-7

    Hyper-parameters should be fine-tuned in approximate range to get faster convergence toward the global optimum:
    + n_limits (int): Limit of trials before abandoning a food source.
                      If not set, it defaults to (pop_size/2) * n_dims as suggested by the paper.

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),
    >>>     "minmax": "min",
    >>>     "obj_func": objective_function
    >>> }
    >>>
    >>> model = ABCX(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Hakli, H., & Kiran, M. S. (2020). An improved artificial bee colony algorithm for balancing local
        and global search behaviors in continuous optimization. International Journal of Machine Learning
        and Cybernetics, 11, 2051-2076.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, n_limits: int = None, **kwargs: object) -> None:
        """
        Args:
            epoch (int): Maximum number of iterations, default = 10000
            pop_size (int): Number of population size (onlooker + employed bees), default = 100
            n_limits (int): Limit of trials before abandoning a food source, default=None (auto-calculated)
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.n_limits_user = self.validator.check_int("n_limits", n_limits, [1, 1000]) if n_limits is not None else None
        self.is_parallelizable = False
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = False

        self.NS = int(np.ceil(self.pop_size / 2))
        self.pop_size = self.NS  # In ABC, pop_size is the number of food sources (employed bees)

    def initialize_variables(self):
        """
        Initialize the trials holder for scout bee phase and set the limit parameter.
        """
        # Set n_limits based on the paper's formula if not provided by the user
        if self.n_limits_user is None:
            self.n_limits = self.NS * self.problem.n_dims
        else:
            self.n_limits = self.n_limits_user
        self.trials = np.zeros(self.NS)

    def evolve(self, epoch):
        """
        The main operations (equations) of the algorithm.

        Args:
            epoch (int): The current iteration
        """
        ## Employed Bee Phase - Update one dimension [cite: 193, 295, 310]
        for i in range(self.NS):
            # Randomly select an update rule for employed bees (Eq. 7, 8, or 9) [cite: 290, 308]
            eq_choice = self.generator.integers(0, 3)
            j = self.generator.integers(0, self.problem.n_dims)  # Select one dimension

            # Select random partners, ensuring they are different from the current bee
            k, r = self.generator.choice(list(set(range(self.NS)) - {i}), 2, replace=False)

            pos_new = self.pop[i].solution.copy()
            phi = self.generator.uniform(-1, 1)

            if eq_choice == 0:
                # Eq. 7: V_i^j = X_i^j + Phi * (X_i^j - X_k^j) [cite: 261]
                pos_new[j] = self.pop[i].solution[j] + phi * (self.pop[i].solution[j] - self.pop[k].solution[j])
            elif eq_choice == 1:
                # Eq. 8: V_i^j = X_r^j + Phi * (X_r^j - X_k^j) [cite: 264]
                pos_new[j] = self.pop[r].solution[j] + phi * (self.pop[r].solution[j] - self.pop[k].solution[j])
            else:  # eq_choice == 2
                # Eq. 9: V_i^j = X_r^j + Phi * (X_i^j - X_k^j) [cite: 266]
                pos_new[j] = self.pop[r].solution[j] + phi * (self.pop[i].solution[j] - self.pop[k].solution[j])

            # Greedy Selection
            agent = self.generate_agent(self.correct_solution(pos_new))
            if self.compare_target(agent.target, self.pop[i].target):
                self.pop[i] = agent
                self.trials[i] = 0
            else:
                self.trials[i] += 1

        ## Onlooker Bee Phase - Update three dimensions [cite: 194, 295, 314]
        # Calculate selection probabilities based on fitness [cite: 231]
        fitness_values = np.array([agent.target.fitness for agent in self.pop])
        sum_fits = np.sum(fitness_values)
        probs = fitness_values / sum_fits if sum_fits != 0 else np.ones(self.NS) / self.NS

        for _ in range(self.NS):  # There are NS onlooker bees
            # Select a food source using roulette wheel selection
            i = self.generator.choice(range(self.NS), p=probs)
            pos_new = self.pop[i].solution.copy()

            # Randomly select an update rule for onlooker bees (Eq. 10, 11, or 12) [cite: 290, 313]
            eq_choice = self.generator.integers(0, 3)

            # Choose partners once for this bee
            if eq_choice == 0:  # Eq. 10 needs 2 partners
                partners = self.generator.choice(list(set(range(self.NS)) - {i}), 2, replace=False)
                k, r = partners[0], partners[1]
            elif eq_choice == 1:  # Eq. 11 needs 1 partner
                r = self.generator.choice(list(set(range(self.NS)) - {i}))
            else:  # Eq. 12 needs 4 partners
                partners = self.generator.choice(list(set(range(self.NS)) - {i}), 4, replace=False)
                k, r, u, y = partners[0], partners[1], partners[2], partners[3]

            # Select three unique dimensions to update [cite: 314]
            dims_to_update = self.generator.choice(range(self.problem.n_dims), 3, replace=False)

            for j in dims_to_update:
                phi = self.generator.uniform(-1, 1)
                if eq_choice == 0:
                    # Eq. 10: V_i^j = X_best^j + Phi * (X_k^j - X_r^j) [cite: 267]
                    pos_new[j] = self.g_best.solution[j] + phi * (self.pop[k].solution[j] - self.pop[r].solution[j])
                elif eq_choice == 1:
                    # Eq. 11: Distribution-based update using Box-Muller transform [cite: 268, 269, 270]
                    mu = (self.pop[i].solution[j] + self.pop[r].solution[j]) / 2
                    sigma_val = 0.5 * ((self.pop[i].solution[j] - mu) ** 2 + (self.pop[r].solution[j] - mu) ** 2)
                    sigma = np.sqrt(sigma_val) if sigma_val > 0 else 1e-6

                    R1 = self.generator.random() + 1e-10  # Add epsilon to avoid log(0)
                    R2 = self.generator.random()
                    Z = np.sqrt(-2 * np.log(R1)) * np.cos(2 * np.pi * R2)
                    pos_new[j] = mu + sigma * Z
                else:  # eq_choice == 2
                    # Eq. 12: V_i^j = X_best^j + Phi*(X_k^j - X_r^j) + Phi*(X_u^j - X_y^j) [cite: 271]
                    phi2 = self.generator.uniform(-1, 1)
                    term1 = phi * (self.pop[k].solution[j] - self.pop[r].solution[j])
                    term2 = phi2 * (self.pop[u].solution[j] - self.pop[y].solution[j])
                    pos_new[j] = self.g_best.solution[j] + term1 + term2

            # Greedy selection after updating 3 dimensions
            agent = self.generate_agent(self.correct_solution(pos_new))
            if self.compare_target(agent.target, self.pop[i].target):
                self.pop[i] = agent
                self.trials[i] = 0
            else:
                self.trials[i] += 1

        # Scout bees phase
        # Check the number of trials for each employed bee and abandon the food source if the limit is exceeded
        abandoned = np.where(self.trials >= self.n_limits)[0]
        for idx in abandoned:
            self.pop[idx] = self.generate_agent()
            self.trials[idx] = 0
            break


class DirectedABC(PaperOptimizer):
    """
    An implementation of: Directed Artificial Bee Colony (dABC)

    Links:
        - 1. https://doi.org/10.1016/j.asoc.2014.10.020

    Hyper-parameters:
        - n_limits (int): The number of trials after which a food source is abandoned (default: 50).

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, evolutionary_based
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),
    >>>     "minmax": "min",
    >>>     "obj_func": objective_function
    >>> }
    >>>
    >>> model = evolutionary_based.ABC.DirectedABC(epoch=1000, pop_size=50, n_limits=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Kıran, M. S., & Fındık, O. (2015). A directed artificial bee colony algorithm.
        [cite_start]Applied Soft Computing, 26, 454-462. [cite: 7]
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, n_limits: int = 50, **kwargs: object) -> None:
        """
        Args:
            epoch (int): The maximum number of iterations.
            pop_size (int): The total number of bees in the colony.
            n_limits (int): The limit for abandoning a food source.
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.n_limits = self.validator.check_int("n_limits", n_limits, [1, 1000])
        self.set_parameters(["epoch", "pop_size", "n_limits"])
        self.sort_flag = False

        # [cite_start]The number of employed or onlooker bees is half the colony size[cite: 92].
        # self.pop will be initialized by the framework with this size.
        self.NS = int(self.pop_size / 2)
        self.pop_size = self.NS

    def initialize_variables(self):
        """
        Initializes the trial counters and the direction matrix.
        """
        self.trials = np.zeros(self.NS)
        # [cite_start]Direction matrix: 0=undirected, 1=positive, -1=negative [cite: 173-174]
        # [cite_start]Initially, the search is undirected for all dimensions[cite: 190].
        self.direction = np.zeros((self.NS, self.problem.n_dims))

    def _get_fitness(self, agent):
        """
        [cite_start]Calculates the fitness value for an agent based on its objective function value (Eq. 2)[cite: 102].
        """
        obj_value = agent.target.fitness
        if obj_value >= 0:
            return 1 / (1 + obj_value)
        else:
            return 1 + np.abs(obj_value)

    def evolve(self, epoch):
        """
        The main evolutionary loop of the dABC algorithm.

        Args:
            epoch (int): The current iteration number.
        """
        # --- Employed Bee Phase ---
        for i in range(self.NS):
            # [cite_start]Select a random neighbor solution (k) and a random dimension (j) to perturb [cite: 108, 139]
            k = self.generator.choice(list(set(range(self.NS)) - {i}))
            j = self.generator.integers(0, self.problem.n_dims)

            pos_old = self.pop[i].solution
            pos_new = pos_old.copy()

            # [cite_start]Generate a new candidate solution using the direction matrix (Eq. 6) [cite: 171-175]
            d_ij = self.direction[i, j]
            if d_ij == 0:  # Undirected search (like basic ABC)
                phi = self.generator.uniform(-1, 1)
                pos_new[j] = pos_old[j] + phi * (pos_old[j] - self.pop[k].solution[j])
            elif d_ij == 1:  # Directed search towards the neighbor
                r = self.generator.uniform(0, 1)
                pos_new[j] = pos_old[j] + r * np.abs(pos_old[j] - self.pop[k].solution[j])
            else:  # d_ij == -1, Directed search away from the neighbor
                r = self.generator.uniform(0, 1)
                pos_new[j] = pos_old[j] - r * np.abs(pos_old[j] - self.pop[k].solution[j])

            agent_new = self.generate_agent(self.correct_solution(pos_new))

            # Greedy selection and direction matrix update
            if self.compare_target(agent_new.target, self.pop[i].target, self.problem.minmax):
                self.pop[i] = agent_new
                self.trials[i] = 0
                # [cite_start]Update direction if the move was successful [cite: 191-193]
                self.direction[i, j] = -1 if agent_new.solution[j] > pos_old[j] else 1
            else:
                self.trials[i] += 1
                # [cite_start]Reset direction if the move was unsuccessful [cite: 194]
                self.direction[i, j] = 0

        # --- Onlooker Bee Phase ---
        # [cite_start]Calculate selection probabilities based on fitness (Eq. 4) [cite: 145]
        fitness_values = np.array([self._get_fitness(agent) for agent in self.pop])
        total_fitness = np.sum(fitness_values)
        if total_fitness == 0:
            probs = np.ones(self.NS) / self.NS
        else:
            probs = fitness_values / total_fitness

        for _ in range(self.NS):
            # [cite_start]Onlookers select a food source via roulette wheel selection [cite: 144]
            i = self.generator.choice(range(self.NS), p=probs)

            # The search process is identical to the employed bee phase
            k = self.generator.choice(list(set(range(self.NS)) - {i}))
            j = self.generator.integers(0, self.problem.n_dims)

            pos_old = self.pop[i].solution
            pos_new = pos_old.copy()

            d_ij = self.direction[i, j]
            if d_ij == 0:
                phi = self.generator.uniform(-1, 1)
                pos_new[j] = pos_old[j] + phi * (pos_old[j] - self.pop[k].solution[j])
            elif d_ij == 1:
                r = self.generator.uniform(0, 1)
                pos_new[j] = pos_old[j] + r * np.abs(pos_old[j] - self.pop[k].solution[j])
            else:  # d_ij == -1
                r = self.generator.uniform(0, 1)
                pos_new[j] = pos_old[j] - r * np.abs(pos_old[j] - self.pop[k].solution[j])

            agent_new = self.generate_agent(self.correct_solution(pos_new))

            if self.compare_target(agent_new.target, self.pop[i].target, self.problem.minmax):
                self.pop[i] = agent_new
                self.trials[i] = 0
                self.direction[i, j] = -1 if agent_new.solution[j] > pos_old[j] else 1
            else:
                self.trials[i] += 1
                self.direction[i, j] = 0

        # --- Scout Bee Phase ---
        # [cite_start]An employed bee becomes a scout if its source hasn't improved [cite: 151]
        for i in range(self.NS):
            if self.trials[i] > self.n_limits:
                self.pop[i] = self.generate_agent()  # Replace with a new random source [cite: 152]
                self.trials[i] = 0
                self.direction[i, :] = 0  # Reset direction info for the new scout
                break  # Only one scout can occur per iteration [cite: 153]


class ABCVSS(PaperOptimizer):
    """
    The original version of: Artificial Bee Colony with Variable Search Strategy (ABCVSS)

    Links:
        1. https://doi.org/10.1016/j.ins.2014.12.043

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + n_limits (int): Limit of trials before abandoning a food source, default=50
        + counter_init_val (int): Initial constant value for equation counters, default=10

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, ABC
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),
    >>>     "minmax": "min",
    >>>     "obj_func": objective_function
    >>> }
    >>>
    >>> model = ABCVSS(epoch=1000, pop_size=50, n_limits=750) # limit = D * N as per paper
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Kiran, M. S., Gunduz, M., & Uguz, H. (2015). Artificial bee colony algorithm with variable
        search strategy for continuous optimization. Information Sciences, 300, 140-157.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, n_limits: int = 50, counter_init_val: int = 10,
                 **kwargs: object) -> None:
        """
        Args:
            epoch (int): Maximum number of iterations.
            pop_size (int): Number of population size (employed bees).
            n_limits (int): Limit of trials before abandoning a food source.
            counter_init_val (int): Initial constant value for equation counters.
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.n_limits = self.validator.check_int("n_limits", n_limits, [1, 100000])
        self.counter_init_val = self.validator.check_int("counter_init_val", counter_init_val, [1, 1000])

        self.is_parallelizable = False
        self.set_parameters(["epoch", "pop_size", "n_limits", "counter_init_val"])
        self.sort_flag = False

        self.NS = self.pop_size  # Number of employed bees is equal to pop_size

    def initialize_variables(self):
        """
        Initialize the trials and the equation counters for the search strategies.
        """
        self.trials = np.zeros(self.NS)
        # [cite_start]Initialize counters for each of the 5 search strategies [cite: 184]
        self.equation_counters = np.ones(5) * self.counter_init_val

    def _generate_candidate(self, idx, g_best_solution):
        """
        Selects a search strategy via roulette wheel and generates a candidate solution.

        Args:
            idx (int): The index of the current bee.
            g_best_solution (np.array): The best solution found so far.

        Returns:
            tuple: The new candidate position and the index of the equation used.
        """
        # --- Equation Selector using Roulette-Wheel Mechanism ---
        total_counts = np.sum(self.equation_counters)
        probs = self.equation_counters / total_counts if total_counts > 0 else np.ones(5) / 5
        eq_idx = self.get_index_roulette_wheel_selection(probs)

        pos_new = np.copy(self.pop[idx].solution)
        phi = self.generator.uniform(-1, 1)
        j = self.generator.integers(0, self.problem.n_dims)  # Select one dimension to update

        # --- Five Search Strategies ---
        if eq_idx == 0:
            # [cite_start]Equation (5): Basic ABC update rule [cite: 155]
            k = self.generator.choice(list(set(range(self.NS)) - {idx}))
            pos_new[j] = self.pop[idx].solution[j] + phi * (self.pop[idx].solution[j] - self.pop[k].solution[j])
        elif eq_idx == 1:
            # [cite_start]Equation (6): Crossover-like rule [cite: 156]
            k, r = self.generator.choice(list(set(range(self.NS)) - {idx}), 2, replace=False)
            pos_new[j] = self.pop[idx].solution[j] + phi * (self.pop[k].solution[j] - self.pop[r].solution[j])
        elif eq_idx == 2:
            # [cite_start]Equation (7): Similar to DE/rand/1 [cite: 156]
            k, r = self.generator.choice(list(set(range(self.NS)) - {idx}), 2, replace=False)
            pos_new[j] = self.pop[r].solution[j] + phi * (self.pop[k].solution[j] - self.pop[idx].solution[j])
        elif eq_idx == 3:
            # [cite_start]Equation (8): Global best-guided (GABC) rule [cite: 157]
            k, r = self.generator.choice(list(set(range(self.NS)) - {idx}), 2, replace=False)
            pos_new[j] = g_best_solution[j] + phi * (self.pop[k].solution[j] - self.pop[r].solution[j])
        else:  # eq_idx == 4
            # [cite_start]Equation (9): Mean-guided rule [cite: 158]
            all_solutions = np.array([agent.solution for agent in self.pop])
            x_mean = np.mean(all_solutions, axis=0)
            pos_new[j] = self.pop[idx].solution[j] + phi * (self.pop[idx].solution[j] - x_mean[j])

        return pos_new, eq_idx

    def evolve(self, epoch):
        """
        The main operations of the ABCVSS algorithm.

        Args:
            epoch (int): The current iteration.
        """
        # --- Employed Bee Phase ---
        for idx in range(0, self.NS):
            pos_new, eq_idx = self._generate_candidate(idx, self.g_best.solution)
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_agent(pos_new)

            if self.compare_target(agent.target, self.pop[idx].target, self.problem.minmax):
                self.pop[idx] = agent
                self.trials[idx] = 0
                self.equation_counters[eq_idx] += 1  # Increment counter on successful update [cite: 185]
            else:
                self.trials[idx] += 1

        # --- Onlooker Bee Phase ---
        # [cite_start]Calculate selection probabilities for food sources using fitness values [cite: 119]
        employed_fits = np.array([agent.target.fitness for agent in self.pop])
        if self.problem.minmax == "max":
            total_fit = np.sum(employed_fits)
            probs = employed_fits / total_fit if total_fit > 0 else np.ones(self.NS) / self.NS
        else:  # For minimization problems
            # Convert to maximization problem for probability calculation
            obj_vals = 1.0 / (employed_fits + 1e-10)
            total_fit = np.sum(obj_vals)
            probs = obj_vals / total_fit if total_fit > 0 else np.ones(self.NS) / self.NS

        for _ in range(0, self.NS):  # One onlooker bee for each employed bee
            # Select a food source based on its probability (roulette wheel)
            selected_bee_idx = self.get_index_roulette_wheel_selection(probs)

            pos_new, eq_idx = self._generate_candidate(selected_bee_idx, self.g_best.solution)
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_agent(pos_new)

            if self.compare_target(agent.target, self.pop[selected_bee_idx].target, self.problem.minmax):
                self.pop[selected_bee_idx] = agent
                self.trials[selected_bee_idx] = 0
                self.equation_counters[eq_idx] += 1  # Increment counter on successful update [cite: 185]
            else:
                self.trials[selected_bee_idx] += 1

        # --- Scout Bee Phase ---
        abandoned_indices = np.where(self.trials >= self.n_limits)[0]
        for idx in abandoned_indices:
            self.pop[idx] = self.generate_agent()
            self.trials[idx] = 0
            # [cite_start]A new scout bee can emerge in each iteration [cite: 132]
            break


# paper names
AMDABC = MD3ABC
AMDABC1 = MD31ABC
AMDABC2 = MD32ABC
AMDABC3 = MD33ABC
ABC = OriginalABC2
GBABC = GBestGuidedABC2
DABC = DirectedABC
