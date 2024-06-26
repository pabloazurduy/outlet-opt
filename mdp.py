"""
code extracted from 
https://github.com/griffinbholt/decisionmaking-code-py/blob/9291865f9c9b0f2863989fc42e89eabafec86a56/src/ch07.py
this is the python translation from the julia code present on the book "algorithms for decision making", 
the chapter 07 presents most of the algorithms to solve MDP's. The code here is adapted for readability 
"""

"""Chapter 7: Exact Solution Methods"""

import cvxpy as cp
import numpy as np
import random
import warnings

from abc import ABC, abstractmethod
from typing import Any, Callable, List

warnings.simplefilter(action='ignore', category=FutureWarning)


class MDP():
    """
    Data structure for a Markov Decision Process. In mathematical terms,
    MDPs are sometimes defined in terms of a tuple consisting of the various
    components of the MDP, written (S, A, T, R, gamma):

    gamma: discount factor
    states: state space
    actions_space: action space
    t_prob: transition function
    rewards: reward function
    r_sample: sample transition and reward. We will us `TR` later to sample the next
        state and reward given the current state and action: s_prime, r = TR(s, a)
        r_sample is a sample from q function
    """
    def __init__(self,
                 gamma: float, 
                 states: list[Any],
                 t_prob:   Callable[[Any, Any, Any], float] | np.ndarray,
                 rewards:  Callable[[Any, Any], float] | np.ndarray,
                 actions_space: list[Any] = None,
                 actions:  Callable[[Any], List[Any]] = None,
                 r_sample: Callable[[Any, Any], tuple[Any, float]] = None):
        self.gamma = gamma     # discount factor
        self.states = states   # state space
        
        self.actions = actions if actions is not None else lambda s: actions_space 
        self.actions_space = actions_space if actions_space is not None else list(set([action for s in states for action in actions(s)]))
        

        # reward function R(s, a)
        if type(rewards) == np.ndarray:
            self.rewards = lambda s, a: rewards[s, a]
        else:
            self.rewards = rewards

        # transition function T(s, a, s')
        # sample next state and reward given current state and action: s', r = TR(s, a)
        if type(t_prob) == np.ndarray:
            self.t_prob   = lambda s, a, s_prime: t_prob[s, a, s_prime]
            self.q_sample = lambda s, a: (np.random.choice(len(self.states), p=t_prob[s, a]), self.rewards(s, a)) if not np.all(t_prob[s, a] == 0) else (np.random.choice(len(self.states)), self.rewards(s, a))
        else:
            self.t_prob = t_prob
            self.q_sample = r_sample

    def lookahead(self, V: Callable[[Any], float] | np.ndarray, s: Any, a: Any) -> float:
        if callable(V):
            return self.rewards(s, a) + self.gamma * np.sum([self.t_prob(s, a, s_prime) * V(s_prime) for s_prime in self.states])
        return self.rewards(s, a) + self.gamma * np.sum([self.t_prob(s, a, s_prime) * V[i] for i, s_prime in enumerate(self.states)])

    def iterative_policy_evaluation(self, policy: Callable[[Any], Any], k_max: int) -> np.ndarray:
        V = np.zeros(len(self.states)) # V is the expected utility of the policy \pi on the state S. also known as $V_{\pi}(s)$
        for _ in range(k_max):
            V = np.array([self.lookahead(V, s, policy(s)) for s in self.states])
        return V

    def policy_evaluation(self, policy: Callable[[Any], Any]) -> np.ndarray:
        R_prime = np.array([self.rewards(s, policy(s)) for s in self.states])
        T_prime = np.array([[self.t_prob(s, policy(s), s_prime) for s_prime in self.states] for s in self.states])
        I = np.eye(len(self.states))
        return np.linalg.solve(I - self.gamma * T_prime, R_prime)

    def greedy(self, V: Callable[[Any], float] | np.ndarray, s: Any) -> tuple[float, Any]:
        expected_rewards = [self.lookahead(V, s, a) for a in self.actions(s)]
        idx = np.argmax(expected_rewards)
        return self.actions(s)[idx], expected_rewards[idx]

    def backup(self, V: Callable[[Any], float] | np.ndarray, s: Any) -> float:
        return np.max([self.lookahead(V, s, a) for a in self.actions(s)])

    def randstep(self, s: Any, a: Any) -> tuple[Any, float]:
        return self.q_sample(s, a)

    def simulate(self, s: Any, policy: Callable[[Any], Any], d: int) -> list[tuple[Any, Any, float]]:  # TODO - Create test
        trajectory = []
        for _ in range(d):
            a = policy(s)
            s_prime, r = self.q_sample(s, a)
            trajectory.append((s, a, r))
            s = s_prime
        return trajectory
    
    def random_policy(self):
        return lambda s, actions=self.actions: random.choices(actions(s))[0]


class ValueFunctionPolicy():
    """gets the policy given a value function
    """
    def __init__(self, problem: MDP, V: Callable[[Any], float] | np.ndarray):
        self.problem = problem  # problem
        self.V = V        # expected utility function (given a policy)

    def __call__(self, s: Any) -> Any:
        return self.problem.greedy(self.V, s)[0]


class MDPSolutionMethod(ABC):
    pass


class OfflinePlanningMethod(MDPSolutionMethod):
    @abstractmethod
    def solve(self, problem: MDP) -> Callable[[Any], Any]:
        pass


class ExactSolutionMethod(OfflinePlanningMethod):
    pass


class PolicyIteration(ExactSolutionMethod):
    """
    Policy iteration (algorithm 7.6) is one way to compute an optimal policy. 
    It involves iterating between policy evaluation (section 7.2) and policy 
    improvement through a greedy policy (algorithm 7.5). Policy iteration is 
    guaranteed to converge given any initial policy. It converges in a finite 
    number of iterations because there are finitely many policies and every 
    iteration improves the policy if it can be improved. Although the number 
    of possible policies is exponential in the number of states, policy iteration 
    often converges quickly.

    Args:
        ExactSolutionMethod (_type_): _description_
    """
    def __init__(self, initial_policy: Callable[[Any], Any], k_max: int):
        self.initial_policy = initial_policy
        self.k_max = k_max

    def solve(self, problem: MDP) -> Callable[[Any], Any]:
        policy = self.initial_policy
        for _ in range(self.k_max):
            V = problem.policy_evaluation(policy)
            policy_prime = ValueFunctionPolicy(problem, V)
            if all([policy(s) == policy_prime(s) for s in problem.states]):
                break
            print([policy_prime(s) for s in problem.states], f'{V = }')
            policy = policy_prime
        return policy


class ValueIteration(ExactSolutionMethod):
    """Bellman's algorithm
    Value iteration is an alternative to policy iteration that is often used 
    because of its simplicity. Unlike policy improvement, value iteration 
    updates the value function directly. 

    Args:
        ExactSolutionMethod (_type_): _description_
    """    
    def __init__(self, k_max: int):
        self.k_max = k_max

    def solve(self, problem: MDP) -> Callable[[Any], Any]:
        V = np.zeros(len(problem.states))
        for _ in range(self.k_max):
            V = np.array([problem.backup(V, s) for s in problem.states])
            print(V)
        return ValueFunctionPolicy(problem, V)


class GaussSeidelValueIteration(ExactSolutionMethod):
    """    
    In asynchronous value iteration, only a subset of the states are updated with 
    each iteration. Asynchronous value iteration is still guaranteed to converge on 
    the optimal value function, provided that each state is updated an infinite number of times.
    
    The computational savings lies in not having to construct a second value function
    in memory with each iteration. Gauss-Seidel value iteration can converge more
    quickly than standard value iteration, depending on the ordering chosen.1

    Args:
        ExactSolutionMethod (_type_): _description_
    """

    def __init__(self, k_max: int):
        self.k_max = k_max

    def solve(self, problem: MDP) -> Callable[[Any], Any]:
        V = np.zeros(len(problem.states))
        for _ in range(self.k_max):
            for i, s in enumerate(problem.states):
                V[i] = problem.backup(V, s)
        return ValueFunctionPolicy(problem, V)


class LinearProgramFormulation(ExactSolutionMethod):



    def solve(self, problem: MDP) -> Callable[[Any], Any]:
        S, A, R, T = self.numpyform(problem)
        V = cp.Variable(len(S))
        objective = cp.Minimize(cp.sum(V))
        constraints = [V[s] >= R[s, a] + problem.gamma * (T[s, a] @ V) for s in S for a in A]
        problem = cp.Problem(objective, constraints)
        problem.solve()
        return ValueFunctionPolicy(problem, V.value)

    @staticmethod
    def numpyform(problem: MDP) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        S_prime = np.arange(len(problem.states))
        A_prime = np.arange(len(problem.actions_space))
        R_prime = np.array([[problem.rewards(s, a) for a in problem.actions_space] for s in problem.states])
        T_prime = np.array([[[problem.t_prob(s, a, s_prime) for s_prime in S_prime] for a in problem.actions_space] for s in problem.states])
        return S_prime, A_prime, R_prime, T_prime


class LinearQuadraticProblem():
    def __init__(self, Ts: np.ndarray, Ta: np.ndarray, Rs: np.ndarray, Ra: np.ndarray, h_max: int):
        assert np.all(np.linalg.eigvals(Rs) <= 0), "Rs must be NSD"  # TODO - not most numerically safe method
        assert np.all(np.linalg.eigvals(Ra) < 0), "Ra must be ND"  # TODO - not most numerically safe method
        self.Ts = Ts        # transition matrix with respect to state
        self.Ta = Ta        # transition matrix with respect to action
        self.Rs = Rs        # reward matrix with respect to state (negative semidefinite)
        self.Ra = Ra        # reward matrix with respect to action (negative definite)
        self.h_max = h_max  # horizon

    def solve(self) -> list[Callable[[np.ndarray], np.ndarray]]:
        Ts, Ta, Rs, Ra = self.Ts, self.Ta, self.Rs, self.Ra
        V = np.zeros(self.Rs.shape)
        policies = [lambda s: np.zeros(self.Ta.shape[1])]
        for _ in range(1, self.h_max):
            V = (Ts.T @ (V - (V @ Ta @ (np.linalg.inv(Ta.T @ V @ Ta + Ra) @ (Ta.T @ V)))) @ Ts) + Rs
            L = -np.linalg.inv(Ta.T @ V @ Ta + Ra) @ Ta.T @ V @ Ts
            policies.append(lambda s, L=L: L @ s)
        return policies