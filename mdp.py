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
from typing import Any, Callable

warnings.simplefilter(action='ignore', category=FutureWarning)


class MDP():
    """
    Data structure for a Markov Decision Process. In mathematical terms,
    MDPs are sometimes defined in terms of a tuple consisting of the various
    components of the MDP, written (S, A, T, R, gamma):

    gamma: discount factor
    S: state space
    A: action space
    T: transition function
    R: reward function
    TR: sample transition and reward. We will us `TR` later to sample the next
        state and reward given the current state and action: s_prime, r = TR(s, a)
    """
    def __init__(self,
                 gamma: float, 
                 S: list[Any],
                 A: list[Any],
                 T: Callable[[Any, Any, Any], float] | np.ndarray,
                 R: Callable[[Any, Any], float] | np.ndarray,
                 TR: Callable[[Any, Any], tuple[Any, float]] = None):
        self.gamma = gamma  # discount factor
        self.S = S          # state space
        self.A = A          # action space

        # reward function R(s, a)
        if type(R) == np.ndarray:
            self.R = lambda s, a: R[s, a]
        else:
            self.R = R

        # transition function T(s, a, s')
        # sample next state and reward given current state and action: s', r = TR(s, a)
        if type(T) == np.ndarray:
            self.T = lambda s, a, s_prime: T[s, a, s_prime]
            self.TR = lambda s, a: (np.random.choice(len(self.S), p=T[s, a]), self.R(s, a)) if not np.all(T[s, a] == 0) else (np.random.choice(len(self.S)), self.R(s, a))
        else:
            self.T = T
            self.TR = TR

    def lookahead(self, U: Callable[[Any], float] | np.ndarray, s: Any, a: Any) -> float:
        if callable(U):
            return self.R(s, a) + self.gamma * np.sum([self.T(s, a, s_prime) * U(s_prime) for s_prime in self.S])
        return self.R(s, a) + self.gamma * np.sum([self.T(s, a, s_prime) * U[i] for i, s_prime in enumerate(self.S)])

    def iterative_policy_evaluation(self, policy: Callable[[Any], Any], k_max: int) -> np.ndarray:
        U = np.zeros(len(self.S))
        for _ in range(k_max):
            U = np.array([self.lookahead(U, s, policy(s)) for s in self.S])
        return U

    def policy_evaluation(self, policy: Callable[[Any], Any]) -> np.ndarray:
        R_prime = np.array([self.R(s, policy(s)) for s in self.S])
        T_prime = np.array([[self.T(s, policy(s), s_prime) for s_prime in self.S] for s in self.S])
        I = np.eye(len(self.S))
        return np.linalg.solve(I - self.gamma * T_prime, R_prime)

    def greedy(self, U: Callable[[Any], float] | np.ndarray, s: Any) -> tuple[float, Any]:
        expected_rewards = [self.lookahead(U, s, a) for a in self.A]
        idx = np.argmax(expected_rewards)
        return self.A[idx], expected_rewards[idx]

    def backup(self, U: Callable[[Any], float] | np.ndarray, s: Any) -> float:
        return np.max([self.lookahead(U, s, a) for a in self.A])

    def randstep(self, s: Any, a: Any) -> tuple[Any, float]:
        return self.TR(s, a)

    def simulate(self, s: Any, policy: Callable[[Any], Any], d: int) -> list[tuple[Any, Any, float]]:  # TODO - Create test
        trajectory = []
        for _ in range(d):
            a = policy(s)
            s_prime, r = self.TR(s, a)
            trajectory.append((s, a, r))
            s = s_prime
        return trajectory
    
    def random_policy(self):
        return lambda s, A=self.A: random.choices(A)[0]


class ValueFunctionPolicy():
    def __init__(self, P: MDP, U: Callable[[Any], float] | np.ndarray):
        self.P = P  # problem
        self.U = U  # utility function

    def __call__(self, s: Any) -> Any:
        return self.P.greedy(self.U, s)[0]


class MDPSolutionMethod(ABC):
    pass


class OfflinePlanningMethod(MDPSolutionMethod):
    @abstractmethod
    def solve(self, P: MDP) -> Callable[[Any], Any]:
        pass


class ExactSolutionMethod(OfflinePlanningMethod):
    pass


class PolicyIteration(ExactSolutionMethod):
    def __init__(self, initial_policy: Callable[[Any], Any], k_max: int):
        self.initial_policy = initial_policy
        self.k_max = k_max

    def solve(self, P: MDP) -> Callable[[Any], Any]:
        policy = self.initial_policy
        for _ in range(self.k_max):
            U = P.policy_evaluation(policy)
            policy_prime = ValueFunctionPolicy(P, U)
            if all([policy(s) == policy_prime(s) for s in P.S]):
                break
            policy = policy_prime
        return policy


class ValueIteration(ExactSolutionMethod):
    def __init__(self, k_max: int):
        self.k_max = k_max

    def solve(self, P: MDP) -> Callable[[Any], Any]:
        U = np.zeros(len(P.S))
        for _ in range(self.k_max):
            U = np.array([P.backup(U, s) for s in P.S])
            print(U)
        return ValueFunctionPolicy(P, U)


class GaussSeidelValueIteration(ExactSolutionMethod):
    def __init__(self, k_max: int):
        self.k_max = k_max

    def solve(self, P: MDP) -> Callable[[Any], Any]:
        U = np.zeros(len(P.S))
        for _ in range(self.k_max):
            for i, s in enumerate(P.S):
                U[i] = P.backup(U, s)
        return ValueFunctionPolicy(P, U)


class LinearProgramFormulation(ExactSolutionMethod):
    def solve(self, P: MDP) -> Callable[[Any], Any]:
        S, A, R, T = self.numpyform(P)
        U = cp.Variable(len(S))
        objective = cp.Minimize(cp.sum(U))
        constraints = [U[s] >= R[s, a] + P.gamma * (T[s, a] @ U) for s in S for a in A]
        problem = cp.Problem(objective, constraints)
        problem.solve()
        return ValueFunctionPolicy(P, U.value)

    @staticmethod
    def numpyform(P: MDP) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        S_prime = np.arange(len(P.S))
        A_prime = np.arange(len(P.A))
        R_prime = np.array([[P.R(s, a) for a in P.A] for s in P.S])
        T_prime = np.array([[[P.T(s, a, s_prime) for s_prime in S_prime] for a in P.A] for s in P.S])
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