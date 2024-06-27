# Outlet-Opt
A set of algorithms to optimize outlet sales. This code is based on ["algorithms for decision making"](https://algorithmsbook.com/), and uses the code transpile from https://github.com/griffinbholt/decisionmaking-code-py/
used under fair use 

## Overview
The aim of this repository is to solve the problem of optimum price policy for a set of items that must be sold on a determinate amount of days. The data is generated based on a `SimOutlet` class that is very straightforward to use.

```python
@dataclass(repr=True, frozen=True)
class Item: 
    id: int
    stock: int
    low_price_bound: float 
    high_price_bound: float
    days_to_dispose: int
    tick_step:float

@dataclass(frozen=True)
class SimItem(Item):
    # a simulated version of an item, it has some simulated parameters useful to understand some purchase probabilities 
    elasticity_beta:float
    weekend_beta:float
    bias_beta:float
    purchase_prob_cap:float
```

The module `outlet` contains the classes for the outlet MDP definition.
The module `mdp` contains the classes that define an MDP and a few optimization algorithms.

In this first version, we assume that the outlet problem is known - we know the transition probabilities -. However, we could extend this problem to non-MDP setups, such as SARSA or other RL algorithms; that's future work.

## Getting Started

### Prerequisites

Ensure you have Python installed on your system. This library requires Python 3.11 or newer. 
You can install these dependencies using pip:

```bash
pip install -r requirements.txt
```

### Installation

Clone the repository to your local machine:

```bash
git clone https://github.com/pabloazurduy/outlet-opt.git
cd outlet-opt
```

### Usage

After installing the necessary dependencies, you can start using Outlet-Opt by importing it into your Python script. Here's a basic example to get you started:

```python
# Initialize a simulation for outlet sales optimization
sim = SimOutlet.new_simulation(n_items=15, days_avg=20, stock_avg=3, purchase_prob_cap_bounds=(0.15,0.2))

# Convert the simulation data into a set of Markov Decision Processes (MDPs)
mdps = sim.to_mdp(mdp_type=MDPVersion.INDIVIDUAL)

# Initialize lists to store policy values, optimal policy values, and reward gaps
pvals: List[float] = []
opt_vals: List[float] = []
gaps: List[float] = []

# Iterate over each MDP representing an individual item
for item_mdp in mdps:
    # Retrieve the action space for the current MDP
    actions = item_mdp.actions_space
    
    # Evaluate the policy using a simple strategy (choosing the minimum action)
    pval = item_mdp.policy_evaluation(policy=lambda x: min(actions))
    print(f'dumb policy value {pval[0]}')  # Print the value of the simple policy
        pi = PolicyIteration(k_max=100, initial_policy=item_mdp.random_policy())
    # Perform policy iteration to find an optimal policy
    pi = PolicyIteration(k_max=100, initial_policy=item_mdp.random_policy())
    opt_policy = pi.solve(problem=item_mdp)
    
    # Print the actions recommended by the optimal policy for each state
    print([opt_policy(s) for s in item_mdp.states])
    
    # Re-evaluate the policy using the newly found optimal policy
    opt_pval = item_mdp.policy_evaluation(policy=opt_policy)
    print(f'opt policy value {opt_pval[0]}')  # Print the value of the optimal policy
    
    # Calculate the gap between the optimal and simple policy values
    rewards_gap = (opt_pval[0] - pval[0]) / pval[0]
    pvals.append(pval[0])  # Store the simple policy value
    opt_vals.append(opt_pval[0])  # Store the optimal policy value
    gaps.append(rewards_gap)  # Store the gap

    print(f'gap val {rewards_gap:+.2%}')  # Print the percentage gap
    print(rewards_gap)
```

> 0.08359726727985081

