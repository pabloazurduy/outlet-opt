from mdp import MDP 
from mdp import ValueIteration, PolicyIteration 

dice_game =  MDP(gamma=0.9, 
                States=['In', 'Out'],
                actions=['stay', 'quit'],
                t_matrix=lambda *key: {('In', 'stay', 'In'):  0.9,
                                ('In', 'stay', 'Out'):  0.1,
                                ('In', 'quit', 'In'):   0,
                                ('In', 'quit', 'Out'):  1,
                                ('Out', 'stay', 'In'):  0,
                                ('Out', 'stay', 'Out'): 0, 
                               }.get((*key,), 0),
                rewards=lambda *key:{('In','stay'):4,
                                     ('In','quit'):10,
                               }.get((*key,), 0)
                
                )

val = dice_game.iterative_policy_evaluation(policy=lambda x:'stay', k_max=1000)
print(val)

#vp = ValueIteration(k_max=10)
#policy_after_one_step = vp.solve(P=dice_game)
pi = PolicyIteration(k_max=100, initial_policy=dice_game.random_policy())
p = pi.solve(P=dice_game)
print([p(s) for s in dice_game.states])
# print("U(s_In) =", policy_after_one_step.U[0])
# print("U(s_Out) =", policy_after_one_step.U[1])