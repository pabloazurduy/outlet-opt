from typing import List, Tuple, Protocol, Self, Any, Dict, Union
from enum import Enum
from dataclasses import dataclass, fields
import itertools as it 
from collections import namedtuple
from functools import cache

import numpy as np
from scipy.stats import poisson
from scipy.special import expit

from mdp import MDP


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
    elasticity_beta:float
    weekend_beta:float
    bias_beta:float
    purchase_prob_cap:float

    def sale_prob(self, dow:int, price:float, units:int) -> float:
        sale_prob=expit(self.bias_beta + 
                     self.weekend_beta*(((dow % 7) == 6) + ((dow % 7) == 0)) + 
                     self.elasticity_beta*price/self.high_price_bound )*self.purchase_prob_cap
        return sale_prob**units 
    
    def individual_t_prob(self, s:'State', a:Any, s_p:'State')-> float:
        """ 
        Probability of transition from one state to the next one
        a:new price, if its the same as s or not the probability of any other price is 0 
        because we always move to the new a price 
        the only one that we need calculate is for the [s.stock,0] states -on the next day-, 
        all the rest is 0 
        Args:
            s (State): _description_
            s_p (State): _description_
            a (Any): _description_

        Raises:
            NotImplementedError: _description_

        Returns:
            float: _description_
        """
        print(f'fval for {s,a,s_p}')
        if (s_p.price != a or  # price is equal to the one defined
            s_p.days != s.days-1 or # not next day
            s_p.stock> s.stock or # not reduced inventory
            s.price < a # we can't increase price
            ):
            return 0   
        if s.stock-s_p.stock>0:
            return self.sale_prob(dow=s_p.days, price=a, units=s.stock-s_p.stock)
        else:
            return 1-sum([self.sale_prob(dow=s_p.days, price=a, units=s.stock-i) for i in range(s.stock)])

@dataclass 
class Outlet: 
    items: List[Item|SimItem]

    def __repr__(self):
        if not self.items:
            return "The inventory is empty."

        # Get attribute names and max widths
        attribute_names = [field.name for field in fields(self.items[0])]
        column_widths = {name: max(len(name), max(len(str(getattr(item, name))) for item in self.items)) for name in attribute_names}

        # Create header and rows
        header = " | ".join(f"{name:<{column_widths[name]}}" for name in attribute_names)
        divider = "-+-".join("-" * column_widths[name] for name in attribute_names)
        rows = "\n".join(" | ".join(f"{str(getattr(item, name)):<{column_widths[name]}}" for name in attribute_names) 
                            for item in sorted(self.items, key=lambda x: getattr(x, attribute_names[0])))

        return f"{header}\n{divider}\n{rows}"

# MDP classes 

class MDPVersion(Enum):
    INDIVIDUAL = 1
    COMPLEX = 2
    ADVANCED = 3

@dataclass
class SimOutlet:
    sim_outlet: List[SimItem]
    
    @classmethod
    def new_simulation(cls, n_items:int=10, days_avg:int=10, stock_avg:int=30, price_range:Tuple[float, float]=(5000, 80_000), 
                       tick_step:float=5000) -> Self:
        
        items_list:List[SimItem] = []
        for id in range(n_items):
            item_id = id
            stock = poisson.rvs(stock_avg)
            days_to_dispose  = poisson.rvs(days_avg)
            low_price_bound  = int(np.random.uniform(low=price_range[0], high=price_range[1])// 1000) * 1000
            high_price_bound = int(low_price_bound*1.3//1000)*1000
            tick_step = tick_step

            
            item = SimItem(id=item_id, 
                           stock=stock, 
                           low_price_bound=low_price_bound, 
                           high_price_bound=high_price_bound, 
                           days_to_dispose=days_to_dispose,
                           tick_step=tick_step,
                           elasticity_beta=np.random.uniform(low=-1, high=-0.2),
                           weekend_beta = np.random.uniform(low=0.5, high=1.5),
                           bias_beta=np.random.uniform(low=-3, high=-2),
                           purchase_prob_cap=np.random.uniform(low=0.15, high=0.3)
                    )
            items_list.append(item)
        return cls(sim_outlet=Outlet(items=items_list))

    def to_mdp(self, mdp_type:MDPVersion = MDPVersion.INDIVIDUAL)-> List[MDP]:   
        if mdp_type == MDPVersion.INDIVIDUAL:
            State = namedtuple('State', ['price', 'stock', 'days'])                
            models:List[MDP]=[]
            for item in self.sim_outlet.items:
                gamma = 0.8
                price_arr = np.arange(item.low_price_bound, item.high_price_bound, item.tick_step)[::-1] 
                stock_arr = np.arange(0,item.stock,1 )[::-1] 
                days_arr = np.arange(0,item.days_to_dispose,1 )[::-1] 
                states = [State(p,s,d) for (p,s,d) in it.product(price_arr,stock_arr,days_arr)]
                actions = lambda s: list(filter(lambda p: p <= s.price, price_arr))
                t_prob =  lambda s, a, sp: item.individual_t_prob(s,a, sp)
                rewards = lambda s, a: sum([a * (s.stock - sp.stock)*t_prob(s ,a,sp) for sp in states if sp.price==a and sp.stock <= s.stock]) # TODO explore feasible state 
                mdp = MDP(gamma=gamma, states=states, actions=actions, t_prob=t_prob, rewards=rewards)
                models.append(mdp)

        return models 
    
if __name__ == "__main__":

    sim = SimOutlet.new_simulation(n_items=10, days_avg=10, stock_avg=10)
    mdps = sim.to_mdp(mdp_type=MDPVersion.INDIVIDUAL)
    actions = mdps[0].actions_space
    val = mdps[0].iterative_policy_evaluation(policy=lambda x:min(actions), k_max=100)
    print(val)
