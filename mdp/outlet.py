from typing import List, Tuple, Protocol, Self, Any, Dict, Union
from enum import Enum
from dataclasses import dataclass, fields
import itertools as it 
from collections import namedtuple

import numpy as np
from scipy.stats import poisson
from scipy.special import expit

from mdp import MDP


@dataclass(repr=True)
class Item: 
    id: int
    stock: int
    low_price_bound: float 
    high_price_bound: float
    days_to_dispose: int
    tick_step:float

@dataclass 
class SimItem(Item):
    elasticity_beta:float
    weekend_beta:float
    bias_beta:float

    def conv_prob(self, dow:int, price:float) -> float:
        return expit(self.bias_beta + 
                     self.weekend_beta*(((dow % 7) == 6) + ((dow % 7) == 0)) + 
                     self.elasticity_beta*price/self.high_price_bound )
    
    def state_prob(self, s:'State', s_p:'State', a:Any)-> float:
        pass

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
        rows = "\n".join(" | ".join(f"{str(getattr(item, name)):<{column_widths[name]}}" for name in attribute_names) for item in sorted(self.items, key=lambda x: getattr(x, attribute_names[0])))

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
    def new_simulation(cls, n_items:int=10, days_avg:int=20, stock_avg:int=100, price_range:Tuple[float, float]=(5000, 80_000), 
                       tick_step:float=1000) -> Self:
        
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
                    )
            items_list.append(item)
        return cls(sim_outlet=Outlet(items=items_list))

    def to_mdp(self, mdp_type:MDPVersion = MDPVersion.INDIVIDUAL)-> List[MDP]:   
        if mdp_type == MDPVersion.INDIVIDUAL:
            State = namedtuple('State', ['price', 'stock', 'days'])                

            for item in self.sim_outlet:
                gamma = 0.8
                price_arr = np.arange(item.low_price_bound, item.high_price_bound, item.tick_step)[::-1] 
                stock_arr = np.arange(0,item.stock,1 )[::-1] 
                days_arr = np.arange(0,item.days_to_dispose,1 )[::-1] 
                states = [State(p,s,d) for (p,s,d) in it.product(price_arr,stock_arr,days_arr)]
                actions = lambda s: list(filter(lambda p: p <= s.price, price_arr))

                def item_t_prob(s:State, sp:State, a:Any) -> float:

                    return None

                t_prob = item_t_prob
                rewards = lambda s, s_p: s.price * (s.stock - s_p.stock)


        mdp_models = MDP()
        return mdp_model 
    
if __name__ == "__main__":
    sim = SimOutlet.new_simulation(n_items=10, days_avg=30, stock_avg=30)
    mpds = sim.to_mdp(mdp_type=MDPVersion.INDIVIDUAL)
    print(sim.outlet)
