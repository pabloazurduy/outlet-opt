from mdp import MDP
from dataclasses import dataclass, fields
from typing import List, Tuple, Protocol
import numpy as np
from scipy.stats import poisson

@dataclass
class Item: 
    id: int
    stock: int
    low_price_bound: float 
    high_price_bound: float
    days_to_dispose: int

@dataclass 
class Outlet: 
    items: List[Item]

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

@dataclass
class SimOutlet(Protocol):
    outlet: Outlet
    
    @classmethod
    def new_simulation(cls, n_items:int=10, days_avg:int=20, stock_avg:int=100, price_range:Tuple[float, float]=(5000, 80_000)):
        
        items_list:List[Item] = []
        for id in range(n_items):
            item_id = id
            stock = poisson.rvs(stock_avg)
            days_to_dispose  = poisson.rvs(days_avg)
            low_price_bound  = int(np.random.uniform(low=price_range[0], high=price_range[1])// 1000) * 1000
            high_price_bound = int(low_price_bound*1.3//1000)*1000
            
            item = Item(id=item_id, stock=stock, low_price_bound=low_price_bound, high_price_bound=high_price_bound, days_to_dispose=days_to_dispose)
            items_list.append(item)
        
        return cls(outlet=Outlet(items=items_list))

    def as_mdp(self):
        raise NotImplementedError
    

@dataclass
class SimOutletSimple(SimOutlet):
    outlet: Outlet


if __name__ == "__main__":
    sim = SimOutletSimple.new_simulation(n_items=10, days_avg=30, stock_avg=30)
    print(sim.outlet)