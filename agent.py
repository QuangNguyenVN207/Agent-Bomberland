from .random_agent import RandomAgent
from .simple_rule_agent import SimpleRuleAgent
from .smarter_rule_agent import SmarterRuleAgent
from .genius_rule_agent import GeniusRuleAgent
from .box_farmer_agent import BoxFarmerAgent
from .tactical_rule_agent import TacticalRuleAgent

class Agent:
    def __init__(self, agent_id: int):
        # agent_id: 0, 1, 2, or 3
        self.agent_id = agent_id

    # obs = {
    #     "map":     np.ndarray,  # shape (13, 13), dtype int
    #                         # 0=Grass, 1=Wall, 2=Box, 3=Item_Radius, 4=Item_Capacity
    #     "players": np.ndarray,  # shape (4, 5), dtype int8
    #                         # Each row: [row, col, alive, bombs_left, bomb_radius_bonus]
    #     "bombs":   np.ndarray,  # shape (N, 4), dtype int8, N = current number of bombs
    #                         # Each row: [row, col, timer, owner_id]
    # }

    # Notes:

    # alive: 1 = alive, 0 = eliminated
    # bomb_radius_bonus: bonus added to radius (Actual radius = 1 + bonus)
    # bombs_left: number of bombs available to place
    # Agents receive full state.
    
    def act(self, obs: dict) -> int:
        # obs: dict containing 'map', 'players', 'bombs'
        # Returns: int in [0, 5]
        grid = obs["map"]
        my_data = obs["players"][self.agent_id]

        if my_data["is_alive"]:
            # If the agent is alive, decide on an action
            return self.decide_action(grid, my_data)
        else:
            # If the agent is dead, return a default action
            return 0

    def decide_action(self, grid, my_data):
        # Implement your action decision logic here
        return 0