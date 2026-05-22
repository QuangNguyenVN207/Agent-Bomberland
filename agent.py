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

    def act(self, obs: dict) -> int:
        # obs: dict containing 'map', 'players', 'bombs'
        # Returns: int in [0, 5]
        my_data = obs["players"][self.agent_id]

        return 0