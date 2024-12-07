from cmad.simulation.agents.cmad_agent import CmadAgent
from cmad.simulation.data.simulator import Simulator
from cmad.agent.action import AbstractAction, AgentAction


class RLAgent(CmadAgent):

    """
    Reinforcement Learning Agent to control the ego via input actions
    """

    def setup(self, actor_config: dict):
        """
        Setup the agent parameters
        """
        super().setup(actor_config)
        self.actor = Simulator.get_actor_by_id(self.actor_config["id"])
        self.action_handler: AgentAction = actor_config.pop("action_handler")
        self.abstract_action: AbstractAction = None
