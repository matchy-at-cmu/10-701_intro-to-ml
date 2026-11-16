### DO NOT CHANGE ###
import gymnasium as gym
import torch

"""
Please read: THE ENVIRONMENT INTERFACE

In this homework you will train an agent on the classic CartPole control
problem. A pole is attached by an unactuated joint to a cart that moves along a
frictionless track. The agent pushes the cart left or right, and receives a
reward of +1 for every timestep the pole stays upright.

The environment below is a thin wrapper around the CartPole-v1 environment from
Gymnasium, a library and API for reinforcement learning that was originally
developed by OpenAI and is now open-source. The wrapper exists so that you can
focus on the reinforcement learning algorithm rather than on library details:
it converts observations to PyTorch tensors and simplifies the return values.

This environment exposes two primary methods for interaction:
- reset: Resets the environment and returns the new initial state.
- step:  Applies an action and returns the next state, a reward, and two
         boolean flags that indicate whether the episode terminated or was
         truncated because it reached the maximum number of steps.

Read the implementation below to understand each method's arguments and
their return types.

For more information about the underlying library, see
https://gymnasium.farama.org/environments/classic_control/cart_pole/
"""


class CartPoleEnvironment:
    def __init__(self, max_steps: int = 200, render: bool = False):
        """
        Initialize the environment.

        Parameters:
            max_steps (int) : Maximum number of steps per episode. Once the
                              agent survives this many steps the episode is
                              cut off and `truncated` is set to True.
            render   (bool) : If True, opens a window showing the cart and
                              pole. This is much slower, so only use it to
                              watch an agent you have already trained.
        """
        self.env = gym.make(
            "CartPole-v1",
            max_episode_steps=max_steps,
            render_mode="human" if render else None,
        )
        self.max_steps = max_steps

        # Size of a state vector and the number of available actions.
        # The state is (cart position, cart velocity, pole angle, pole angular
        # velocity); the actions are 0 (push left) and 1 (push right).
        self.state_size = 4
        self.action_size = 2

    def reset(self, seed: int | None = None) -> torch.Tensor:
        """
        Resets the environment to a random initial state.

        Pass a seed only the FIRST time you reset the environment in a program.
        Re-seeding before every episode would make every episode start from the
        same state, which breaks learning.

        Parameters:
            seed (int | None) : Random seed for the environment.

        Returns:
            state (torch.Tensor) : A float32 tensor of shape (4,) holding the
                                   initial state.
        """
        state, info = self.env.reset(seed=seed)
        return torch.tensor(state, dtype=torch.float32)

    def step(self, action: int) -> tuple[torch.Tensor, float, bool, bool]:
        """
        Updates the environment based on the action taken.

        Parameters:
            action (int) : 0 to push the cart left, 1 to push it right.

        Returns:
            (1) state : A float32 torch.Tensor of shape (4,) representing the
                        new state the agent is in after taking its action.
            (2) reward : A float indicating the reward received at this step.
                         CartPole gives +1 for every step survived.
            (3) terminated : A bool indicating whether the episode ended
                             because the agent failed -- the pole fell past
                             12 degrees, or the cart left the track.
            (4) truncated : A bool indicating whether the episode was cut off
                            because it reached `max_steps`. The agent did NOT
                            fail; we simply stopped watching.

        If either `terminated` or `truncated` is True the episode is over and
        you should stop collecting the trajectory. Failing to do so will
        result in undefined behavior.
        """
        if action not in (0, 1):
            raise ValueError(f"Invalid action: {action}. Action must be 0 or 1.")
        state, reward, terminated, truncated, info = self.env.step(action)
        state = torch.tensor(state, dtype=torch.float32)
        return state, float(reward), bool(terminated), bool(truncated)

    def close(self) -> None:
        self.env.close()
