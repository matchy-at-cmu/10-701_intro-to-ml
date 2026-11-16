import random
from collections import OrderedDict

import numpy as np
import torch
import tqdm
from torch import nn, optim

from environment import CartPoleEnvironment
from utils import plot_returns

torch.manual_seed(10701)
random.seed(10701)
np.random.seed(10701)

# Set device based on GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PolicyNetwork(nn.Module):
    def __init__(self, state_size: int, hidden_size: int, action_size: int):
        super().__init__()

        # TODO: Fill in the layers of the policy network.
        # The network maps a state to one score ("logit") per action, using two
        # hidden layers with ReLU activations in between.
        # Note: keep the layer names exactly as given for check.py.
        self.network = nn.Sequential(OrderedDict([
            ('policy_layer_1', ...),
            ('policy_activation_1', ...),
            ('policy_layer_2', ...),
            ('policy_activation_2', ...),
            ('policy_layer_3', ...),
        ]))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
            state (torch.Tensor) : A state of shape (4,), or a batch of states
                                   of shape (T, 4).

        Returns:
            logits (torch.Tensor) : Action logits of shape (2,) for a single
                                    state, or (T, 2) for a batch.
        """
        # TODO: Implement the forward pass
        pass


def get_action(policy: PolicyNetwork, state: torch.Tensor) -> int:
    """
    Sample an action from the policy's action distribution at the given state.

    Parameters:
        policy (PolicyNetwork) : The policy network
        state   (torch.Tensor) : A single state of shape (4,)

    Returns:
        action (int) : The sampled action, as a Python int (0 or 1)
    """
    # TODO: Run the policy forward to get logits, turn them into probabilities, and sample one action from that distribution.
    # Hint: this is action selection, not part of the loss.
    # Hint: the return type must be a Python int, not a tensor.
    pass


def collect_episode(policy: PolicyNetwork, env: CartPoleEnvironment, max_steps: int):
    """
    Run the policy in the environment for one full episode.

    Parameters:
        policy (PolicyNetwork)      : The policy network
        env    (CartPoleEnvironment): The environment
        max_steps (int)             : Maximum number of steps in the episode

    Returns:
        states  (torch.Tensor) : float32 tensor of shape (T, 4)
        actions (torch.Tensor) : int64   tensor of shape (T,)
        rewards (torch.Tensor) : float32 tensor of shape (T,)

    where T is the number of steps the episode lasted. Pay close attention to
    the shapes and dtypes above -- getting them wrong will cause confusing
    errors in the functions that consume this output.
    """
    states, actions, rewards = [], [], []

    # TODO: Reset the environment to get the initial state.
    # IMPORTANT: do NOT pass a seed here. The environment is seeded once, in
    # main(). Re-seeding every episode would make every episode identical.

    for _ in range(max_steps):
        # TODO: Choose an action with the current policy

        # TODO: Take a step in the environment

        # TODO: Store the state, action and reward for this timestep.
        # Careful: you want the state the action was taken FROM.

        # TODO: Move to the new state, and stop the episode if it is over.
        # Hint: read the docstring of env.step -- there are two different
        # flags that both mean "this episode is finished".
        pass

    # TODO: Stack the lists into tensors with the shapes and dtypes given in the docstring above, and return them.
    pass


def shape_rewards(states: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
    """
    Turn the rewards collected from the environment into the rewards used for
    LEARNING. By default it returns them unchanged, so the agent learns from
    CartPole's native reward of +1 per surviving timestep.

    For Empirical Question 2 you will replace the body of this function with a
    reward of your own design. Everything that gets reported -- training
    returns and evaluation returns alike -- always uses the true environment
    reward, so a more generous reward function cannot inflate your score. The
    only thing it changes is what the agent learns from.

    Parameters:
        states  (torch.Tensor) : float32 states  of shape (T, 4), laid out as
                                 (cart position, cart velocity, pole angle,
                                  pole angular velocity). The angle is in
                                 radians; the pole falls at about 0.21 rad
                                 (12 degrees) and the cart leaves the track at
                                 a position of 2.4.
        rewards (torch.Tensor) : float32 environment rewards of shape (T,)

    Returns:
        rewards (torch.Tensor) : float32 rewards of shape (T,) to learn from
    """
    return rewards


def discounted_returns(rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    """
    Compute the discounted reward-to-go at every timestep of an episode:

        A_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ... + gamma^(T-t) * r_T

    Parameters:
        rewards (torch.Tensor) : float32 rewards of shape (T,)
        gamma           (float): Discount factor

    Returns:
        returns (torch.Tensor) : float32 returns of shape (T,)
    """
    # TODO: Compute A_t for every timestep t.
    # Hint: working backwards from the last timestep lets you reuse A_{t+1}
    # when computing A_t, so you never need a nested loop.
    pass


def standardize_returns(returns: torch.Tensor) -> torch.Tensor:
    """
    Standardize the returns of one episode to have mean 0 and unit scale:

        A_hat_t = (A_t - mu) / (sigma + eps)

    where mu and sigma are the mean and standard deviation of A.

    Parameters:
        returns (torch.Tensor) : float32 returns of shape (T,)

    Returns:
        standardized (torch.Tensor) : float32 tensor of shape (T,)
    """
    # TODO: Subtract the mean and divide by the standard deviation.
    # Hint: add a small epsilon (1e-8) to the denominator so that an episode
    # whose returns are all identical does not divide by zero.
    pass


def policy_loss(policy: PolicyNetwork, states: torch.Tensor,
                actions: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
    """
    Compute the REINFORCE loss for a single episode:

        loss = - mean_t [ A_t * ln pi(a_t | s_t) ]

    Parameters:
        policy (PolicyNetwork) : The policy network
        states  (torch.Tensor) : float32 states  of shape (T, 4)
        actions (torch.Tensor) : int64   actions of shape (T,)
        returns (torch.Tensor) : float32 returns of shape (T,)

    Returns:
        loss (torch.Tensor) : a scalar tensor (zero dimensions)
    """
    # TODO: Compute the log probability of each action that was actually taken, then return the negated, return-weighted average.
    pass


def train(policy, optimizer, env, gamma, episodes_per_update, max_steps,
          standardize) -> float:
    """
    Perform one policy update using a batch of episodes.

    Parameters:
        standardize (bool) : If True, standardize each episode's returns before
                             computing the loss.

    Returns:
        mean_return (float) : The mean undiscounted return over the batch
    """
    batch_returns = []

    for _ in range(episodes_per_update):
        # TODO: Collect one episode.

        # TODO: Pass the rewards through shape_rewards before computing the
        # returns, so that Q3 only requires editing that one
        # function. Keep the ORIGINAL env rewards for batch_returns below.

        # TODO: Compute the returns for this episode.

        # TODO: If standardize is True, standardize the returns.

        # TODO: Compute the loss for this episode. Divide it by
        # episodes_per_update so that what we accumulate over the batch is an
        # average rather than a sum.

        # TODO: Backpropagate this episode's loss.
        # Note we do NOT step the optimizer here -- we let gradients accumulate
        # across all the episodes in the batch first.

        # TODO: Record this episode's total (undiscounted) reward
        pass

    # TODO: Now that the whole batch has been accumulated, take one optimizer
    # step and then zero the gradients.
    # IMPORTANT: zero the gradients only AFTER stepping, never between the
    # episodes above -- that accumulation is the whole point of batching.

    return float(np.mean(batch_returns))


def evaluate(policy, env, num_episodes, max_steps) -> float:
    """
    Run the current policy for num_episodes and return the mean undiscounted
    return. Does not update the policy.
    """
    # TODO: Run num_episodes episodes and average their total rewards.
    # Hint: you already have a function that runs one episode.
    # Hint: the policy stays stochastic here -- we sample actions during
    # evaluation exactly as we do during training.
    pass


def run(policy, optimizer, env, gamma, episodes_per_update, max_steps,
        num_updates, eval_every, eval_episodes, standardize):
    """
    Train the policy for num_updates updates, evaluating periodically.

    Returns:
        train_returns (list[float]) : Mean return of each update's batch
        eval_returns  (list[float]) : Mean return at each evaluation
    """
    train_returns, eval_returns = [], []

    for update in tqdm.tqdm(range(num_updates), desc="Updates", leave=False):
        # TODO: Perform one policy update and record the mean training return

        # TODO: Every eval_every updates, evaluate the policy over
        # eval_episodes episodes and record the result
        pass

    return train_returns, eval_returns


def main(
    # Hyperparameters
    max_steps=200,
    gamma=0.99,
    hidden_size=128,
    lr=1e-3,
    episodes_per_update=10,
    num_updates=500,
    eval_every=10,
    eval_episodes=20,
    seed=10701,
    standardize=True,
):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    env = CartPoleEnvironment(max_steps=max_steps)
    env.reset(seed=seed)  # Seed the environment ONCE, here. DON'T DELETE THIS.

    # TODO: Initialize the policy network.
    # Hint: env.state_size and env.action_size give you the input and output sizes.
    policy = None

    # TODO: Initialize the optimizer (use Adam with the given learning rate)
    optimizer = None

    train_returns, eval_returns = run(
        policy, optimizer, env, gamma, episodes_per_update, max_steps,
        num_updates, eval_every, eval_episodes, standardize,
    )

    plot_returns(train_returns, eval_returns, eval_every)
    env.close()

    print(f"Final evaluation return: {eval_returns[-1]:.1f}")
    print(f"Mean of last 5 evaluations: {np.mean(eval_returns[-5:]):.1f}")
    return train_returns, eval_returns


if __name__ == '__main__':
    main()
