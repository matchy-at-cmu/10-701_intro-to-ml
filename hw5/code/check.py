### DO NOT CHANGE ###
"""
Local self-tests for your REINFORCE implementation.

    python check.py

Each check is INDEPENDENT: it supplies its own inputs and compares your output
against ours. A later check passing does NOT mean an earlier one is correct,
so work through the failures from the top down.

Passing every check here is a good sign, but it does not test your training
loop end to end. You still need to run `python reinforce.py` and confirm that
your agent actually learns.
"""
import os
import pickle
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")

CHECK_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "check_data.pkl")


def compare_tensors(actual, expected, label="", atol=1e-5):
    """
    Compare two tensors and raise a ValueError describing how they differ.
    """
    if not torch.is_tensor(actual):
        raise ValueError(
            f"{label}: expected a torch.Tensor, got {type(actual).__name__}. "
            f"Did you forget to return anything?"
        )

    if actual.shape != expected.shape:
        raise ValueError(
            f"{label} shape mismatch! Expected {list(expected.shape)}, "
            f"got {list(actual.shape)}"
        )

    actual = actual.detach()
    if not torch.allclose(actual.float(), expected.float(), atol=atol):
        diff = (actual.float() - expected.float()).abs()
        idx = int(torch.argmax(diff))
        flat_a, flat_e = actual.reshape(-1).float(), expected.reshape(-1).float()
        preview = min(5, flat_a.numel())
        raise ValueError(
            f"{label} values mismatch!\n"
            f"  Maximum absolute difference: {diff.max().item():.6f} (at flat index {idx})\n"
            f"  Expected vs. got at that index: {flat_e[idx]:.6f} vs. {flat_a[idx]:.6f}\n"
            f"  Mean absolute difference: {diff.mean().item():.6f}\n"
            f"  First {preview} values:\n"
            f"    Expected: {[round(v, 4) for v in flat_e[:preview].tolist()]}\n"
            f"    Actual:   {[round(v, 4) for v in flat_a[:preview].tolist()]}\n"
            f"  (allclose with atol={atol} failed)"
        )


def check_dtype(actual, expected_dtype, label):
    if actual.dtype != expected_dtype:
        raise ValueError(
            f"{label} has the wrong dtype! Expected {expected_dtype}, "
            f"got {actual.dtype}. Check the docstring of collect_episode."
        )


class DummyEnv:
    """
    A tiny deterministic stand-in for CartPoleEnvironment, used to check
    collect_episode without any randomness.

    reset()      -> state [0, 0, 0, 0]
    step(action) -> state [t+1, t+1, t+1, t+1], reward 1.0,
                    terminated once t reaches max_steps - 1
    """

    def __init__(self, max_steps=6):
        self.max_steps = max_steps
        self.t = 0
        self.state_size = 4
        self.action_size = 2

    def reset(self, seed=None):
        self.t = 0
        return torch.zeros(4, dtype=torch.float32)

    def step(self, action):
        self.t += 1
        new_state = torch.tensor([float(self.t)] * 4, dtype=torch.float32)
        terminated = self.t == self.max_steps
        return new_state, 1.0, terminated, False


def build_policy(reinforce, data):
    policy = reinforce.PolicyNetwork(
        data["state_size"], data["hidden_size"], data["action_size"])
    policy.load_state_dict(data["state_dict"])
    policy.eval()
    return policy


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_policy_network(reinforce, data):
    policy = build_policy(reinforce, data)
    with torch.no_grad():
        actual = policy(data["forward_input"])
    compare_tensors(actual, data["forward_output"], label="PolicyNetwork.forward")


def check_get_action(reinforce, data):
    policy = build_policy(reinforce, data)
    torch.manual_seed(10701)  # DO NOT CHANGE THIS
    action = reinforce.get_action(policy, data["action_state"])

    if not isinstance(action, int):
        raise ValueError(
            f"get_action must return a Python int, got {type(action).__name__}. "
            f"Hint: .item() converts a one-element tensor to an int."
        )
    if action != data["expected_action"]:
        raise ValueError(
            f"get_action returned {action}, expected {data['expected_action']}.\n"
            f"  Make sure you sample from the softmax of the logits with "
            f"torch.multinomial, rather than taking the argmax, and that you "
            f"do not draw any other random numbers first."
        )


def check_discounted_returns(reinforce, data):
    actual = reinforce.discounted_returns(
        data["returns_rewards"], data["returns_gamma"])
    compare_tensors(actual, data["expected_returns"],
                    label="discounted_returns (constant rewards)")

    actual2 = reinforce.discounted_returns(
        data["returns_rewards_2"], data["returns_gamma_2"])
    compare_tensors(actual2, data["expected_returns_2"],
                    label="discounted_returns (varying rewards)")


def check_standardize_returns(reinforce, data):
    actual = reinforce.standardize_returns(data["standardize_input"])
    compare_tensors(actual, data["expected_standardized"],
                    label="standardize_returns")

    # All-identical returns: std is 0, so this must not produce nan or inf.
    flat = reinforce.standardize_returns(data["standardize_input_flat"])
    if not torch.isfinite(flat).all():
        raise ValueError(
            "standardize_returns produced nan/inf when every return is "
            "identical (standard deviation 0). Add a small epsilon to the "
            "denominator."
        )
    compare_tensors(flat, data["expected_standardized_flat"],
                    label="standardize_returns (constant returns)")


def check_policy_loss(reinforce, data):
    policy = build_policy(reinforce, data)
    actual = reinforce.policy_loss(
        policy, data["loss_states"], data["loss_actions"], data["loss_returns"])

    if not torch.is_tensor(actual):
        raise ValueError(
            f"policy_loss must return a tensor, got {type(actual).__name__}.")
    if actual.dim() != 0:
        raise ValueError(
            f"policy_loss must return a scalar tensor (0 dimensions), but got "
            f"shape {list(actual.shape)}. Did you forget to average over "
            f"timesteps?"
        )
    compare_tensors(actual, data["expected_loss"], label="policy_loss")


def check_collect_episode(reinforce, data):
    # Stub out get_action so that this check is independent of it.
    original_get_action = reinforce.get_action
    reinforce.get_action = lambda policy, state: 1
    try:
        policy = build_policy(reinforce, data)
        env = DummyEnv(max_steps=data["episode_steps"])
        result = reinforce.collect_episode(policy, env, max_steps=100)
    finally:
        reinforce.get_action = original_get_action

    if result is None or len(result) != 3:
        raise ValueError(
            "collect_episode must return exactly three values: "
            "(states, actions, rewards)."
        )
    states, actions, rewards = result

    compare_tensors(states, data["expected_episode_states"],
                    label="collect_episode states")
    compare_tensors(actions, data["expected_episode_actions"],
                    label="collect_episode actions")
    compare_tensors(rewards, data["expected_episode_rewards"],
                    label="collect_episode rewards")

    check_dtype(states, torch.float32, "collect_episode states")
    check_dtype(actions, torch.int64, "collect_episode actions")
    check_dtype(rewards, torch.float32, "collect_episode rewards")


CHECKS = [
    ("PolicyNetwork", check_policy_network),
    ("get_action", check_get_action),
    ("discounted_returns", check_discounted_returns),
    ("standardize_returns", check_standardize_returns),
    ("policy_loss", check_policy_loss),
    ("collect_episode", check_collect_episode),
]


def main():
    if not os.path.exists(CHECK_DATA_PATH):
        raise FileNotFoundError(
            f"Could not find {CHECK_DATA_PATH}. Run check.py from the same "
            f"directory as check_data.pkl."
        )
    with open(CHECK_DATA_PATH, "rb") as f:
        data = pickle.load(f)

    import reinforce

    print("\n===== REINFORCE implementation checks =====\n")
    print("Checks are INDEPENDENT. A later check passing does not mean an")
    print("earlier component is correct -- fix failures from the top down.\n")

    passed = 0
    for name, check in CHECKS:
        print(f"Checking {name}...")
        try:
            check(reinforce, data)
            passed += 1
            print(f"  {name}: PASSED\n")
        except Exception as e:
            print(f"  {name}: FAILED")
            print(f"  {e}\n")

    print("===== Summary =====")
    print(f"Passed: {passed}/{len(CHECKS)} checks")
    if passed == len(CHECKS):
        print("All checks passed. Now run `python reinforce.py` and confirm "
              "that your agent actually learns.")
    else:
        print(f"{len(CHECKS) - passed} check(s) failed. Read the messages above.")
    print()


if __name__ == '__main__':
    main()
