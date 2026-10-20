from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import torch
from torchvision import datasets, transforms

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
NN_IMPLEMENTATION_DIRECTORY = (
    REPOSITORY_ROOT / "hw3" / "code" / "nn_implementation_code"
)
CHECK_DIRECTORY = REPOSITORY_ROOT / "hw3" / "code" / "check"
DATA_DIRECTORY = Path(__file__).resolve().parent / "data"


def load_base_experiment() -> ModuleType:
    """Import the homework script with its sibling modules on the import path."""
    sys.path.insert(0, str(NN_IMPLEMENTATION_DIRECTORY))
    try:
        return importlib.import_module("base_experiment")
    finally:
        sys.path.remove(str(NN_IMPLEMENTATION_DIRECTORY))


BASE_EXPERIMENT = load_base_experiment()


def load_first_training_sample() -> tuple[torch.Tensor, torch.Tensor]:
    dataset = datasets.FashionMNIST(
        root=str(DATA_DIRECTORY),
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )
    image, label = dataset[0]
    return image.unsqueeze(0), torch.tensor([label], dtype=torch.long)


def test_first_fashion_mnist_update_matches_provided_regression_data() -> None:
    inputs, targets = load_first_training_sample()
    model = BASE_EXPERIMENT.FashionMNISTModel()
    initial_weights = torch.load(
        NN_IMPLEMENTATION_DIRECTORY / "weights.pt",
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(initial_weights)

    logits = model(inputs)
    for layer_name in ("a", "z", "b"):
        expected_nodes = np.loadtxt(
            CHECK_DIRECTORY / f"{layer_name}.txt", delimiter=","
        )
        actual_nodes = getattr(model, f"_debug_{layer_name}")[0, :5]
        np.testing.assert_allclose(
            actual_nodes.detach().numpy(), expected_nodes, atol=1e-4
        )

    loss = BASE_EXPERIMENT.CrossEntropyLoss()(logits, targets)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    expected_weights = torch.load(
        CHECK_DIRECTORY / "updated_params.pt",
        map_location="cpu",
        weights_only=True,
    )
    actual_weights = model.state_dict()
    assert actual_weights.keys() == expected_weights.keys()
    for name, expected in expected_weights.items():
        torch.testing.assert_close(actual_weights[name], expected, atol=1e-5, rtol=1e-5)
