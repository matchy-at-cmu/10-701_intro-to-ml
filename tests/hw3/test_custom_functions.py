import torch
import torch.nn.functional as functional

from hw3.code.nn_implementation_code.custom_functions import (
    CrossEntropyFunction,
    IdentityFunction,
    LinearFunction,
    SigmoidFunction,
)


def test_identity_returns_input_and_passes_gradient_through() -> None:
    inputs = torch.tensor([[1.0, -2.0], [3.5, 0.25]], requires_grad=True)

    outputs = IdentityFunction.apply(inputs)
    outputs.sum().backward()

    assert torch.equal(outputs, inputs)
    assert torch.equal(inputs.grad, torch.ones_like(inputs))


def test_sigmoid_matches_pytorch_forward_and_backward() -> None:
    values = torch.tensor([-20.0, -1.5, 0.0, 2.0, 20.0], dtype=torch.float64)
    custom_input = values.clone().requires_grad_()
    reference_input = values.clone().requires_grad_()
    upstream_gradient = torch.tensor([0.5, -1.0, 2.0, 0.25, -0.75])

    custom_output = SigmoidFunction.apply(custom_input)
    reference_output = torch.sigmoid(reference_input)
    custom_gradient = torch.autograd.grad(
        custom_output, custom_input, upstream_gradient
    )[0]
    reference_gradient = torch.autograd.grad(
        reference_output, reference_input, upstream_gradient
    )[0]

    torch.testing.assert_close(custom_output, reference_output)
    torch.testing.assert_close(custom_gradient, reference_gradient)


def test_linear_matches_pytorch_forward_and_all_gradients() -> None:
    inputs = torch.tensor([[1.0, -2.0, 0.5], [0.0, 3.0, -1.0]], dtype=torch.float64)
    weights = torch.tensor([[0.25, -0.5, 2.0], [-1.0, 0.75, 0.5]], dtype=torch.float64)
    bias = torch.tensor([0.1, -0.2], dtype=torch.float64)
    upstream_gradient = torch.tensor([[1.0, -0.5], [2.0, 0.25]], dtype=torch.float64)

    custom_inputs = tuple(
        value.clone().requires_grad_() for value in (inputs, weights, bias)
    )
    reference_inputs = tuple(
        value.clone().requires_grad_() for value in (inputs, weights, bias)
    )
    custom_output = LinearFunction.apply(*custom_inputs)
    reference_output = functional.linear(*reference_inputs)
    custom_gradients = torch.autograd.grad(
        custom_output, custom_inputs, upstream_gradient
    )
    reference_gradients = torch.autograd.grad(
        reference_output, reference_inputs, upstream_gradient
    )

    torch.testing.assert_close(custom_output, reference_output)
    for custom_gradient, reference_gradient in zip(
        custom_gradients, reference_gradients
    ):
        torch.testing.assert_close(custom_gradient, reference_gradient)


def test_cross_entropy_matches_pytorch_forward_and_logits_gradient() -> None:
    logits = torch.tensor([[2.0, -1.0, 0.5], [-0.5, 1.5, 0.0]], dtype=torch.float64)
    targets = torch.tensor([0, 2])
    custom_logits = logits.clone().requires_grad_()
    reference_logits = logits.clone().requires_grad_()

    custom_loss = CrossEntropyFunction.apply(custom_logits, targets)
    reference_loss = functional.cross_entropy(reference_logits, targets)
    custom_gradient = torch.autograd.grad(custom_loss, custom_logits)[0]
    reference_gradient = torch.autograd.grad(reference_loss, reference_logits)[0]

    torch.testing.assert_close(custom_loss, reference_loss)
    torch.testing.assert_close(custom_gradient, reference_gradient)


def test_cross_entropy_is_stable_for_large_logits() -> None:
    logits = torch.tensor([[1000.0, 1001.0, 999.0], [-1000.0, -1000.0, -999.0]])
    targets = torch.tensor([1, 0])

    custom_loss = CrossEntropyFunction.apply(logits, targets)
    reference_loss = functional.cross_entropy(logits, targets)

    assert torch.isfinite(custom_loss)
    torch.testing.assert_close(custom_loss, reference_loss)
