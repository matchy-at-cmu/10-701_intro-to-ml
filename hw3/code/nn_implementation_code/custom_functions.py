import torch

from torch.autograd import Function


class IdentityFunction(Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, x):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        return x

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        return grad_output


class SigmoidFunction(Function):
    @staticmethod
    def forward(ctx, input):
        pass

    @staticmethod
    def backward(ctx, grad_output):
        pass


class LinearFunction(Function):
    @staticmethod
    def forward(ctx, inp, weight, bias):
        pass

    @staticmethod
    def backward(ctx, grad_output):
        pass


class CrossEntropyFunction(Function):
    @staticmethod
    def forward(ctx, logits, target):
        pass

    @staticmethod
    def backward(ctx, grad_output):
        pass


if __name__ == "__main__":
    from torch.autograd import gradcheck

    num = 4
    inp = 3

    x = torch.rand((num, inp), requires_grad=True).double()

    sigmoid = SigmoidFunction.apply

    assert gradcheck(sigmoid, x)
    print("Backward pass for sigmoid function is implemented correctly")

    out = 2

    x = torch.rand((num, inp), requires_grad=True).double()
    weight = torch.rand((out, inp), requires_grad=True).double()
    bias = torch.rand(out, requires_grad=True).double()

    linear = LinearFunction.apply
    assert gradcheck(linear, (x, weight, bias))
    print("Backward pass for linear function is implemented correctly")

    activations = torch.rand((15, 10), requires_grad=True).double()
    target = torch.randint(10, (15,))
    crossentropy = CrossEntropyFunction.apply
    assert gradcheck(crossentropy, (activations, target))
    print("Backward pass for crossentropy function is implemented correctly")
