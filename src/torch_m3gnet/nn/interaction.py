import torch


class ThreeBodyInteration(torch.nn.Module):
    """Renormalize triplets to edge features."""

    def __init__(self):
        super().__init__()


class SphericalBessel(torch.autograd.Function):
    """Spherical Bessel function of the first kind."""

    @staticmethod
    def forward(ctx, input: torch.Tensor, order: int):
        EPS = 1e-8
        assert order >= 0

        # torch.sinc(x) = sin(pi * x) / (pi * x)
        out = []
        out.append(
            torch.where(
                input > EPS,
                torch.sin(input) / input,
                torch.ones_like(input),
            )
        )
        if order >= 1:
            out.append(
                torch.where(
                    input > EPS,
                    (torch.sin(input) / input - torch.cos(input)) / input,
                    input / 3,
                )
            )
            coeff = 3
            for n in range(1, order):
                coeff *= 2 * n + 3
                out.append(
                    torch.where(
                        input > EPS,
                        (2 * n + 1) / input * out[n] - out[n - 1],
                        input / coeff,
                    )
                )
        out = torch.stack(out)

        ctx.order = order
        ctx.save_for_backward(input, out)
        return out[-1]

    @staticmethod
    def backward(ctx, grad_output):
        EPS = 1e-8
        input, out = ctx.saved_tensors
        order = ctx.order

        if order == 0:
            grad = torch.where(
                input > EPS,
                -(torch.sin(input) / input - torch.cos(input)) / input * grad_output,
                torch.zeros_like(grad_output),
            )
        elif order == 1:
            grad = torch.where(
                input > EPS,
                (out[order - 1] - (order + 1) / input * out[order]) * grad_output,
                grad_output / 3,
            )
        else:
            grad = torch.where(
                input > EPS,
                (out[order - 1] - (order + 1) / input * out[order]) * grad_output,
                torch.zeros_like(grad_output),
            )

        return grad, None


class LegendreCosPolynomial(torch.autograd.Function):
    """Legendre polynomial P_{n}(cos x)."""

    @staticmethod
    def forward(ctx, input: torch.Tensor, order: int):
        assert order >= 0

        out = [torch.ones_like(input)]
        if order >= 1:
            out.append(input)

            for n in range(1, order):
                out.append(((2 * n + 1) * input * out[n] - n * out[n - 1]) / (n + 1))

        out = torch.stack(out)

        ctx.order = order
        ctx.save_for_backward(input, out)
        return out[-1]

    @staticmethod
    def backward(ctx, grad_output):
        input, out = ctx.saved_tensors
        order = ctx.order

        grad = [torch.zeros_like(grad_output)]
        for n in range(1, order + 1):
            grad.append((n * out[n - 1] + input * grad[n - 1]) * grad_output)

        return grad[-1], None


spherical_bessel = SphericalBessel.apply
legendre_cos = LegendreCosPolynomial.apply
