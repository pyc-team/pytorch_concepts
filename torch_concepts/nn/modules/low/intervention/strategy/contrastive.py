import warnings
import numpy as np
import torch
from scipy.optimize import minimize

from ...base.intervention import BaseConceptInterventionStrategy


class ContrastiveIntervention(BaseConceptInterventionStrategy):
    """
    Finds the closest point x* to x in concept space such that
    x_to_out_module(x*) == out_target_tensor.

    Solves per-sample:
        min_{x*}  ||x* - x||_p
        s.t.      x_to_out_module(x*) = out_target

    via scipy SLSQP (exact equality constraints).
    Gradients flow back to x via the STE trick.

    Args:
        out_target_tensor:      Target output, shape [B, out_size].
        x_to_out_module:        Module mapping concept space -> output space.
        norm:                   Norm order for the objective (default: 2).
        max_iter:               Max SLSQP iterations (default: 200).
        tol:                    Solver tolerance (default: 1e-6).
        provide_constraint_jac: Pass analytic Jacobian to SLSQP (default: True).
    """

    def __init__(
        self,
        out_target_tensor: torch.Tensor,
        x_to_out_module: torch.nn.Module,
        norm: float = 2.0,
        max_iter: int = 200,
        tol: float = 1e-6,
        provide_constraint_jac: bool = True,
    ):
        super().__init__()
        self.register_buffer("out_target_tensor", out_target_tensor)
        self.x_to_out_module = x_to_out_module
        self.norm = norm
        self.max_iter = max_iter
        self.tol = tol
        self.provide_constraint_jac = provide_constraint_jac

    def _solve_one(self, x_b: torch.Tensor, target_b: torch.Tensor) -> np.ndarray:
        dtype, device = x_b.dtype, x_b.device

        def obj_and_grad(z_np):
            z_t = torch.tensor(z_np, dtype=dtype, device=device, requires_grad=True)
            loss = torch.linalg.norm(z_t - x_b, ord=self.norm)
            loss.backward()
            return loss.item(), z_t.grad.detach().cpu().numpy().astype(np.float64)

        def constraint_fun(z_np):
            z_t = torch.tensor(z_np, dtype=dtype, device=device)
            with torch.no_grad():
                out = self.x_to_out_module(z_t.unsqueeze(0)).squeeze(0)
            return (out - target_b).cpu().numpy().astype(np.float64)

        def constraint_jac(z_np):
            z_t = torch.tensor(z_np, dtype=dtype, device=device).unsqueeze(0)
            jac = torch.autograd.functional.jacobian(
                lambda z: self.x_to_out_module(z).squeeze(0), z_t
            )  # [out_size, 1, F]
            return jac.squeeze(1).detach().cpu().numpy().astype(np.float64)

        constraint = {'type': 'eq', 'fun': constraint_fun}
        if self.provide_constraint_jac:
            constraint['jac'] = constraint_jac

        result = minimize(
            fun=obj_and_grad,
            x0=x_b.detach().cpu().numpy().astype(np.float64),
            jac=True,   # obj_and_grad returns (f, grad) as a tuple
            method='SLSQP',
            constraints=[constraint],
            options={'maxiter': self.max_iter, 'ftol': self.tol}
        )

        if not result.success:
            warnings.warn(f"ContrastiveIntervention: SLSQP did not converge: {result.message}")

        return result.x.astype(np.float32)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        B = x.shape[0]
        assert B == self.out_target_tensor.shape[0], (
            f"Batch size mismatch: x has {B} samples, out_target_tensor has "
            f"{self.out_target_tensor.shape[0]}."
        )

        was_training = self.x_to_out_module.training
        self.x_to_out_module.eval()

        x_star_np = np.stack([
            self._solve_one(x[b].detach(), self.out_target_tensor[b].detach())
            for b in range(B)
        ])  # [B, F]

        if was_training:
            self.x_to_out_module.train()

        x_star = torch.tensor(x_star_np, dtype=x.dtype, device=x.device)  # no grad

        # STE: forward sees x_star, backward straight-through to x
        return x + (x_star - x).detach()
