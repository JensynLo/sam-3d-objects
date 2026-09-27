# Copyright (c) Meta Platforms, Inc. and affiliates.
from typing import Callable, Sequence, Union
import torch
from functools import partial
import optree

from sam3d_objects.model.backbone.generator.base import Base
from sam3d_objects.data.utils import tree_reduce_unique
from sam3d_objects.model.backbone.generator.flow_matching.solver import ODESolver, Euler


# https://arxiv.org/pdf/2403.03206
def lognorm_sampler(mean=0.0, std=1.0, **kwargs):
    logit = torch.randn(**kwargs) * std + mean
    return torch.nn.functional.sigmoid(logit)


# https://arxiv.org/pdf/2210.02747
class FlowMatching(Base):
    SOLVER_METHODS = {
        "euler": Euler,
    }

    def __init__(
        self,
        reverse_fn: Callable,
        sigma_min: float = 0.0,  # 0. = rectifier flow
        inference_steps: int = 100,
        time_scale: float = 1000.0,  # scale [0,1]-time before passing to `reverse_fn`
        training_time_sampler_fn: Callable = partial(
            lognorm_sampler,
            mean=0,
            std=1,
        ),
        reversed_timestamp=False,
        rescale_t=1.0,
        loss_fn=partial(torch.nn.functional.mse_loss, reduction="mean"),
        loss_weights=1.0,
        solver_method: Union[str, ODESolver] = "euler",
        solver_kwargs: dict = {},
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.reverse_fn = reverse_fn
        self.sigma_min = sigma_min
        self.inference_steps = inference_steps
        self.time_scale = time_scale
        self.training_time_sampler_fn = training_time_sampler_fn
        self.reversed_timestamp = reversed_timestamp
        self.rescale_t = rescale_t
        self.loss_fn = loss_fn
        self.loss_weights = loss_weights
        self._solver_method, self._solver = self._get_solver(
            solver_method, solver_kwargs
        )

    def _get_solver(self, solver_method, solver_kwargs):
        if solver_method in FlowMatching.SOLVER_METHODS:
            solver = FlowMatching.SOLVER_METHODS[solver_method](**solver_kwargs)
        elif isinstance(solver_method, ODESolver):
            solver_method = f"custom[{solver_method.__class__.__name__}]"
            solver = solver_method
        else:
            raise ValueError(
                f"Invalid solver `{solver_method}`, should be in {set(self.SOLVER_METHODS.keys())} or an ODESolver instance"
            )
        return solver_method, solver

    def _generate_noise_tensor(self, x_shape, x_device):
        return torch.randn(
            x_shape,
            # generator=self.random_generator,
            device=x_device,
        )

    def _generate_noise(self, x_shape, x_device):
        def is_shape(maybe_shape):
            return isinstance(maybe_shape, Sequence) and all(
                (isinstance(s, int) and s >= 0) for s in maybe_shape
            )

        return optree.tree_map(
            partial(self._generate_noise_tensor, x_device=x_device),
            x_shape,
            is_leaf=is_shape,
            none_is_leaf=False,
        )


    def _prepare_t(self, steps=None):
        steps = self.inference_steps if steps is None else steps
        t_seq = torch.linspace(0, 1, steps + 1)

        if self.rescale_t:
            t_seq = t_seq / (1 + (self.rescale_t - 1) * (1 - t_seq))

        if self.reversed_timestamp:
            t_seq = 1 - t_seq

        return t_seq

    def generate_iter(
        self,
        x_shape,
        x_device,
        *args_conditionals,
        **kwargs_conditionals,
    ):
        x_0 = self._generate_noise(x_shape, x_device)
        t_seq = self._prepare_t().to(x_device)

        for x_t, t in self._solver.solve_iter(
            self._generate_dynamics,
            x_0,
            t_seq,
            *args_conditionals,
            **kwargs_conditionals,
        ):
            yield t, x_t, ()

    def _generate_dynamics(
        self,
        x_t,
        t,
        *args_conditionals,
        **kwargs_conditionals,
    ):
        return self.reverse_fn(x_t, t * self.time_scale, *args_conditionals, **kwargs_conditionals)




def _get_device(x):
    device = tree_reduce_unique(lambda tensor: tensor.device, x)
    return device


