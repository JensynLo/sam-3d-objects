# Copyright (c) Meta Platforms, Inc. and affiliates.
import torch
import numpy as np

from sam3d_objects.model.backbone.generator.flow_matching.model import FlowMatching, _get_device


# https://arxiv.org/pdf/2410.12557
class ShortCut(FlowMatching):
    def __init__(
        self,
        no_shortcut=False,
        self_consistency_prob=0.25,
        shortcut_loss_weight=1.0,
        self_consistency_cfg_strength=3.0,
        ratio_cfg_samples_in_self_consistency_target=0.5,
        fm_in_shortcut_target_prob=0.0,
        fm_eps_max=0,
        batch_mode=False,
        cfg_modalities=["shape"],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.no_shortcut = no_shortcut
        self.self_consistency_prob = self_consistency_prob
        self.shortcut_loss_weight = shortcut_loss_weight
        self.self_consistency_cfg_strength = self_consistency_cfg_strength
        self.ratio_cfg_samples_in_self_consistency_target = ratio_cfg_samples_in_self_consistency_target
        self.fm_in_shortcut_target_prob = fm_in_shortcut_target_prob
        self.fm_eps_max = fm_eps_max
        self.batch_mode = batch_mode
        self.cfg_modalities = cfg_modalities


        
    def _prepare_t_and_d(self, steps=None):
        """Prepare time sequence and step size for inference"""
        steps = self.inference_steps if steps is None else steps
        t_seq = np.linspace(0, 1, steps + 1)

        if self.no_shortcut:
            d = 0
        else:
            # Use uniform step size for inference
            d = 1 / steps

        if self.rescale_t:
            t_seq = t_seq / (1 + (self.rescale_t - 1) * (1 - t_seq))

        if self.reversed_timestamp:
            t_seq = 1 - t_seq

        return t_seq, d

    def generate_iter(
        self,
        x_shape,
        x_device,
        *args_conditionals,
        **kwargs_conditionals,
    ):
        """Generate samples using shortcut model"""
        x_0 = self._generate_noise(x_shape, x_device)
        t_seq, d = self._prepare_t_and_d()

        for x_t, t in self._solver.solve_iter(
            self._generate_dynamics,
            x_0,
            t_seq,
            d,
            *args_conditionals,
            **kwargs_conditionals,
        ):
            yield t, x_t, ()

    def _generate_dynamics(
        self,
        x_t,
        t,
        d,
        *args_conditionals,
        **kwargs_conditionals,
    ):
        """Generate dynamics for ODE solver"""
        t = torch.tensor(
            [t * self.time_scale], device=_get_device(x_t), dtype=torch.float32
        )
        d = torch.tensor(
            [d * self.time_scale], device=_get_device(x_t), dtype=torch.float32
        )
        return self.reverse_fn(x_t, t, *args_conditionals, d=d, **kwargs_conditionals)
