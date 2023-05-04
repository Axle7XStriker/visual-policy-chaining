from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym.spaces

from .encoder import Encoder
from .utils import MLP
from ..utils.pytorch import to_tensor, center_crop, center_crop_images


class Discriminator(nn.Module):
    def __init__(
        self,
        config,
        ob_space,
        ob_next_space=None,
        ac_space=None,
        mlp_dim=[256, 256],
        activation="tanh",
        encoder=None,
    ):
        super().__init__()
        self._config = config

        if encoder:
            self.encoder = encoder
        else:
            self.encoder = Encoder(config, ob_space)

        input_dim = self.encoder.output_dim
        if ob_next_space:
            input_dim += self.encoder.output_dim
        if ac_space:
            input_dim += gym.spaces.flatdim(ac_space)

        self.fc = MLP(
            config,
            input_dim,
            1,
            mlp_dim,
            getattr(F, activation),
        )

    def forward(self, ob, ob_next=None, ac=None):
        # # flatten observation
        # if isinstance(ob, dict):
        #     ob = list(ob.values())
        #     if len(ob[0].shape) == 1:
        #         ob = [x.unsqueeze(0) for x in ob]
        #     ob = torch.cat(ob, dim=-1)

        # if ob_next is not None:
        #     if isinstance(ob_next, dict):
        #         ob_next = list(ob_next.values())
        #         if len(ob_next[0].shape) == 1:
        #             ob_next = [x.unsqueeze(0) for x in ob_next]
        #         ob_next = torch.cat(ob_next, dim=-1)
        #     ob = torch.cat([ob, ob_next], dim=-1)

        # encode observations
        ob = self.encoder(ob)
        if ob_next is not None:
            ob_next = self.encoder(ob_next)
            ob = torch.cat([ob, ob_next], dim=-1)

        if ac is not None:
            # flatten action
            if isinstance(ac, dict):
                ac = list(ac.values())
                if len(ac[0].shape) == 1:
                    ac = [x.unsqueeze(0) for x in ac]
                ac = torch.cat(ac, dim=-1)
            ob = torch.cat([ob, ac], dim=-1)

        out = self.fc(ob)
        return out
