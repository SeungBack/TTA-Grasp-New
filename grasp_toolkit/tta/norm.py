# Copyright 2020-2021 Evgenia Rusak, Steffen Schneider, George Pachitariu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#
# ---
# This licence notice applies to all originally written code by the
# authors. Code taken from other open-source projects is indicated.
# See NOTICE for a list of all third-party licences used in the project.

"""Batch norm variants
AlphaBatchNorm builds upon: https://github.com/bethgelab/robustness/blob/main/robusta/batchnorm/bn.py
"""


import sys
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from copy import deepcopy

# ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.join(ROOT_DIR, '../..'))

from .base import TTA_Base

class AlphaBatchNorm(nn.Module):
    """ Use the source statistics as a prior on the target statistics """

    @staticmethod
    def find_bns(parent, alpha):
        replace_mods = []
        if parent is None:
            return []
        for name, child in parent.named_children():
            if isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d)):
                module = AlphaBatchNorm(child, alpha)
                replace_mods.append((parent, name, module))
            else:
                replace_mods.extend(AlphaBatchNorm.find_bns(child, alpha))

        return replace_mods

    @staticmethod
    def adapt_model(model, alpha):
        replace_mods = AlphaBatchNorm.find_bns(model, alpha)
        print(f"| Found {len(replace_mods)} modules to be replaced.")
        for (parent, name, child) in replace_mods:
            setattr(parent, name, child)
        return model

    def __init__(self, layer, alpha):
        assert alpha >= 0 and alpha <= 1

        super().__init__()
        self.layer = layer
        self.layer.eval()
        self.alpha = alpha

        if isinstance(self.layer, nn.BatchNorm1d):
            self.norm = nn.BatchNorm1d(self.layer.num_features, affine=False, momentum=1.0)
        elif isinstance(self.layer, nn.BatchNorm2d):
            self.norm = nn.BatchNorm2d(self.layer.num_features, affine=False, momentum=1.0)
        else:
            raise ValueError(f"Unsupported layer type: {type(self.layer)}")


    def forward(self, input):

        self.norm(input)
        running_mean = ((1 - self.alpha) * self.layer.running_mean + self.alpha * self.norm.running_mean)
        running_var = ((1 - self.alpha) * self.layer.running_var + self.alpha * self.norm.running_var)

        return F.batch_norm(
            input,
            running_mean,
            running_var,
            self.layer.weight,
            self.layer.bias,
            False,
            0,
            self.layer.eps,
        )


class EMABatchNorm(nn.Module):
    @staticmethod
    def adapt_model(model):
        model = EMABatchNorm(model)
        return model

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # store statistics, but discard result
        self.model.train()
        self.model(x)
        # store statistics, use the stored stats
        self.model.eval()
        return self.model(x)



class Norm(TTA_Base):
    def __init__(self, cfg, model):
        super().__init__(cfg, model)

    @torch.no_grad()
    def forward_and_adapt(self, x):
        end_points = self.model(x)
        grasp_preds = self.pred_decode(end_points)
        return grasp_preds, end_points

    def configure_model(self):
        self.model.eval()
        self.model.requires_grad_(False)
        if self.cfg.tta.method == "bn-1":  # BN--1
            for m in self.model.modules():
                # Re-activate batchnorm layer
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    m.train()
        elif self.cfg.tta.method == "bn-adapt":  # BN--0.1
            # (1-alpha) * src_stats + alpha * test_stats
            self.model = AlphaBatchNorm.adapt_model(self.model, alpha=0.1).cuda()
        elif self.cfg.tta.method == "bn-ema":  # BN--EMA
            self.model = EMABatchNorm.adapt_model(self.model).cuda()