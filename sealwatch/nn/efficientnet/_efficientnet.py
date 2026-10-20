"""

Author: Martin Benes
Affiliation: University of Innsbruck
"""

import logging
from pathlib import Path
import timm
import torch
from typing import Union

from .. import _common


class EfficientNet(torch.nn.Module):
    """EfficientNet from timm, adapted for steganalysis.

    The state dict uses the keys of timm, without the backbone prefix.
    """
    variant: str = None

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        no_stem_stride: bool = True,
        weight_seed: int = None,
        **kw,
    ):
        super().__init__()
        with torch.random.fork_rng():
            if weight_seed is not None:
                torch.manual_seed(weight_seed)
            backbone = timm.create_model(
                self.variant,
                in_chans=in_channels,
                num_classes=num_classes,
                **kw,
            )
        if no_stem_stride:
            backbone.conv_stem.stride = (1, 1)
        self.add_module('backbone', backbone)
        self.pretrained_cfg = backbone.pretrained_cfg
        # hooks respect the prefix, so that they also work when nested
        self.register_state_dict_post_hook(self._strip_backbone_prefix)
        self.register_load_state_dict_pre_hook(self._add_backbone_prefix)

    @staticmethod
    def _strip_backbone_prefix(module, state_dict, prefix, local_metadata):
        for k in [k for k in state_dict if k.startswith(f'{prefix}backbone.')]:
            state_dict[prefix + k[len(f'{prefix}backbone.'):]] = state_dict.pop(k)

    @staticmethod
    def _add_backbone_prefix(module, state_dict, prefix, *args):
        for k in [k for k in state_dict if k.startswith(prefix)]:
            if not k.startswith(f'{prefix}backbone.'):
                state_dict[f'{prefix}backbone.' + k[len(prefix):]] = state_dict.pop(k)

    def forward(self, x):
        """Must be defined explicitly."""
        return self.backbone(x)

    def __repr__(self) -> str:
        return repr(self.backbone)

    @classmethod
    def pretrained(
        cls,
        path: Union[str, Path] = None,
        *,
        in_channels: int = 1,
        num_classes: int = 2,
        device: torch.device = torch.device('cpu'),
        strict: bool = True,
        **kw,
    ) -> 'EfficientNet':
        """Loads pretrained model. Downloads if missing.

        By default, loads the ImageNet weights of timm,
        with the input convolution adapted to in_channels,
        and the classifier initialized randomly, i.e., to be fine-tuned.

        :param path: local path to the weights of a trained detector,
            by default downloads the ImageNet weights into the torch hub cache
        :type path: str or pathlib.Path
        :param in_channels: number of input channels
        :type in_channels: int
        :param num_classes: number of output classes
        :type num_classes: int
        :param device: torch device
        :type device: torch.device
        :param strict: whether the weights must match exactly
        :type strict: bool
        :return: loaded model in eval mode
        :rtype: EfficientNet
        """
        # create model
        model = cls(in_channels=in_channels, num_classes=num_classes, **kw)

        if path is None:
            # download weights
            source = model.pretrained_cfg['url']
            state_dict = _common.load_state_dict(source)

            # adapt input channels
            if state_dict['conv_stem.weight'].size(1) != in_channels:
                state_dict['conv_stem.weight'] = timm.models.adapt_input_conv(
                    in_channels, state_dict['conv_stem.weight'])

            # remove FC
            model_state_dict = model.state_dict()
            state_dict['classifier.weight'] = model_state_dict['classifier.weight']
            state_dict['classifier.bias'] = model_state_dict['classifier.bias']
        else:
            source = path
            state_dict = _common.load_state_dict(source)

        # load weights
        model.load_state_dict(state_dict, strict=strict)
        logging.info(f'using pretrained weights from {source}')
        return model.to(device).eval()


class B0(EfficientNet):
    """EfficientNet-B0."""
    variant = 'efficientnet_b0'


class B4(EfficientNet):
    """EfficientNet-B4."""
    variant = 'efficientnet_b4'
