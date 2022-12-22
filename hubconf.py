import os
import torch
import torchvision

from . import models 

dependencies = ['torch', 'torchvision']

def slip_vits16_clip_25ep_yfcc15M(**kwargs):
  return torch.hub.load("harvard-visionlab/slip", "slip_vits16_clip_25ep_yfcc15M", **kwargs)
