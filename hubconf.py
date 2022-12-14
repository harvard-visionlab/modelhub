import os
import torch
import torchvision

import models 

dependencies = ['torch', 'torchvision']

def slip_vits16_clip_25ep_yfcc15M(**kwargs):
  return torch.hub.load("harvard-visionlab/slip", "vits16_clip_25ep_yfcc15M", **kwargs)
