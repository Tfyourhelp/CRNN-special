import torch
#print(torch.__version__)
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import CTCLoss
from torch.utils.data import Dataset

from scipy import signal
from scipy.io import wavfile

from tqdm import tqdm
from PIL import Image
from typing import Union, List, Tuple, Callable, Optional, Any
from collections import defaultdict
from torch.utils.data import  ConcatDataset


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import os
import glob
import string
import cv2
import copy
import argparse
import yaml
import random
import datetime
import math
import json

import wandb