import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

!pip install hf_xet grad-cam transformers torchvision imbalanced-learn opencv-python ipywidgets
!huggingface-cli login

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from transformers import ViTForImageClassification
import numpy as np
from collections import Counter
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from google.colab import drive
import time
from imblearn.over_sampling import RandomOverSampler
import cv2
import seaborn as sns
from IPython.display import clear_output
import ipywidgets as widgets
from ipywidgets import interact
import random
from IPython.display import display, HTML
import warnings
warnings.filterwarnings('ignore')
