import os

from dataset_constructor import clear_directory
from torch.utils.tensorboard import SummaryWriter

root_dir = str(os.path.dirname(os.path.abspath(__file__))).split("util")[0]
log_dir = root_dir + "tensorboard-logs/"
clear_directory(log_dir)
writer = SummaryWriter(log_dir=log_dir)
