# Ideas
# 1) Run evaluation for pushing task with parameters optimized by bayesian optimization
#       print confirmation info, print metrics, plot metrics, save to figures
# 2) Run eval with two baseline parametersn
#       print confirmation info, print metrics, plot metrics, save to figures
# * pushing task: which environment with which obstacle setup
# Note: print out an expected time to run before starting

import torch
import os
from optimizer.panda_pushing_optimizer import PandaBoxPushingStudy, PandaHoverStudy
from env.visualizers import GIFVisualizer
import numpy as np
import time
import random

seed = 666
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)



class TerminalColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'

def print_visualization_warning():
    print(TerminalColors.BOLD + TerminalColors.RED + "=====================================" + TerminalColors.ENDC)
    for _ in range(5):
        print(TerminalColors.BOLD + TerminalColors.RED + "Attention please: Visualization Window May Lay at the bottom!" + TerminalColors.ENDC)
    print(TerminalColors.BOLD + TerminalColors.RED + "=====================================" + TerminalColors.ENDC)



if __name__ == "__main__":

    print(TerminalColors.BOLD + "=====================================" + TerminalColors.ENDC)
    print(TerminalColors.BOLD + "Here Is Demo From Yulun Zhuang & Ziqi Han. \n" + TerminalColors.ENDC)
    print(TerminalColors.BOLD + "Hover Task Demo:\n" + TerminalColors.ENDC)
    print(TerminalColors.BOLD + "The robot arm will hover at a target position.\n" + TerminalColors.ENDC)
    print(TerminalColors.BOLD + "Please Type Enter To Start!\n" + TerminalColors.ENDC)
    print(TerminalColors.BOLD + "====================================" + TerminalColors.ENDC)


    EPOCH = 5
    RENDER = True
    LOGDIR = "logs/"

    if not os.path.exists(LOGDIR):
        os.mkdir(LOGDIR)

    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    # DEVICE = "cpu"
    visualizer = GIFVisualizer()
    
    print(TerminalColors.OKGREEN + f"Start Hover task for {EPOCH} epoch(s)!" + TerminalColors.ENDC)
    print(TerminalColors.OKGREEN + f"Note that the visualization window may lay at the bottom!" + TerminalColors.ENDC)
    confirm_message = TerminalColors.OKGREEN + "Please Enter to continue..." + TerminalColors.ENDC
    confirm = input(confirm_message)
    print("Confirmed")
    print_visualization_warning()
    time.sleep(2)

    # Test parameters for hover task (may need tuning)
    test_param_hover = [0.01, 2.5, 2.5, 2.5]

    # visualizer.reset()
    test_hover = PandaHoverStudy(EPOCH, RENDER, LOGDIR, 
                                 study_name="test_hover", 
                                 random_target=True,
                                 opt_type="test", 
                                 step_scale=0.1, 
                                 device=DEVICE,
                                 test_params=test_param_hover,
                                 visualizer=visualizer)
    test_hover.run()
