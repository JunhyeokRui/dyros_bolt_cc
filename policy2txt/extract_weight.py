import json
import os
import argparse
import torch
from torch.utils.data import DataLoader
from configparser import ConfigParser
from datetime import datetime

import numpy as np

def run():
    
    try: os.mkdir("./result/bolt")
    except: pass
    model_path = os.path.join("checkpoints", "policy_1.pt")
    model = torch.jit.load(model_path, map_location=torch.device('cpu'))
    
    ## save training weights to .txt file
    for name, param in model.named_parameters():
        with open(f"./result/weights/{name}.txt", "w") as text_file:
            # Convert parameter to numpy array and then to string
            param_str = str(param.detach().numpy())
            text_file.write(param_str)
    

if __name__ == '__main__':

    run()


