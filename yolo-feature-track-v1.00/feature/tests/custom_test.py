import argparse

import torch

from ..main import get_model_class

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='center loss example')
    parser.add_argument('--arch', type=str, default='resnet18',
                        help='network arch to use, support resnet18 and '
                             'resnet50 (default: resnet50)')
    args = parser.parse_args()
    ######################################################################################
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ######################################################################################
    model_class = get_model_class(args)
    model = model_class(False).to(device)
    checkpoint = torch.load("../weights/models/epoch_100.pth.tar")
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.eval()
    ######################################################################################
    torch.zeros(())
    print(model_class)