import argparse
import torch
import datetime
import json
import yaml
import os

from scripts import dataloader
from scripts import main_model

from scripts import train
from scripts import loggin
import wandb

parser = argparse.ArgumentParser(description="CSDI")
parser.add_argument("--config", type=str, default="base.yaml")
parser.add_argument('--device', default='cuda:0', help='Device for Attack')
parser.add_argument("--modelfolder", type=str, default="")


parser.add_argument("--nsample", type=int, default=10)
parser.add_argument("--unconditional", action="store_true")

args = parser.parse_args()
print(args)

path = "config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

config["model"]["is_unconditional"] = args.unconditional

print(json.dumps(config, indent=4))


# Initialize wandb
wandb.init(project="test_3", entity="jegomezg")

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") 
foldername = (
    "./save/pm25_validationindex" + "_" + current_time + "/"
)

print('model folder:', foldername)
os.makedirs(foldername, exist_ok=True)
with open(foldername + "config.json", "w") as f:
    json.dump(config, f, indent=4)

train_loader, valid_loader = dataloader.get_data_loaders(
    config["data"]["data_folder"],config["data"]["start_hour"],config["data"]["end_hour"],config["train"]["batch_size"]
)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

model = main_model.CSDI_PM25(config, args.device).to(args.device)
print(f'Model parameters {count_parameters(model)}')

if args.modelfolder == "":
    train.train(
        model,
        config["train"],
        train_loader,
        valid_loader=valid_loader,
        foldername=foldername,
    )
else:
    model.load_state_dict(torch.load("./save/" + args.modelfolder + "/model.pth"))

train.evaluate(
    model,
    valid_loader,
    nsample=args.nsample,
    foldername=foldername,
)