import numpy as np
import torch
from PIL import Image
import os
import pandas as pd
import random
from engine import train_one_epoch, evaluate
import utils
import transform_data as td

#import transforms as T
import torchvision.transforms.v2 as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator,RPNHead
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from functools import partial
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
from torchvision.models.detection import generalized_rcnn
import torchvision
import torchvision.transforms as transforms
from ray import tune
from ray.air import CheckpointConfig, session
from ray.tune.schedulers import ASHAScheduler
from ray import train
import math

from hyperopt import hp
from ray.tune.search.hyperopt import HyperOptSearch
from Train_Pits_FasterRCNN import Pits_Dataset, get_transform, get_model_object_detection

num_train = 1110
num_eval = 180
num_epochs = 50
batch_size = 2

# weight_decay = 0.0005
# lr = 0.008
# momentum = 0.9
step_size = 15
gamma = 0.1
#





def train_func(model, optimizer, train_loader, i):
    print("train_func")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model.to(device)
    model.train()
    """for batch_idx, (images, targets) in enumerate(train_loader):
        # We set this just for the example to run quickly.
        if batch_idx * len(images) > num_epochs*10:
            return
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()

        output = model(images, targets)
        loss_dict_reduced = utils.reduce_dict(output)

        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)

        losses_reduced.backward()
        optimizer.step()"""
    train_one_epoch(model, optimizer, train_loader, device, i, 1)


def test_func(model, data_loader_test):
    print("test_func")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model.to(device)
    model.eval()
    coc_eval = evaluate(model, data_loader_test, device)
    AR = coc_eval.coco_eval["bbox"].stats[-1]
    AP = coc_eval.coco_eval["bbox"].stats[1]
    return AR, AP

def train_faster(config):
    print("train_faster")
    Data_Path_tr = r"D:\Trainingsdaten_v07\Trainingsdaten_v07\crop"
    Data_JSON_tr = "labels_croptrain_v07.json"
    Data_Path_val = r"D:\Trainingsdaten_v07\Validationsdaten_v06\crop"
    Data_JSON_val = "labels_cropval_v06_cropd.json"


    dataset_train = Pits_Dataset(Data_Path_tr,
                                 datafile=Data_JSON_tr, train=True,
                                 transforms=get_transform(train=True))

    dataset_test = Pits_Dataset(Data_Path_val,
                                datafile=Data_JSON_val, train=False,
                                transforms=get_transform(train=False))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model_object_detection(3, model_arch=50)
    model.to(device)

    data_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True, num_workers=0,
        collate_fn=utils.collate_fn)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=utils.collate_fn)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, weight_decay=config["weightdecay"], lr=config["lr"], momentum=config["momentum"])

    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=step_size,
                                                   gamma=gamma)

    for i in range(10):
        train_func(model, optimizer, data_loader, i)
        AR, AP = test_func(model, data_loader_test)

        # Send the current training result back to Tune
        train.report({"AR": AR, "AP":AP})

        if i % 5 == 0:
            # This saves the model to the trial directory
            #torch.save(model.state_dict(), "./model.pth")
            a = 12

"""search_space = {
    "lr": tune.uniform(0.008, 0.0095),
    "momentum": tune.uniform(0.85, 0.95),
    "weightdecay": tune.uniform(0.0001, 0.001)
}


tuner = tune.Tuner(
    train_faster,
    param_space=search_space,
    tune_config= tune.TuneConfig(search_alg=search_alg)
)
results = tuner.fit()

dfs = {result.path: result.metrics_dataframe for result in results}
[d.AR.plot() for d in dfs.values()]"""

space = {
    "lr": hp.uniform("lr", 0.001, 0.03),
    "momentum": hp.uniform("momentum", 0.8, 1.2),
    "weightdecay": hp.uniform("weightdecay", 0,0.005)
}

hyperopt_search = HyperOptSearch(space, metric="AR", mode="max")
hyperopt_search = tune.search.ConcurrencyLimiter(hyperopt_search, max_concurrent=1)


tuner = tune.Tuner(
    train_faster,
    tune_config=tune.TuneConfig(
        num_samples=20,
        search_alg=hyperopt_search,

    ),
)
results = tuner.fit()

# To enable GPUs, use this instead:
analysis = tune.run(
     train_faster, config=space, resources_per_trial={'gpu': 1})
