import numpy as np
import torch
from PIL import Image
import os
import pandas as pd
import random
from engine import train_one_epoch, evaluate
import utils
import transform_data as td
from torchvision.transforms.v2 import functional as F
import torchvision
import transforms as T
import torchvision.transforms.v2 as v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator,RPNHead
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone,BackboneWithFPN
from matplotlib import pyplot as plt


class Pits_Dataset(torch.utils.data.Dataset):

    """this dataset can be used to create a dataset containing images, bounding boxes and labels from
     data that was labeled in matlab and extracted as .json file, the matlab script to extract data using correct format is in the
     folder matlab_data_extraction"""

    def __init__(self, root, datafile, transforms, train):
        self.train = train
        self.transforms = transforms
        self.root = root
        self.datafile = datafile
        # load image names and bounding box data from root folder and json file:
        self.data = pd.read_json(os.path.join(root, datafile))

        # this to only pick labeled samples
        self.empty_ind = []
        for i in range(len(self.data)):
            if len(self.data.iloc[i]["Pit"]) + len(self.data.iloc[i]["Cluster"]) == 0:
               self.empty_ind.append(i)
        self.data.drop(index = self.empty_ind, inplace=True)
        self.data.reset_index(drop=True, inplace=True)
        self.img_names = self.data["names"].tolist() # os.listdir(os.path.join(root, "raw"))
        print(f"{len(self.data)} annotated imgs")

        # this to only pick images from json that are in raw directory
        self.empty_ind = []
        for namei in self.img_names:
            if namei not in os.listdir(os.path.join(root, "raw")):
                self.empty_ind.append(np.argmax(self.data["names"] == namei))
        self.data.drop(index=self.empty_ind, inplace=True)
        self.data.reset_index(drop=True, inplace=True)
        self.img_names = self.data["names"].tolist()  # os.listdir(os.path.join(root, "raw"))
        print("Ok")



    def __getitem__(self, idx):
        # load images and gt boxes
        #d = self.data.iloc[idx].to_dict()
        #name = d["names"]


        name = self.img_names[idx]
        d = self.data[self.data["names"] == name].to_dict("records")[0]
        del d["names"]
        d["Pit"] = torch.tensor(d["Pit"])
        d["Cluster"] = torch.tensor(d["Cluster"])


        box_list = []
        lab_list = []

        # With N objects in an image, create a list of N Tensors, each with 4 bbox coordinates
        # Also create list of N Labels
        for key, ele in d.items():
            if torch.numel(ele) == 4:
                box_list.append(ele.reshape(4).to(torch.float32))
                lab_list.append(key)
            elif torch.numel(ele) > 4:
                if torch.numel(ele) % 4 == 0:
                    num = int(torch.numel(ele) / 4)
                    lab_list += num * [key]
                    for box in range(num):
                        box_list.append(ele[box].to(torch.float32))


        # Change matlab box coordinates [x1 x2 w h] to [x1 y1 x2 y2]:
        mat_to_py_boxes = []
        #print(name)

        """
        for ind, box in enumerate(targets[0]["boxes"]):
            if box[0] > box[2]:
                print(f"x; ind {ind}; {box}")
            if box[1] > box[3]:
                print(f"y; ind {ind}; {box}")"""
        for box in box_list:
            x1 = box[0]
            y1 = box[1]
            x2 = x1 + box[2]
            y2 = y1 + box[3]
            mat_to_py_boxes.append(torch.FloatTensor([x1,y1,x2,y2]))
            #print("oi")

        # Encode labels into numeric values (see below)
        encoded_labels = []
        iscrowd_list = []
        for lab in lab_list:
            if lab == "Pit":
                encoded_labels.append(1)
                iscrowd_list.append(0)
            elif lab == "Cluster":
                encoded_labels.append(2)
                iscrowd_list.append(1)



        img_path = os.path.join(self.root, "raw", name)

        # converting 16 bit images back down to 8 bit for now, might need images correctly extracted 8-bit imgs
        # in the future. Is 16 bit training also a possibility (?)
        img = Image.open(img_path).convert("RGB")  #TODO: RGB needed?

        # create empty tensors for boxes and areas if no gt given
        if len(mat_to_py_boxes) > 0:
            boxes = torch.stack(mat_to_py_boxes)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        else:
            boxes = torch.tensor([])
            area = torch.tensor([])

        labels = torch.tensor(encoded_labels)
        #iscrowd = torch.zeros((len(labels),), dtype=torch.uint8)
        iscrowd =torch.tensor(iscrowd_list, dtype=torch.int64)
        id = torch.tensor([idx])
        #data_id = torch.tensor(int(name.split("img")[1].split(" ")[0]))


        # new pytorch api, still sucks
        #boxes = dp.BoundingBox(boxes,format="XYXY", spatial_size=(img.size[-1], img.size[0]))
        #img = dp.Image(img)

        """# TODO : remove this after finishing testing on blacked out clusters
        if torch.max(labels).item() > 1:
            box_ind = np.where(labels.numpy() == 2)
            cl_boxes = boxes.numpy()[box_ind]
            img = np.array(img)
            for k in range(len(cl_boxes)):
                cbox = cl_boxes[k]
                cbox = cbox.astype(np.int32)
                img[cbox[1]:cbox[3], cbox[0]:cbox[2]] = [0,0,0] # indexed correctly?
            boxes = torch.tensor(np.delete(boxes.numpy(), box_ind, axis=0))
            torch.tensor(np.delete(labels.numpy(), box_ind, axis=0))
            img = Image.fromarray(img)
        # TODO: up to here"""

        # suppose all instances are not crowd
        image = img
        target = {"boxes": boxes, "labels":labels, "image_id": id, "area":area, "iscrowd":iscrowd}



        # custom data augmentation:
        if self.train == True:
            #data augm:

            # this used the custom tranformation functions, pytorch released official ones by now, using them now
            image, target = td.MyToTensor(img, target)
            image, target = td.MyRandomVericalFlip(image, target, p=0.2)
            image, target = td.MyRandomVericalFlip(image, target, p=0.2)

            A = v2.RandomPhotometricDistort()
            image = A(image)

            if np.random.random() > 0.7:
                B = v2.GaussianBlur(3, (0.1, 2))
                image = B(image)

            if np.random.random() > 0.7:
                bits = np.random.randint(5,8)
                C = v2.RandomPosterize(bits, p= 0.2)
                image = C(image)

            # plt.imshow(image.permute(1,2,0))
            # plt.show()
            a = 2
            # is torchvision over v.0.16 ?
            #TODO: new v2 transform pipeline


        elif self.train == False:
            image, target = td.MyToTensor(img, target)
        return image, target

    def __len__(self):
        return len(self.img_names)


def get_transform(train):

    transforms = []
    #transforms.append(T.ToImageTensor())
    #transforms.append(T.ConvertImageDtype())
    if train:
        p = 1
        #transforms.append(T.RandomHorizontalFlip(0.5))
        #transforms.append(T.RandomPhotometricDistort(contrast=1.2))
        #transforms.append(T.MyRandomVerticalFlip(0.7))
       #transforms.append(T.RandomHorizontalFlip(0.5))
       #transforms.append(T.RandomPhotometricDistort())
    #transforms.append(T.ToDtype(torch.float))


    return T.Compose(transforms)


def get_model_object_detection(num_classes,model_arch = 50, **kwargs):

    if model_arch == 50:
        # load an object detection model pre-trained on COCO
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, trainable_backbone_layers = 3, **kwargs)

        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # now get the number of input features for the mask classifier

        anchor_generator = AnchorGenerator(
            sizes=((8,), (16,), (32,), (64,), (128,), (256,), (512,)),
            aspect_ratios=tuple([(0.5, 1.0, 2.0) for _ in range(5)]))
        model.rpn.anchor_generator = anchor_generator
        # 256 because that's the number of features that FPN returns
        model.rpn.head = RPNHead(256, anchor_generator.num_anchors_per_location()[0])


    elif model_arch == 18:
        # load a different backbone
        # backbone = torchvision.models.mobilenet_v2(weights="DEFAULT").features
        # backbone = torchvision.models.resnet18(weights="IMAGENET1K_V1")
        backbone = resnet_fpn_backbone('resnet18', pretrained=True)
        #backbone = BackboneWithFPN(backbone,return_layers=1)

        anchor_generator = AnchorGenerator(sizes=((16, ), (32,), (64,), (128,), (256,), (516,)),
                                           aspect_ratios=((0.5, 1.0, 2.0)))

        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=["0"],
                                                        output_size=7,
                                                        sampling_ratio=2)

        mask_roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=["0"],
                                                             output_size=14,
                                                             sampling_ratio=2)

        # put the pieces together inside a FasterRCNN model
        model = FasterRCNN(backbone,
                         num_classes=num_classes,
                         rpn_anchor_generator=anchor_generator,
                         box_roi_pool=roi_pooler,
                         mask_roi_pool=mask_roi_pooler)

    elif model_arch == 101:
        backbone = resnet_fpn_backbone('resnet101', pretrained=True)
        # backbone = BackboneWithFPN(backbone,return_layers=1)

        anchor_generator = AnchorGenerator(sizes=((32,), (64,), (128,), (256,), (516,)),
                                           aspect_ratios=((0.5, 1.0, 2.0)))

        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=["0"],
                                                        output_size=7,
                                                        sampling_ratio=2)

        mask_roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=["0"],
                                                             output_size=14,
                                                             sampling_ratio=2)

        # put the pieces together inside a FasterRCNN model
        model = FasterRCNN(backbone,
                           num_classes=num_classes,
                           rpn_anchor_generator=anchor_generator,
                           box_roi_pool=roi_pooler,
                           mask_roi_pool=mask_roi_pooler)

    return model

#OLD
"""def get_model_object_detection(num_classes, **kwargs):
    # load a pretrained faster rcnn model (fpn helps with detecting small objects)
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights= "DEFAULT", **kwargs)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    #in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    #hidden_layer = 256
    # and replace the mask predictor with a new one
    #model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,hidden_layer,num_classes)

    return model"""

"""def get_model_object_detection(num_classes):
    model = torchvision.models.detection.retinanet_resnet50_fpn_v2(num_classes = 3, weights = None, pretrained_backbone = True)

    return model""" # retinanet, doesnt work yet


if __name__ == '__main__':

    # Enter name to save weights, losses and metrics as (e.g #train_#val_#epochs)                                    ***
    # Also add the interval for extracting data to csv (from engine.py)
    name = "v14_lowioumetrics"  # "numtrain_numeval_epochs_batchsize_other"
    num_train = 1500
    num_eval = 500
    num_epochs = 50
    batch_size = 2

    Data_Path_tr = r"D:\Trainingsdaten_v07\Trainingsdaten_v07\crop"
    Data_JSON_tr = "labels_croptrain_v07.json"
    Data_Path_val = r"D:\Trainingsdaten_v06d\Validationsdaten_v06\crop"
    Data_JSON_val = "labels_cropval_v06_cropd.json"

    """np.random.seed(420)
    random.seed(420)
    torch.manual_seed(420)"""

    hp_str = f"num_train: {num_train}\nnum_eval: {num_eval}\nnum_epochs: {num_epochs}\nbatch_size: {batch_size}" \
             f"\nData_Path_tr: {Data_Path_tr}\nJSON_tr: {Data_JSON_tr}"

    loss_exp = []
    loss_classifier_exp = []
    loss_box_reg_exp = []
    loss_mask_exp = []
    loss_objectness_exp = []
    loss_rpn_box_reg_exp = []

    precision_recall_bbox = []
    precision_recall_segm = []

    save_path = os.path.join("Benchmarks", name)
    try:
        print("Creating Benchmarks (name) folder")
        os.mkdir(save_path)
    except:
        print(f"folder {save_path} exists, overwriting old benchmarks")

    def main(precision_recall_bbox, hp_str):
        # train on the GPU or on the CPU, if a GPU is not available
        global losses_needed, accuracy_needed, metrics_needed, loss_complete, dataset_test
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # our dataset has 2 (0-1) classes: [Pit, Cluster]
        num_classes = 3
        # use our dataset and defined transformations
        #dataset = Pits_Dataset("Data", "box_data.json", transforms=get_transform(True))
        dataset_train = Pits_Dataset(Data_Path_tr,
                               datafile=Data_JSON_tr,train=True,
                               transforms=get_transform(train=True))

        dataset_test = Pits_Dataset(Data_Path_val,
                               datafile=Data_JSON_val,train=False,
                               transforms=get_transform(train=False))
        # create a subset only containing data that is already labeled:
        """good_ind = []
        for i in range(len(dataset)):
            if len(dataset[i][1]["boxes"]) > 0:
                good_ind.append(i)
        # this is the dataset only containing labeled gt data
        dataset = torch.utils.data.Subset(dataset, good_ind) # this is the dataset only containing labeled gt data"""

        # split the dataset in train and test set
        """indices = good_ind
        random.shuffle(indices)"""
        indices = torch.randperm(len(dataset_train))
        #dataset_train = torch.utils.data.Subset(dataset_train, indices[:num_train])
        #dataset_test = torch.utils.data.Subset(dataset_test, indices[num_train:num_eval+num_train])
        #print(dataset_test.dataset.img_names[dataset_test.indices])
        # define training and validation data loaders
        data_loader = torch.utils.data.DataLoader(
            dataset_train, batch_size=batch_size, shuffle=True, num_workers=0,
            collate_fn=utils.collate_fn)
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=1, shuffle=False, num_workers=0,
            collate_fn=utils.collate_fn)

        # get the model using our helper function
        model = get_model_object_detection(num_classes,
                                           )
        # things that seemed to improve:
        # box_detections_per_img = 300


        # move model to the right device
        model.to(device)

        weight_decay = 0.00025
        lr = 0.008
        momentum = 0.9

        step_size = 5
        gamma = 0.2

        # construct an optimizer
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, weight_decay=weight_decay, lr=lr, momentum = momentum)

        # and a learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                       step_size=step_size,
                                                       gamma=gamma)

        hp_str = hp_str + f"\nweight_decay: {weight_decay}\nlr: {lr}\nmomentum: {momentum}\nstep_size: {step_size}\ngamma: {gamma}\ndevice: {device}"
        complete_dict = {}
        bbox_complete = []
        segm_complete = []
        mean_metrics = []
        # resume training from an earlier set of weights (evtl. map_location)                                 ***
        #model.load_state_dict(torch.load(r'Benchmarks/v07_crop_550_70_60_hvflip_fasterparams/Weights/weights_v07_crop_550_70_60_hvflip_fasterparams_MAX_METRICS.pth'))

        try:
            print("Creating Weights folder")
            os.mkdir(os.path.join(save_path, "Weights"))
        except:
            print("Weights folder already exists")

        for epoch in range(num_epochs):

            losses_needed = train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=1)
            batch_dict = losses_needed[1]

            # compare the dict of the current epoch with the one of all past epochs and accumulate all values
            # the dict of the current epoch is created in train_one_epoch from engine.py
            for key, value in batch_dict.items():
                if key in complete_dict:
                    if isinstance(complete_dict[key], list):
                        for i in range(len(batch_dict[key])):
                            complete_dict[key].append(batch_dict[key][i])
                            # print("losses zugefügt, bestehende keys und war liste")
                    else:
                        temp_list = [complete_dict[key]]
                        temp_list.append(value)
                        complete_dict[key] = temp_list
                        # print("losses zugefügt, mit temp_list")
                else:
                    complete_dict[key] = value
                    # print("losses zugefügt, neuer key")

            # update the learning rate
            lr_scheduler.step()
            # evaluate on the test dataset
            # evaluate(model, data_loader_test, device=device)
            # Save accuracies
            cocoeval = evaluate(model, data_loader_test, device=device)

            bbox_complete.extend([cocoeval.coco_eval["bbox"].stats])
            #mean_metrics.append(np.mean(cocoeval.coco_eval["bbox"].stats[cocoeval.coco_eval["bbox"].stats > -1]))
            mean_metrics.append(cocoeval.coco_eval["bbox"].stats[1]) # takes AP50 as metric

            if (epoch+1) % 50 == 0 or epoch+1 == num_epochs:
                weight_path = os.path.join(save_path, f'Weights\weights_{name}_e{epoch+1}.pth')
                torch.save(model.state_dict(), weight_path)
                print(f"Saved weights @ epoch {epoch+1}")
            elif epoch > 2 and np.argmax(mean_metrics) == epoch:
                weight_path = os.path.join(save_path, f'Weights\weights_{name}_MAX_METRICS.pth')
                torch.save(model.state_dict(), weight_path)
                print(f"Saved max weights @ epoch {epoch}")





        print("That's it!")

        return complete_dict, bbox_complete, hp_str, mean_metrics

        # Training function main()
    (loss_complete, precision_recall_bbox, hp_str, mean_metrics) = main(precision_recall_bbox, hp_str)

    losses = pd.DataFrame.from_dict(loss_complete)
    # write data to csv


    losses_path = os.path.join(save_path, f"losses_{name}.csv")
    losses.to_csv(losses_path)
    print("losses erstellt")

    prec_reca_bbox = pd.DataFrame(precision_recall_bbox, columns= ['prec_bbox_IoU_avg_areaall_maxdets100',
                                                                  'prec_bbox_IoU_05_areaall_maxdets100',
                                                                  'prec_bbox_IoU_095_areaall_maxdets100',
                                                                  'prec_bbox_IoU_avg_areasmall_maxdets100',
                                                                  'prec_bbox_IoU_avg_areamedium_maxdets100',
                                                                  'prec_bbox_IoU_avg_arealarge_maxdets100',
                                                                  'reca_bbox_IoU_avg_areaall_maxdets1',
                                                                  'reca_bbox_IoU_avg_areaall_maxdets10',
                                                                  'reca_bbox_IoU_avg_areaall_maxdets100',
                                                                  'reca_bbox_IoU_avg_areasmall_maxdets100',
                                                                  'reca_bbox_IoU_avg_areamedium_maxdets100',
                                                                  'reca_bbox_IoU_avg_arealarge_maxdets100',
                                                                'reca_bbox_IoU_05_areaall_maxdets100'])

    bbox_met_path = os.path.join(save_path, f"precision_recall_bbox_{name}.csv")
    prec_reca_bbox.to_csv(bbox_met_path)

    if num_epochs > 5:
        a = np.array(mean_metrics)
        top5_ind = np.argpartition(a, -5)[-5:]
        top5 = a[top5_ind]
        hp_str = hp_str + f"\nTop 5 epochs with max metrics: epochs {top5_ind} \nwith AP50 {top5}\n\nnotes: "

    with open(os.path.join(save_path, f"HP_{name}"), "w") as file:
        file.write(hp_str)



