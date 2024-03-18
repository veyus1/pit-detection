import os
import pandas as pd
from PIL import Image
import torch
import utils
from matplotlib import pyplot as plt
import numpy as np
from Old_scripts.Train_Pits_FasterRCNN_21122023 import get_transform, Pits_Dataset

#TODO: cut images in 512 by 423 crops. donde. convert and save box data in JSON. done.
# Just need to save crop images properly! -> Train w. crops (also need to adjyust inference later)

def cut_target(target, minx, maxx, miny, maxy):
    """using a single crop of an image (det. by min/max k and i) and the target of the original (full-sized) image
       box coordinates (and labels) are calculated in crop frame"""
    boxes_img = torch.tensor(())
    box_abs = torch.tensor(())
    labels_imgl = []
    c = 0
    for box, lab in zip(target["boxes"], target["labels"]):
        # box - [x1,y1,x2,y2] in coordinates relative to current crop

        # top left and bottom right box corners in crop coordinates
        x1 = box[0].item() - minx
        y1 = box[1].item() - miny
        x2 = box[2].item() - minx
        y2 = box[3].item() - miny

        w = x2-x1
        h = y2-y1

        # corner 3 is bottom left, corner 4 is top right
        x3 = x1
        y3 = y2
        x4 = x2
        y4 = y1

        is1in = (0 < x1 < 512 and 0 < y1 < 432)
        is2in = (0 < x2 < 512 and 0 < y2 < 432)
        is3in = (0 < x3 < 512 and 0 < y3 < 432)
        is4in = (0 < x4 < 512 and 0 < y4 < 432)

        if is1in and is2in and is3in and is4in:
            box_abs = torch.tensor([x1,y1,x2,y2])

        elif is1in and not is2in:
            if not is3in and not is4in:
                box_abs = torch.tensor([x1, y1, 512, 432])
            elif is3in and not is4in:
                box_abs = torch.tensor([x1, y1, 512, y2])
            elif is4in and not is3in:
                box_abs = torch.tensor([x1, y1, x2, 432])

        elif is2in and not is1in:
            if not is3in and not is4in:
                box_abs = torch.tensor([0, 0, x2, y2])
            elif is3in and not is4in:
                box_abs = torch.tensor([x1, 0, x2, y2])
            elif is4in and not is3in:
                box_abs = torch.tensor([0, y1, x2, y2])

        elif is3in and not is4in and not is2in and not is1in:
            box_abs = torch.tensor([x1, 0, 512, y2])
        elif is4in and not is3in and not is2in and not is1in:
            box_abs = torch.tensor([0, y1, x2, 432])

        elif not is1in and not is2in and not is3in and not is4in:
            box_abs = torch.tensor(())

        if len(box_abs) > 0:
            labels_imgl.append(lab.item())
            c += 1
            boxes_img = torch.cat((boxes_img, box_abs), 0)


    #print(f"{c} boxes")
    boxes_img = boxes_img.view((-1, 4))
    labels_img = torch.tensor(labels_imgl)
    return boxes_img, labels_img


dataset = Pits_Dataset(r"D:\Trainingsdaten_v07\Trainingsdaten_v07\full",
                               datafile="labels_full_v07.json",train=False,
                               transforms=get_transform(train=False))

#dataset = torch.utils.data.Subset(dataset, range(20,50))



data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=utils.collate_fn)

pred_list = []
count = 0
for item in data_loader:
    #ind = np.random.randint(0,len(data_loader)-1)
    #n = 52
    name = dataset.data["names"][item[1][0]["image_id"].item()]
    name = name.split(sep = ".")[0]
    """if name != "S36C2_0020000_B00001":
        continue"""


    im = item[0][0]
    data = item[1][0]
    labs = data["labels"]

    maxx = 0
    maxy = 0

    """plt.figure(1)
    imc = (im * 255).to(dtype=torch.uint8)
    plt.imshow(np.transpose(imc, (1, 2, 0)))
    plt.show()"""


    for i in range(int(2560/512)):
        maxx = maxx + 512
        minx = maxx - 512
        maxy = 0
        for k in range(int(2160/432)):

            plt.title(f"{name}_r{k}_c{i}")
            maxy = maxy + 432
            miny = maxy - 432

            curr_boxes, curr_labels = cut_target(data, minx, maxx, miny, maxy) # per crop
            imc_bare = im[:, miny:maxy, minx:maxx]
            imc = (imc_bare * 255).to(dtype=torch.uint8)
            col = []
            for k2 in curr_labels:
                if k2.item() == 1:
                    col.append((255,0,0))
                elif k2.item() == 2:
                    col.append((125,0,255))

            # add folder to save images in if using save_faster_rcnn_predictions
            save_in = r"D:\Trainingsdaten_v07\Trainingsdaten_v07\crop\raw"
            save_json_name = "labels_croptrain_v07.json"
            Image.fromarray(imc.numpy().transpose(1, 2, 0)).save(os.path.join(save_in, f"{name}_row{k}_col{i}.tif"))


            #imcb = draw_bounding_boxes(imc,curr_boxes, colors=col)
            #plt.imshow(np.transpose(imcb, (1, 2, 0)))
            #plt.show()
            pit_boxes = []
            cluster_boxes = []
            if len(curr_labels) > 0:
                for lab, box in zip(list(curr_labels), list(curr_boxes)):
                    if lab.item() == 1:
                        pit_boxes.append(box)
                    elif lab.item() == 2:
                        cluster_boxes.append(box)
            else:
                pit_boxes = []
                cluster_boxes = []


            curr_data = {}
            curr_data["names"] = f"{name}_row{k}_col{i}.tif"

            for class_data, class_name in zip([pit_boxes, cluster_boxes], ["Pit", "Cluster"]):
                if len(class_data) > 0:
                    prepd_data = np.rint([class_data[i2].tolist() for i2 in range(len(class_data))])

                    # remove this if matlab coordinates not needed in JSON:
                    # [x1 y1 x2 y2] to [x1 x2 w h]:
                    for i3, box in enumerate(prepd_data):
                        x1 = box[0]
                        y1 = box[1]
                        w = box[2] - x1
                        h = box[3] - y1
                        prepd_data[i3] = [x1, y1, w, h]

                    curr_data[class_name] = prepd_data
                elif len(class_data) == 0:
                    curr_data[class_name] = []

            pred_list.append(curr_data)

    save_coords_in = save_in.split(sep = "raw")[0]
    pd.DataFrame(pred_list).to_json(os.path.join(save_coords_in,save_json_name),
                                            orient="records")
    count += 1
    print(f"finished saving box data (JSON) and cropped image for {name}. [{count}/{len(dataset)}]")