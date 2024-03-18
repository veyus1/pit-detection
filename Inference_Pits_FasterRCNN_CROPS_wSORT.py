import matplotlib
import numpy as np
import pandas as pd
import torch
from torchvision.ops import nms
from torchvision.utils import draw_bounding_boxes
import utils
from Train_Pits_FasterRCNN import get_transform, get_model_object_detection
import os
from PIL import Image
import time
import transform_data as td
from Old_scripts.SORT import Sort

"""Takes all images from Inference folder, 
creates predictions for raw images and saves the predictions in Predictions folder"""

#TODO: Inference_..._CROPS.py needs to be able to:
# (i) cut an input image (2560x2160) into crops (512x432)
# then (ii) perform inference as usual (on the crops),
# then (iii) retransform boxes from local to global coordinates in the correct way


matplotlib.use('TKagg')
class Pits_Dataset_Inf(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(root)))

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, self.imgs[idx])
        img = Image.open(img_path)
        image_id = torch.tensor([idx])

        target = {"image_id": image_id}

        image, target = td.MyToTensor(img, target)
        return(image, target)

    def __len__(self):
        return len(self.imgs)


def cut_image(image):
    """using a single crop of an image (det. by min/max k and i) and the target of the original (full-sized) image
       box coordinates (and labels) are calculated in crop frame"""
    minx_list = []
    miny_list =  []
    crops_list = []
    maxx = 0
    maxy = 0
    for i in range(int(2560/512)):
        maxx = maxx + 512
        minx = maxx - 512
        maxy = 0
        for k in range(int(2160/432)):
            maxy = maxy + 432
            miny = maxy - 432
            imc_bare = image[:, miny:maxy, minx:maxx] # image is tensor(3,2160, 2560)

            # check if bare images are normalized or standard uint8 format
            #imc = (imc_bare * 255).to(dtype=torch.uint8)

            minx_list.append(minx)
            miny_list.append(miny)
            crops_list.append(imc_bare)
    return crops_list, minx_list,  miny_list

def retransform_boxes(boxes_crop, minx, miny):
    """from local (crop) box coordinates to global (full-image) coordinates
    boxes_crop as torch.tensor (N,4), minx and miny int (origin of crop in full image)"""
    box_abs = torch.tensor([])
    boxes_crop_abs = torch.tensor([])

    for ind in range(len(boxes_crop)):
        box = boxes_crop[ind]
        # top left and bottom right box corners in crop coordinates
        x1 = box[0].item() + minx
        y1 = box[1].item() + miny
        x2 = box[2].item() + minx
        y2 = box[3].item() + miny
        box_abs = torch.tensor([x1, y1, x2, y2])
        boxes_crop_abs = torch.cat((boxes_crop_abs, box_abs), 0)

    boxes_crop_abs = boxes_crop_abs.view((-1, 4))
    return boxes_crop_abs


def save_faster_rcnn_predictions_crops(model,dataset, data_loader, save_in_folder, json_name, param_notes,  nms_iou_thresh, device, save_coords_every = 50):
    """this function saves faster rcnn predictions,
        to use enter a dataset (use above class Pits_Dataset_Inf), dataloader
        path to the weight file, num_classes and a folder to save the predicitons to."""
    count = 1
    max_imgs = len(dataset)
    start_time = time.time()

    # init a single online realtime tracker:
    tracked_list = []
    min_hits = 1
    max_age = 4
    iou_threshold_sort = 0.1
    mot_tracker = Sort(min_hits=min_hits, max_age=max_age, iou_threshold=iou_threshold_sort)


    param_notes += f"SORT params: min_hits = {min_hits}, max_age = {max_age}, iou_threshold_sort = {iou_threshold_sort}\n" \
                  f"Saved in: {save_in_folder}\n" \
                  f"JSON name: {json_name}\n" \
                  f"Device: {device}\n\n" \
                  f"Nms_threshold: {nms_iou_thresh}\n" \
                  f"Notes:\n" \
                  f"______________________________________________________________________________________\n"

    if os.path.exists(os.path.join(save_in_folder, "preds")) == False:
        os.mkdir(os.path.join(save_in_folder, "preds"))

    pred_list = []
    pred_scores = []
    for images, targets in data_loader:

        images_list_gpu = []
        for b in range(len(images)):
            image = images[b].to(device)
            images_list_gpu.append(image)

        crops, minx_list, miny_list = cut_image(images_list_gpu[0])
        boxes_img_abs = torch.tensor([]).to(device)
        labels_img = torch.tensor([]).to(device)
        scores_img = torch.tensor([]).to(device)
        for c, minx, miny in zip(crops, minx_list, miny_list):
            # perform inference, save box data (local coordinates)
            # print("aho")

            outputs = model([c])  # forward pass durch Modell, zeitaufwändig

            output = outputs[0]
            id = targets[0]["image_id"].item()
            scores = output["scores"]
            boxes = output["boxes"].data
            labels = output["labels"]

            boxes_crop_abs = retransform_boxes(boxes, minx, miny).to(device)
            boxes_img_abs = torch.cat((boxes_img_abs, boxes_crop_abs), 0) # should be Nx4 tensor (float32)
            labels_img = torch.cat((labels_img, labels), 0) # should be N tensor (int64)
            scores_img = torch.cat((scores_img, scores), 0)# should be


        # use non maximum supression (nms) to find boxes/masks with best iou:
        nms_boxes = []
        nms_scores = []
        nms_labels = []
        nms_indices = nms(boxes=boxes_img_abs, scores=scores_img, iou_threshold=nms_iou_thresh)
        for j in nms_indices:
            nms_boxes.append(boxes_img_abs[j])
            nms_labels.append(labels_img[j])
            nms_scores.append(scores_img[j])

        # concatenate all boxes and masks that pass nms threshold criteria (N)
        if len(nms_boxes) > 0:
                boxes = torch.cat(nms_boxes).view(nms_indices.size(dim=0), 4)   # Nx4"""

        # decode labels:
        decoded_labels = []
        box_color = []
        pit_boxes = []
        cluster_boxes = []
        pit_scores = []
        cluster_scores = []
        for lab, sco, box in zip(list(nms_labels), list(nms_scores), list(boxes)):
            lab = lab.item()
            sco = round(sco.item(), 3)
            if lab == 1:
                decoded_labels.append(str(sco))
                box_color.append((255, 0, 0))  # red - pit
                pit_boxes.append(box)
                pit_scores.append(sco)
            elif lab == 2:
                decoded_labels.append(str(sco))
                box_color.append((125, 0, 255))  # purpl - cluster
                cluster_boxes.append(box)
                cluster_scores.append(sco)


        # TODO: Add tracking algo somewhere here (needs score per box), (before saving to file)
        # SORT needs [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...] np array
        # in the end need [[x1,y1,x2,y2,score, id],[x1,y1,x2,y2,score, id],...] or smth like that,
        # or maybe one json with just boxes and one with score and id ?

        sort_dets = np.array([pit_boxes[k].to("cpu").numpy().tolist() + [pit_scores[k]] for k in range(len(pit_boxes))], dtype=np.float32)
        if len(sort_dets) > 0:
            track_data = mot_tracker.update(sort_dets)
        else:
            track_data = np.empty((0,5))
        track_boxes = track_data[:, :4]
        track_ids = track_data[:, 4]
        tracked_list.append(track_data)

        """if moving_average:
            # calculate average apprearance of track ids from 5 prior boxes and filter using that information
            PRIOR_IDS = []
            if len(tracked_list) > 5 and len(tracked_list) < (len(dataset) - 5):
                # perform moving average calculation (5 len)
                tracks_curr = tracked_list[-1]
                #track_ids = tracked_list[-1][:,4]
                tracks_prior = tracked_list[len(tracked_list) - 6:-1]

                [PRIOR_IDS.append(tracks_prior[k][:, 4]) for k in range(len(tracks_prior))]

                acc = np.zeros(len(track_ids))
                for idc in range(len(track_ids)):
                    val = track_ids[idc]
                    for idp in range(len(PRIOR_IDS)):
                        if val in PRIOR_IDS[idp]:
                            acc[idc] += 1

                # remove tracks that are inconsistent
                track_ids = track_ids[acc > 3]
                track_boxes = track_boxes[acc > 3]
            #-----------------------------------------------------------------------------------------------------
            """




        pit_boxes_tracked = [torch.tensor(track_boxes.tolist()[k]) for k in range(len(track_boxes))]

        boxes_pc = track_boxes.tolist() + cluster_boxes
        boxes_t = torch.tensor(boxes_pc)
        labels = [str(track_ids[i]) for i in range(len(track_ids))]
        col = []
        labs = []
        for i in track_ids:
            i = int(i)
            col.append((255, 0, 0))  # tuple(col_list[i])
            labs.append(str(int(i)))
        for i in range(len(cluster_boxes)):
            col.append((125,0,125))
            labs.append(str(0))


        # use torchvision io functions to draw boxes on background image
        img = (images_list_gpu[0] * 255).to(dtype=torch.uint8)
        img_name = dataset.dataset.imgs[id]
        img_name = img_name.split(sep=".")[0]
        format = "tif"

        # TODO need to adapt, so that this saves correctly with (score n track id) boxes
        if len(boxes_t) != 0:
                boxed_image = draw_bounding_boxes(img, boxes_t, colors=col, labels=labs,
                                              font=r"C:\Users\veyse\Desktop\font\arial")  # optionally add labels here (decoded_labels)
                save_path = os.path.join(save_in_folder,"preds", str("pred_"+ img_name+"." + format))

                try:
                    os.mkdir(os.path.join(save_in_folder,"preds"))
                except:
                    a = 2

                #save_path_ov = os.path.join("Inference/raw_vis", str(img_name + "." + format))
                boxed_img_np = np.array(boxed_image.permute(1,2,0))
                #plt.imsave(save_path,im_form,format= format )
                # convert image to preferred format (P, L, RGB...) and save in specified folder/naem:
                boxed_img_I = Image.fromarray(boxed_img_np).convert("RGB").save(save_path)

                # export current data as dict
                curr_data = {}
                curr_scores = {}
                curr_data["names"] = img_name + ".tif"

                for class_data, class_name in zip([pit_boxes_tracked, cluster_boxes], ["Pit", "Cluster"]):
                    if len(class_data) > 0:
                        prepd_data = np.rint([class_data[i].tolist() for i in range(len(class_data))])

                        # remove this if matlab coordinates not needed in JSON:
                        # [x1 y1 x2 y2] to [x1 x2 w h]:
                        for i, box in enumerate(prepd_data):
                            x1 = box[0]
                            y1 = box[1]
                            w = box[2] - x1
                            h = box[3] - y1
                            prepd_data[i] = [x1, y1, w, h]

                        curr_data[class_name] = prepd_data
                    elif len(class_data) == 0:
                        curr_data[class_name] = []

                curr_scores["pit_scores"] = pit_scores
                curr_scores["cluster_scores"] = cluster_scores

                pred_list.append(curr_data)
                pred_scores.append(curr_scores)

                curr_df = pd.DataFrame(pred_list)

                print(f"[{count}/{max_imgs}] pred for {img_name} saved, found {len(boxes_t)} tracked objects. {len(pit_boxes_tracked)} pits and {len(cluster_boxes)} clusters.")
                param_notes += f"[{count}/{max_imgs}] pred for {img_name} saved, found {len(boxes)} objects. {len(pit_boxes_tracked)} pits and {len(cluster_boxes)} clusters.\n"

        else:
                # save input image
                boxed_image = img.repeat(3, 1, 1).to("cpu")
                save_path = os.path.join(save_in_folder, "preds", str("pred_" + img_name + "." + format))
                boxed_img_np = np.array(boxed_image.permute(1, 2, 0))
                # plt.imsave(save_path,im_form,format= format )
                # convert image to preferred format (P, L, RGB...) and save in specified folder/naem:
                boxed_img_I = Image.fromarray(boxed_img_np).convert("RGB").save(save_path)

                curr_data = {}
                curr_data["names"] = img_name + ".tif"
                curr_data["Pit"] = []
                curr_data["Cluster"] = []
                pred_list.append(curr_data)
                curr_df = pd.DataFrame(pred_list)

                print(f"[{count}/{max_imgs}] no bounding boxes found for {img_name}")
                param_notes += f"[{count}/{max_imgs}] no bounding boxes found for {img_name}\n"



        if count % save_coords_every == 0 or count == max_imgs:
            # save prediction coordinates in JSON every 10 images
            curr_df.to_json(os.path.join(save_in, json_name),
                                             orient="records")

            print(f"[{count}/{max_imgs}] updated JSON file")

        count += 1


        # comment out to see every prediction before saving it
        # plt.title(f"model prediction masks for {img_name}")
        # plt.show()"""

    end_time = time.time()
    print(f"total inference time for {max_imgs} images: {(end_time - start_time):.4f} seconds")
    param_notes += f"\n\ntotal inference time for {max_imgs} images: {(end_time - start_time):.4f} seconds"
    with open(os.path.join(save_in_folder, f"Inference_Params"), "w") as file:
        file.write(param_notes)

    mot_tracker = None
    return tracked_list







if __name__ == '__main__':

    # input root folder, NOTICE: input images full size (2560x2160):
    names_list = os.listdir(r"F:\Jonas\24\preds_Cav3\tif")

    for sample_name in names_list: #os.listdir(r"D:\DatenAuswertung_180124\Cavi2\tif\good"):  #["S58_C1_01"]:
        print(f"NOW DOING SAMPLE: {sample_name}")
        moving_average = False
        #sample_name = "S32_02"
        root = os.path.join(r"F:\Jonas\24\preds_Cav3\tif", sample_name)
        # add folder to save images in
        save_in = os.path.join(r"F:\Jonas\24\preds_Cav3\predictions", sample_name) # change this if different weights
        json_name = sample_name + ".json"

        weight_path = r"Benchmarks/v12_final/Weights/weights_v12_final_e50.pth"  # BUT: weights for model trained on crops (512x432)


        try:
            print(f"Trying to create folder to save in:\n {save_in}")
            os.mkdir(save_in)
        except:
            print("Folder already exists")

        # create dataset
        param_notes = f"Weights: {weight_path}\n"
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu' )
        num_classes = 3
        batch_size = 1
        dataset = Pits_Dataset_Inf(root, get_transform(train=False))  # use root folder
        indices = list(range(len(dataset)))
        #indices = torch.randperm(len(dataset)).tolist()
        dataset = torch.utils.data.Subset(dataset, indices[:])
        #dataset = dataset_test
        data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                collate_fn=utils.collate_fn)

        full_coords = []

        #inference
        with torch.no_grad():


            model = get_model_object_detection(num_classes = num_classes, box_score_thresh = 0.47) #, box_nms_thresh=0.05, box_score_thresh=0.55)
            param_notes += f"Faster RCNN params:\ndets_per_img: {model.roi_heads.detections_per_img}\nnms_thresh: {model.roi_heads.nms_thresh}\nscore_thresh: {model.roi_heads.score_thresh}\nmoving average filtering: {moving_average}\n(add other params as needed)"

            if device == 'cuda':
                model.load_state_dict(torch.load(weight_path))
            else:
                model.load_state_dict(torch.load(weight_path, map_location='cpu')) # map location to run gpu wights on cpu
            model.to(device)
            model.eval()

            #plot_faster_rcnn_predictions(model, dataset, data_loader)
            tracked_list = save_faster_rcnn_predictions_crops(model, dataset, data_loader, save_in, json_name, param_notes, nms_iou_thresh=0.35, device=device, save_coords_every=30)



            # PERSISTENCE:
            if len(tracked_list) > 5:

                final_5ids = np.unique(np.concatenate([tracked_list[len(tracked_list)-5:len(tracked_list)][k][:,4] for k in range(5)]))

                first_app = np.ones(int(len(final_5ids)))*-1
                last_app = np.zeros(int(len(final_5ids)))
                num_app = np.zeros(int(len(final_5ids)))

                # preparation for persistence analysis:
                for im_id in range(len(dataset)):
                    for track_id in range(len(final_5ids)):
                        track_id = int(track_id)
                        # check if track in current image
                        if int(final_5ids[track_id]) in tracked_list[im_id][:, 4]:

                            # count number of times each id appears:
                            num_app[track_id] += 1

                            # check if this is first appearance of id
                            if first_app[track_id] == -1:
                                first_app[track_id] = im_id

                            # check if this is (up to now) last appearance of id
                            if last_app[track_id] < im_id:
                                last_app[track_id] = im_id


                avr_app = num_app / (last_app - first_app+1)

                # percent appearances format: track_id; first_app, last_app, num_app, avr_app)
                percent_appearances = np.array([final_5ids, first_app, last_app, num_app, avr_app])

                passed_boxes_list = []
                passed_ids_list = []
                num_pits = []
                img_names_list = []
                pred_list = []


                per_dir = os.path.join(save_in, "persistence")
                try:
                    print(f"Trying to create folder to save in:\n {per_dir}")
                    os.mkdir(per_dir)
                except:
                    print("Folder already exists")

                # starting persistence analysis
                app_thresh = 0.83
                param_notes += f"threshold for appearance of track over its lifetime: {app_thresh}\n"
                for im_ind in range(len(dataset)):
                    curr_tracks = tracked_list[im_ind]
                    curr_ids = curr_tracks[:,4]
                    acc = np.zeros(len(curr_ids))
                    for track_id in range(len(curr_ids)):

                        # check if id exists at end of sequence & how consistently a track appears
                        try:
                            id_in_percent_array = np.where(percent_appearances[0, :] == curr_ids[track_id])[0].item()
                        except:
                            id_in_percent_array = -1

                        if curr_ids[track_id] in final_5ids:
                            if id_in_percent_array > -1 and percent_appearances[4,id_in_percent_array] > app_thresh:
                                acc[track_id] = 1


                    passed_tracks = curr_tracks[acc > 0]
                    passed_boxes_list.append(passed_tracks[:, :-1])
                    passed_ids_list.append(passed_tracks[:, -1])

                    # take prev boxes if a track doesn't exist at this timestep
                    # TODO: change the -2 to -3 to pred two images prior
                    # TODO: for S51, is one image missing? passed ids len = 49?
                    if im_ind > 2:
                        for idc in range(len(passed_ids_list[-1])):
                            if passed_ids_list[-1][idc] not in passed_ids_list[-2]:
                                passed_ids_list[-2] = np.append(passed_ids_list[-2], passed_ids_list[-1][idc])
                                passed_boxes_list[-2] = np.append(passed_boxes_list[-2], np.array(passed_boxes_list[-1][idc]).reshape((1,4)), axis = 0)

                    num_pits.append(len(passed_tracks[:, -1]))
                    img_names_list.append(dataset.dataset.imgs[im_ind])

        # export current data as dict
        for ii, (namei, pitdatai, track_idi) in enumerate(zip(img_names_list, passed_boxes_list, passed_ids_list)):
            curr_data = {}
            curr_scores = {}
            curr_data["names"] = namei

            # TODO: get this to work with cluster boxes aswell (not tracked currently, thus not exported from save... function)
            boxes_t = torch.tensor(pitdatai)
            cluster_boxes = []


            for class_data, class_name in zip([list(boxes_t), cluster_boxes], ["Pit", "Cluster"]):
                if len(class_data) > 0:
                    prepd_data = np.rint([class_data[i].tolist() for i in range(len(class_data))])

                    # remove this if matlab coordinates not needed in JSON:
                    # [x1 y1 x2 y2] to [x1 x2 w h]:
                    for i, box in enumerate(prepd_data):
                        x1 = box[0]
                        y1 = box[1]
                        w = box[2] - x1
                        h = box[3] - y1
                        prepd_data[i] = [x1, y1, w, h]

                    curr_data[class_name] = prepd_data
                elif len(class_data) == 0:
                    curr_data[class_name] = []

            pred_list.append(curr_data)

            if ii % 10 == 0 or ii+1 == len(passed_boxes_list):
                curr_df = pd.DataFrame(pred_list)
                curr_df.to_json(os.path.join(save_in, "persistence_boxdata_" + json_name), orient="records")

            n_diff = [0] + [num_pits[k] - num_pits[k - 1] for k in range(1, len(num_pits))]
            #npos_diff = [0 if i < 0 else i for i in n_diff]
            num_pits_0 = [num_pits[k] - num_pits[0] for k in range(len(num_pits))]

            #n_diff.append(np.mean(n_diff))
            #npos_diff.append(np.mean(npos_diff))

            # for plotting boxes
            img = (dataset[ii][0] * 255).to(dtype=torch.uint8)
            lab = [str(int(passed_ids_list[ii][k])) for k in range(len(passed_ids_list[ii]))]
            boxed_image = draw_bounding_boxes(img, boxes_t, colors=[(255, 0, 0)] * len(boxes_t), labels=lab,
                                              font=r"C:\Users\veyse\Desktop\font\arial")

            img_name = namei.split(sep=".")[0]
            format = "tif"

            save_path = os.path.join(per_dir, str("per_" + img_name + "." + format))
            boxed_img_np = np.array(boxed_image.permute(1, 2, 0))
            # convert image to preferred format (P, L, RGB...) and save in specified folder/naem:

            # squeeze if gray, i.e. no boxes
            if boxed_img_np.shape[-1] == 1:
                boxed_img_np = boxed_img_np.squeeze()

            boxed_img_I = Image.fromarray(boxed_img_np).convert("RGB").save(save_path)


        pitting_rate_data = pd.DataFrame([dataset.dataset.imgs, num_pits, num_pits_0])
        pitting_rate_data.transpose().to_excel(
                os.path.join(save_in, "persistence_numpits_" + json_name.split(sep=".")[0] + ".xlsx"),
                header=["names", "num_pits", "num_pits - n0"])

    print("Finished!")









r"""# for plotting boxes
img = (dataset[im_ind][0] * 255).to(dtype=torch.uint8)
boxes_t = torch.tensor(passed_tracks[:, :-1])
lab = [str(int(passed_tracks[:, 4].tolist()[k])) for k in range(len(passed_tracks[:, 4].tolist()))]
boxed_image = draw_bounding_boxes(img, boxes_t, colors=[(255,0,0)]*len(boxes_t), labels= lab, font= r"C:\Users\veyse\Desktop\font\arial"),
img_name = dataset.dataset.imgs[im_ind]
img_name = img_name.split(sep=".")[0]
format = "tif"

# TODO need to adapt, so that this saves correctly with (score n track id) boxes
if len(boxes_t) != 0:
    save_path = os.path.join(per_dir, str("per_" + img_name + "." + format))
    boxed_img_np = np.array(boxed_image[0].permute(1, 2, 0))
    # convert image to preferred format (P, L, RGB...) and save in specified folder/naem:
    boxed_img_I = Image.fromarray(boxed_img_np).convert("RGB").save(save_path)"""








