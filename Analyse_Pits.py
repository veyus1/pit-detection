import copy
import os
import sklearn.neighbors as skcl
import scipy.ndimage
import torchvision.ops
from PIL import Image
import torch
import utils
from skimage.measure import EllipseModel
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import RadiusNeighborsClassifier
from matplotlib import pyplot as plt
from PIL import ImageDraw, ImageFont
from torchvision.utils import draw_bounding_boxes
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcol
import matplotlib.cm as cm
from PIL import Image
from torchvision.ops import box_convert, box_iou
import cv2
import glob
from skimage import transform
from scipy.spatial import ConvexHull
import xlsxwriter
# this was used to create center of box and track jittering over sequence

"""root = "F:\Veysel\DatenAuswertung\S5x"
dirs = os.listdir(os.path.join(root, "tif\good"))

#this takes a set of sequences and creates the track using center of boxes (might need changes for )
# per sequence k
for k in range(len(dirs)):

    data_path = os.path.join(root,"finalpreds", dirs[k])
    df = pd.read_json(os.path.join(data_path,"persistence_boxdata_"+f"{dirs[k]}.json"))
    images_list = os.listdir(os.path.join(root,"tif\good", dirs[k]))


    # Make a user-defined colormap.
    cb_r = np.linspace(255,0,len(df))
    cb_g = np.linspace(100,0,len(df))
    cb_b = np.linspace(0,255,len(df))

    jitter_img = np.ones((3,2160,2560), dtype=np.uint8)*255


    # per img
    c = 0
    for img_name in df["names"]:

        color = int(cb_r[c]), int(cb_g[c]), int(cb_b[c])

        im_data = df.loc[df["names"] == img_name]


        for box in list(im_data["Pit"])[0]:
            x = int(box[0] + box[2]/2)
            y = int(box[1] + box[3]/2)

            jitter_img[:,y,x] = color

        c += 1

    pil_img = Image.fromarray(np.transpose(jitter_img, (1,2,0)))
    pil_img.save(os.path.join(r"F:\Veysel\DatenAuswertung\S5x\tracking",f"{dirs[k]}.tif"))
"""


# this was used to calculate center of box and track jittering over sequence for preds and targets
"""root = "F:\DatenAuswertung_final\S5x"
dirs = os.listdir(os.path.join(root, "tif\good"))



#this takes a set of sequences and creates the track using center of boxes (might need changes for )
# per sequence k
for k in range(len(dirs)):

    data_path = os.path.join(root,"finalpreds", dirs[k])
    df = pd.read_json(os.path.join(data_path,"persistence_boxdata_"+f"{dirs[k]}.json"))

    try:
        df_gt = pd.read_json(rf"F:\DatenAuswertung_final\S5x\labels\{dirs[k]}\labels_{dirs[k]}.json")

    except:
        print(f"no targets for: {dirs[k]}")
        im_data_gt = None

    # df_pred = pd.read_json(rf"F:\Veysel\DatenAuswertung\S5x\finalpreds\{name}\persistence_boxdata_{name}.json")
    # df_pred = pd.read_json(rf"F:\Veysel\DatenAuswertung\S5x\finalpreds\{name}\{name}.json") # pure faster

    images_list = os.listdir(os.path.join(root,"tif\good", dirs[k]))


    # Make a user-defined colormap.
    cb_r = np.linspace(255,0,len(df))
    cb_g = np.linspace(100,0,len(df))
    cb_b = np.linspace(0,255,len(df))

    jitter_img = np.ones((3,2160,2560), dtype=np.uint8)*255


    # per img
    c = 0
    for img_name in df["names"]:

        color = int(cb_r[c]), int(cb_g[c]), int(cb_b[c])

        im_data = df.loc[df["names"] == img_name]
        im_data_gt = df_gt.loc[df["names"] == img_name]

        for box in list(im_data["Pit"])[0]:
            x = int(box[0] + box[2]/2)
            y = int(box[1] + box[3]/2)

            jitter_img[:,y,x] = (255,0,0)



        for box in list(im_data_gt["Pit"])[0]:
            if type(box) is int:
                box = im_data_gt["Pit"].item()

            x = int(box[0] + box[2] / 2)
            y = int(box[1] + box[3] / 2)

            if np.sum(jitter_img[:, y, x] == [255, 0, 0]) == 3:
                jitter_img[:, y, x] = (0, 0, 255)
            else:
                jitter_img[:, y, x] = (0, 255, 0)

        c += 1

    pil_img = Image.fromarray(np.transpose(jitter_img, (1,2,0)))
    pil_img.save(os.path.join(r"F:\DatenAuswertung_final\S5x\tracking2\mitoverlap",f"{dirs[k]}.tif"))
"""

# this takes image folder, save in path and dataframe path and saves boxed images (from json file)
r"""
images_path =  r"D:\DCGAN\syn\raw"
save_in = r"D:\DCGAN\syn\boxed"
df = pd.read_json("D:\DCGAN\syn\labels_syn.json")
images_list = os.listdir(images_path)

# per img
c = 0
for img_name in df["names"]:

    img = torch.tensor(np.array(Image.open(os.path.join(images_path, img_name))))

    im_data = df.loc[df["names"] == img_name]

    cl_boxes = im_data["Cluster"].values[0]
    pit_boxes = im_data["Pit"].values[0]
    boxes = cl_boxes + pit_boxes



    if len(boxes) > 0:
        box_t = torch.tensor(np.squeeze(boxes)).view((len(boxes),4))
        box_tc = torchvision.ops.box_convert(box_t, in_fmt="xywh", out_fmt="xyxy")
        bi1 = draw_bounding_boxes(torch.permute(img, (2,0,1)), box_tc, colors= [(125,0, 125)]*len(cl_boxes) + [(20,230, 20)]*len(pit_boxes))

        pil_img = Image.fromarray(np.transpose(bi1, (1,2,0)).numpy())
        pil_img.save(os.path.join(save_in,f"{img_name}"))
"""



# this filters small boxes from a json, saves the filtered json & number of pits per image in xlxs
r"""images_path =  r"F:\DatenAuswertung_180124\Cavi2\tif\good\S51_01"
save_in = r"F:\DatenAuswertung_180124\Cavi2\preds_05_nosmall\S59"
df = pd.read_json(r"F:\DatenAuswertung_180124\Cavi2\preds_05\S59\persistence_boxdata_S59.json")
images_list = os.listdir(images_path)

# per img
num_pits = []
data_new = []
for img_name in df["names"]:
    curr_data = {}
    curr_data["names"] = img_name

    im_data = df.loc[df["names"] == img_name]

    passed_pit_boxes = []

    if len(im_data["Pit"].item()) > 0:
        for pit_box in im_data["Pit"].item():
            if pit_box[2] < 10 and pit_box[3] < 10:
                print(f"smallbox : {pit_box}")
            else:
                passed_pit_boxes.append(pit_box)


    num_pits.append(len(passed_pit_boxes))
    cl_list = im_data["Cluster"].item()

    curr_data["Pit"] = passed_pit_boxes
    curr_data["Cluster"] = cl_list

    data_new.append(curr_data)

df_new = pd.DataFrame(data_new)

df_new.to_json(os.path.join(save_in, "smallfiltered_pers.JSON"),
                                             orient="records")
pd.DataFrame(num_pits).to_excel(os.path.join(save_in, "num_pits.xlsx"))
"""

# this to calculate sqrt(area) of all boxes (ie. a measure for pit size) given a JSON file
r"""
names = os.listdir(r"D:\Daten_Seagate_Sequences_3\3CameraSetup2LST\finalpreds_oldersamples")#["S51_01", "S52", "S53", "S57_C1", "S58_C1_01", "S59"]
data_list = []
names_tot = []
areas_tot = []
for name in names:
    save_in = rf"D:\Daten_Seagate_Sequences_3\3CameraSetup2LST\evaluation_oldersamples"
    df = pd.read_json(rf"D:\Daten_Seagate_Sequences_3\3CameraSetup2LST\finalpreds_oldersamples\{name}\persistence_boxdata_{name}.json")

    # per img
    areas = []
    data = {}
    for img_name in df["names"]:

        im_data = df.loc[df["names"] == img_name]

        pit_datagt = np.array(im_data["Pit"].tolist()).squeeze()

        if pit_datagt.size == 4:
            pit_datagt = np.expand_dims(pit_datagt, 0)

        if pit_datagt.size > 0:
            for pit_box in pit_datagt:
                #area = np.sqrt(pit_box[2]**2+pit_box[3]**2)
                area =pit_box[3]
                areas.append(area)

    names_tot.append(name)
    areas_tot.append(areas)
    #data["names"] = name
    #data["areas"] = areas
    data_list.append(data)


areas_df = pd.DataFrame(areas_tot).transpose()
areas_df.columns = names_tot
areas_df.to_excel(os.path.join(save_in, "width.xlsx"))

"""

# this to calculate diagonal length of all boxes (ie. a measure for pit size) given a JSON file
"""
#names = ["S51_01", "S52", "S53", "S57_C1", "S58_C1_01", "S59"]
names = os.listdir(r"F:\Jonas\24\preds_Cav3\predictions")
data_list = []
names_tot = []
areas_tot = []
for name in names:
    save_in = rf"F:\Jonas\24\preds_Cav3\spatial"
    df = pd.read_json(rf"F:\Jonas\24\preds_Cav3\predictions\{name}\persistence_boxdata_{name}.json")

    # per img
    areas = []
    data = {}
    for img_name in df["names"]:
        #for img_name in ["B00050.tif"]:
        im_data = df.loc[df["names"] == img_name]

        pit_datagt = np.array(im_data["Pit"].tolist()).squeeze()

        if pit_datagt.size == 4:
            pit_datagt = np.expand_dims(pit_datagt, 0)

        if pit_datagt.size > 0:
            for pit_box in pit_datagt:
                area = np.sqrt(pit_box[2]**2 + pit_box[3]**2)
                #area = pit_box[2]*pit_box[3]
                areas.append(area)

    names_tot.append(name)
    areas_tot.append(areas)
    #data["names"] = name
    #data["areas"] = areas
    data_list.append(data)


areas_df = pd.DataFrame(areas_tot).transpose()
areas_df.columns = names_tot
areas_df.to_excel(os.path.join(save_in, "diagonals_newsamples.xlsx"))
"""


# this to calculate mean diagonal of boxes per image & then per sequence
"""
names = ["S51_01", "S52", "S53", "S57_C1", "S58_C1_01", "S59"]
data_list = []
names_tot = []
avrgboxsize_tot = []
for name in names:
    #save_in = rf"F:\DatenAuswertung_180124\Cavi2\labeling\labels"
    #df = pd.read_json(rf"F:\DatenAuswertung_180124\Cavi2\labeling\labels\{name}\labels_{name}.json")

    save_in = "F:\DatenAuswertung_180124\Cavi2\preds_05"
    df = pd.read_json(rf"F:\DatenAuswertung_180124\Cavi2\preds_05\{name}\persistence_boxdata_{name}.json")

    lens_sequence = []
    img_names = []
    data = {}

    # per img

    for img_name in df["names"]:
        box_lens = []
        im_data = df.loc[df["names"] == img_name]

        pit_datagt = np.array(im_data["Pit"].tolist()).squeeze()

        if pit_datagt.size == 4:
            pit_datagt = np.expand_dims(pit_datagt, 0)

        if pit_datagt.size > 0:
            for pit_box in pit_datagt:
                avr_box_len = np.sqrt(pit_box[2]**2 + pit_box[3]**2)
                #area = pit_box[2]*pit_box[3]
                box_lens.append(avr_box_len)
            mean_box_average_len = np.sum(box_lens) / len(box_lens)
        else:
            mean_box_average_len = None
        lens_sequence.append(mean_box_average_len)
        img_names.append(img_name)

    names_tot.append(name)
    avrgboxsize_tot.append(lens_sequence)


    #data["names"] = name
    #data[f"{name}"] = lens_sequence
    #data["img_name"] = img_names
    data_list.append(lens_sequence)


df = pd.DataFrame(data_list).transpose()
df.columns = names_tot
df.to_excel(os.path.join(save_in, "Diag_sqeqlen_time_lab.xlsx"))
"""


# this to plot pred (red) & gt (green) boxes on an image
"""
names = ["S51_01", "S52", "S53", "S57_C1", "S58_C1_01", "S59"]
data_list = []
names_tot = []
avrgboxsize_tot = []

cp_gt = (20,230,20)
cc_gt = (125,0,125)
cp_pred = (230,20,20)
cc_pred = (20,20,230)


for name in os.listdir(rf"E:\DatenAuswertung_final\preds_Cav3"):
    #save_in = rf"F:\DatenAuswertung_180124\Cavi2\labeling\labels"
    #df = pd.read_json(rf"F:\DatenAuswertung_180124\Cavi2\labeling\labels\{name}\labels_{name}.json")

    df_gt = pd.read_json(rf"F:\DatenAuswertung_final\S5x\labels\{name}\labels_{name}.json")

    save_in = r"E:\DatenAuswertung_final\preds_Cav3\boxes_gr\faster"
    df_pred = pd.read_json(rf"E:\DatenAuswertung_final\preds_Cav3\predictions\{name}\persistence_boxdata_{name}.json")
    #df_pred = pd.read_json(rf"F:\Veysel\DatenAuswertung\S5x\finalpreds\{name}\{name}.json") # pure faster


    im_dir = rf"E:\DatenAuswertung_final\preds_Cav3\tif\{name}"

    try:
        save_dir = os.path.join(save_in, f"{name}")
        os.mkdir(save_dir)
    except:
        a = 1


    # per img
    for img_name in df_gt["names"]:
        gt_data = df_gt.loc[df_gt["names"] == img_name]
        pred_data = df_pred.loc[df_pred["names"] == img_name]

        img = torch.tensor(np.array(Image.open(os.path.join(im_dir, img_name)).convert("RGB")))
        img = torch.permute(img, (2, 0, 1))

        for k, c in zip([gt_data, pred_data], [(cp_gt, cc_gt), (cp_pred, cc_pred)]):



            if k["Pit"].size == 4:
                k["Pit"] = np.expand_dims(k["Pit"], 0)
            if k["Cluster"].size == 4:
                k["Cluster"] = np.expand_dims(k["Cluster"], 0)

            if len(k) != 0:
                cl_boxes = k["Cluster"].values[0]
                pit_boxes = k["Pit"].values[0]
                boxes = cl_boxes + pit_boxes
                colors = [c[1]]*len(cl_boxes) + [c[0]]*len(pit_boxes)


                if np.array(boxes).size == 4:
                    boxes = np.expand_dims(np.array(boxes),0)


                if len(boxes) > 0:
                    box_t = torch.tensor(np.squeeze(boxes)).view((len(boxes), 4))
                    box_tc = torchvision.ops.box_convert(box_t, in_fmt="xywh", out_fmt="xyxy")
                    img = draw_bounding_boxes(img, box_tc,
                                              colors=colors)

        pil_img = Image.fromarray(np.transpose(img, (1, 2, 0)).numpy())
        pil_img.save(os.path.join(save_in,name, f"{img_name}"))

"""

# this is to calculate precision and recall (at iou .5) for each image & then each sequence#
r"""
names = ["S51_01", "S52", "S53", "S57_C1", "S58_C1_01", "S59"]
p_seq = []
r_seq = []
names_tot = []
p_full = []
r_full = []

for name in names:
    p_list = []
    r_list = []
    # save_in = rf"F:\DatenAuswertung_180124\Cavi2\labeling\labels"
    # df = pd.read_json(rf"F:\DatenAuswertung_180124\Cavi2\labeling\labels\{name}\labels_{name}.json")

    df_gt = pd.read_json(rf"F:\DatenAuswertung_final\S5x\labels\{name}\labels_{name}.json")

    save_in = r"F:\DatenAuswertung_final\S5x\metrics_system\system"
    df_pred = pd.read_json(rf"F:\DatenAuswertung_final\S5x\finalpreds\{name}\persistence_boxdata_{name}.json")
    #df_pred = pd.read_json(rf"F:\Veysel\DatenAuswertung\S5x\finalpreds\{name}\{name}.json") # pure faster


    im_dir = rf"F:\DatenAuswertung_final\S5x\tif\good\{name}"

    # iou threshold for prec, reca per image
    thresh = 0.01
    pref = "system"



    lens_sequence = []
    img_names = []
    data = {}

    # per img
    for img_name in os.listdir(im_dir):

        gt_data  = df_gt[df_gt["names"] == img_name]
        pred_data = df_pred[df_pred["names"] == img_name]


        gt_pits = np.array(gt_data["Pit"].tolist()).squeeze()
        pred_pits = np.array(pred_data["Pit"].tolist()).squeeze()

        if len(gt_pits) > 0 and len(pred_pits) > 0:
            # expand dims if only one box
            if gt_pits.size == 4:
                gt_pits = np.expand_dims(gt_pits, 0)
            if pred_pits.size == 4:
                pred_pits = np.expand_dims(pred_pits, 0)

            # compare ious between a pred box and all gt boxes
            # pick pred box w. highest iou
            # calculate precision and recall per image

            # we have N gt boxes and M pred boxes
            gt_t = box_convert(torch.tensor(gt_pits), in_fmt="xywh", out_fmt="xyxy")
            pred_t = box_convert(torch.tensor(pred_pits), in_fmt="xywh", out_fmt="xyxy")

            # labels are created in matlab with 1 indexing, python has 0 indexing correcting that offset here:
            off_tensor = -torch.ones(np.shape(gt_t))
            gt_t = gt_t + off_tensor


            # tensor of shape NxM
            ious = box_iou(gt_t, pred_t)
            ious_np = ious.numpy()


            prec = np.sum(np.max(ious_np, axis = 0) > thresh) / ious_np.shape[1]
            reca = np.sum(np.max(ious_np, axis = 1) > thresh) / ious_np.shape[0]

            p_list.append(prec)
            r_list.append(reca)

        else:
            p_list.append(None)
            r_list.append(None)

            #lens_sequence.append(mean_box_average_len)
            #img_names.append(img_name)

    names_tot.append(name)
    p_seq.append(np.mean([d for d in p_list if d is not None]))
    r_seq.append(np.mean([d for d in r_list if d is not None]))
    p_full.append(p_list)
    r_full.append(r_list)


df_tot = pd.DataFrame([p_seq, r_seq])
df_tot.columns = names_tot
df_tot.index = [f"AP_{thresh}", f"AR_{thresh}"]

p_fulldf = pd.DataFrame(p_full).transpose()
p_fulldf.columns = names_tot

r_fulldf = pd.DataFrame(r_full).transpose()
p_fulldf.columns = names_tot

try:
    print("Creating metrics folder")
    metrics_dir = os.path.join(save_in, f"metrics_pred{thresh}")
    os.mkdir(metrics_dir)
except:
    print(f"folder {metrics_dir} exists, overwriting old metrics")

df_tot.to_excel(os.path.join(metrics_dir, f"AP_AR_iou_{thresh}_{pref}.xlsx"))
p_fulldf.to_excel(os.path.join(metrics_dir, f"P_iou_{thresh}_{pref}.xlsx"))
r_fulldf.to_excel(os.path.join(metrics_dir, f"R_iou_{thresh}_{pref}.xlsx"))
"""

# this is to create a video (avi) from a dir of images
r"""for dir in os.listdir(r"D:\Daten_Seagate_Sequences_3\3CameraSetup2LST\tif"):
    #["S25_100", "S25_400", "S26_PC", "S26_PC_01", "S26_PC_02", "S26_PCSide", "S26_PCSide_01", "S29",  "S29_01", "S32" ,"S32_01", "S32_02" ]
    img_array = []
    for filename in glob.glob(rf'D:\Daten_Seagate_Sequences_3\3CameraSetup2LST\tif\{dir}\*.tif'):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter(rf'D:\Daten_Seagate_Sequences_3\3CameraSetup2LST\videos\{dir}.avi', cv2.VideoWriter_fourcc(*'DIVX'), 5, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
"""


# this was used to create center of box and correct jittering over sequence (doent work)
"""root = "D:\DatenAuswertung_180124\Cavi2"
dirs = os.listdir(os.path.join(root, "tif\good"))
save_in = r"D:\DatenAuswertung_180124\Cavi2\preds_05\weitere spielereien\pit_test\img_stab\homographcompensate"

#this takes a set of sequences and creates the track using center of boxes (might need changes for )
# per sequence k
pit_offsets = []
for k in range(len(dirs)):

    try:
        print("Creating folder")
        new_dir = os.path.join(save_in, f"{dirs[k]}")
        os.mkdir(new_dir)
    except:
        print(f"folder {new_dir} exists already.")



    data_path = os.path.join(root,"preds_05", dirs[k])
    df = pd.read_json(os.path.join(data_path,"persistence_boxdata_"+f"{dirs[k]}.json"))
    images_list = os.listdir(os.path.join(root,"tif\good", dirs[k]))


    # Make a user-defined colormap.
    cb_r = np.linspace(255,0,len(df))
    cb_g = np.linspace(100,0,len(df))
    cb_b = np.linspace(0,255,len(df))




    # per img
    c = 0
    for img_name in df["names"]:


        curr_dict = {}
        curr_dict["names"] = img_name
        jitter_img = np.ones((2160, 2560, 3), dtype=np.uint8) * 255
        xs = []
        ys = []
        #color = int(cb_r[c]), int(cb_g[c]), int(cb_b[c])

        im_data = df.loc[df["names"] == img_name]


        for box in list(im_data["Pit"])[0]:
            x = int(box[0] + box[2]/2)
            y = int(box[1] + box[3]/2)

            xs.append(x) # all x
            ys.append(y) # all y centerpoints per curr img

        xy = []
        xy_p = []
        off_l = []
        if c > 0:
            match_idx = []
            min_dist = []

            g_xy = []
            g_xyp = []
            # for each box center in i
            for x,y in zip(xs, ys):
                dis = []
                offsets = []
                # for each centerpoint in i-1
                for xo, yo in zip(prev_xs, prev_ys):
                    # check which centerpoint in i has lowest distance to centerpoint in i-1
                    dx = x-xo
                    dy = y-yo
                    dis.append(np.sqrt(dx**2+dy**2))
                    offsets.append([dx, dy])

                if np.min(dis) < 7:
                    match_idx.append(np.argmin(dis)) # list index -> box in i; list value @ index -> box in i-1
                    min_dist.append(offsets[np.argmin(dis)])
                    a = 2

                else:
                    match_idx.append(None)  # list index -> box in i; list value @ index -> box in i-1
                    min_dist.append(None)

            curr_dict["ids_i_to_iprev"] = match_idx
            curr_dict["offsets_i_to_iprev"] = min_dist

            cps = np.array([xs, ys]).transpose()
            cps_prev = np.array([prev_xs, prev_ys]).transpose()

            test_dic = {}
            for idc, idp in enumerate(match_idx):
                if idp is not None:
                    x1, y1 = cps_prev[idp]
                    x2, y2 = cps[idc]

                    # min_dist: 0:x, 1:y
                    #mean_off = np.mean(min_dist, axis=0)
                    # mean_off = np.mean([k for k in min_dist if k is not None], axis=0)
                    # mean_del = abs(min_dist[idc] - mean_off)
                    # dis = np.sqrt(mean_del[0]**2 + mean_del[1]**2)
                    # color = (100+ 50 * dis, 150, 20) # more blue, more delta in y, more red more delta in x
                    # cv2.arrowedLine(jitter_img, (x1, y1), (x1+min_dist[idc][0]*25, y1+min_dist[idc][1]*25), color, thickness=3, tipLength = 0.6)

                    xy_p.append([x1,y1])
                    xy.append([x2,y2])
                    off_l.append(min_dist[idc])

                    # test_dic["name"] = img_name + f"_pit_{idc}"
                    # test_dic["current"] = [x2,y2]
                    # test_dic["previous"] = [x1, y1]
                    # test_dic["offset"] = min_dist[idc]
                    # 
                    # pit_offsets.append(copy.deepcopy(test_dic))



            if len(xy) > 3 and len(xy_p) > 3:
                img_cv = cv2.imread(os.path.join(root, "tif\good", dirs[k], img_name))





                # M = cv2.getAffineTransform(np.float32([[50, 50],
                #                                        [200, 50],
                #                                        [50, 200]]), np.float32([[10, 100],
                #                                                                 [200, 50],
                #                                                                 [100, 250]]))

                #Translation (mean offset)
                # mean_off = np.mean(off_l, axis=0)
                # T = np.float32([[1, 0, -mean_off[0]], [0, 1, -mean_off[1]]])
                # dst = cv2.warpAffine(img_cv, T, (2560,2160))

                # affine
                #h, status = cv2.estimateAffine2D(np.float32(xy), np.float32(xy_p))
                #dst = cv2.warpAffine(img_cv, h, (2560, 2160))

                # homographic
                #h, status = cv2.findHomography(np.float32(xy), np.float32(xy_p))


                # dst = cv2.warpPerspective(img_cv, h, (2560,2160))
                # for (x,y),(x2,y2) in zip(xy_p, xy):
                #
                #     xa, ya = h[:, :2] @ [x,y] + h[:, 2] #red arrows: affine transform
                #     dx = xa-x
                #     dy = ya-y
                #
                #     dx2 = x2-x # green: actual movement vector
                #     dy2 = y2-y
                #     cv2.arrowedLine(img_cv, (x, y), (int(xa+25*dx), int(ya+25*dy)), (255,0,0), thickness=2, tipLength=0.5)
                #     cv2.arrowedLine(img_cv, (x, y), (int(x2 + 25 * dx2), int(y2 + 25 * dy2)), (0, 255, 0), thickness=2,
                #                     tipLength=0.5)


                # trying skimage tforms
                # tform3 = transform.ProjectiveTransform()
                # tform3.estimate(np.float32(xy_p), np.float32(xy))
                # warped = np.array(transform.warp(img_cv, tform3, output_shape=(2160, 2560))*255, dtype = np.uint8)



                # homographic
                #h, status = cv2.findHomography(np.float32(xy), np.float32(xy_p))
                #dst = cv2.warpPerspective(img_cv, h, (2560,2160))


                #save stab
                #pil_img = Image.fromarray(dst)
                #pil_img.save(os.path.join(save_in, f"{dirs[k]}", f"dist_{img_name}"))

                #save flow
                pil_img = Image.fromarray(warped)
                pil_img.save(os.path.join(save_in, f"{dirs[k]}", f"affine_{img_name}"))

                #prev_xs = (np.array(xy)[:,0] + np.array(off_l)[:,0]).tolist()
                #prev_ys = (np.array(xy)[:,1] + np.array(off_l)[:,1]).tolist()
                prev_img_cv = copy.deepcopy(img_cv)
                prev_img_name = copy.deepcopy(img_name)
                prev_xs = copy.deepcopy(np.array(xs))
                prev_ys = copy.deepcopy(np.array(ys))
            else:
                prev_xs = copy.deepcopy(np.array(xs))
                prev_ys = copy.deepcopy(np.array(ys))
        else:
            prev_xs = copy.deepcopy(np.array(xs))
            prev_ys = copy.deepcopy(np.array(ys))
        c += 1
"""
"""
# this is used to calculate ring center & diameter, then ONE (constant) standard deviation per image
root = r"F:\Daten_Seagate_Sequences_3\3CameraSetup2LST"
dirs = os.listdir(os.path.join(r"F:\Daten_Seagate_Sequences_3\3CameraSetup2LST\finalpreds_oldersamples"))
save_in = r"F:\Daten_Seagate_Sequences_3\3CameraSetup2LST\evaluation_oldersamples\spatial"

#this takes a set of sequences and creates the track using center of boxes (might need changes for )
# per sequence k
pit_offsets = []
seq_dic = {}

writer = pd.ExcelWriter(os.path.join(save_in, "ring_variables_oldsamples.xlsx"), engine='xlsxwriter')

c1 = (255,255,255)
c2 = (255,255,255)

for k in range(len(dirs)):

    try:
        print("Creating folder")
        new_dir = os.path.join(save_in, f"{dirs[k]}")
        os.mkdir(new_dir)
    except:
        print(f"folder {new_dir} exists already.")



    data_path = os.path.join(root,"finalpreds_oldersamples", dirs[k])
    df = pd.read_json(os.path.join(data_path,"persistence_boxdata_"+f"{dirs[k]}.json"))
    images_list = os.listdir(os.path.join(root,"tif", dirs[k]))


    # Make a user-defined colormap.
    cb_r = np.linspace(255,0,len(df))
    cb_g = np.linspace(100,0,len(df))
    cb_b = np.linspace(0,255,len(df))




    # per img

    c = 0
    for img_name in df["names"]:


        curr_dict = {}
        curr_dict["names"] = img_name
        jitter_img = np.ones((2160, 2560, 3), dtype=np.uint8) * 255
        jitter_img = np.array(Image.open(os.path.join(root, "tif", dirs[k], img_name)).convert("RGB"))
        xs = []
        ys = []


        #color = int(cb_r[c]), int(cb_g[c]), int(cb_b[c])

        im_data = df.loc[df["names"] == img_name]


        for box in list(im_data["Pit"])[0]:
            x = int(box[0] + box[2]/2)
            y = int(box[1] + box[3]/2)

            xs.append(x) # all x
            ys.append(y) # all y centerpoints per curr img

            # draw center of boxes (of pits)
            jitter_img = cv2.circle(jitter_img, (x,y), radius=3, color=(0,0,255), thickness=6)


        if len(xs) >2:
            # mean centerpoint
            x_m = np.mean(xs)
            y_m = np.mean(ys)

            # calculation of the reduced coordinates
            u = xs - x_m
            v = ys - y_m

            # linear system defining the center in reduced coordinates (uc, vc):
            #    Suu * uc +  Suv * vc = (Suuu + Suvv)/2
            #    Suv * uc +  Svv * vc = (Suuv + Svvv)/2
            Suv = sum(u * v)
            Suu = sum(u ** 2)
            Svv = sum(v ** 2)
            Suuv = sum(u ** 2 * v)
            Suvv = sum(u * v ** 2)
            Suuu = sum(u ** 3)
            Svvv = sum(v ** 3)

            # Solving the linear system
            A = np.array([[Suu, Suv], [Suv, Svv]])
            B = np.array([Suuu + Suvv, Svvv + Suuv]) / 2.0
            uc, vc = np.linalg.solve(A, B)

            xc_1 = x_m + uc
            yc_1 = y_m + vc

            # Calculation of all distances from the center (xc_1, yc_1)
            Ri_1 = np.sqrt((xs - xc_1) ** 2 + (ys - yc_1) ** 2)
            R_1 = np.mean(Ri_1)
            residu_1 = sum((Ri_1 - R_1) ** 2)
            residu2_1 = sum((Ri_1 ** 2 - R_1 ** 2) ** 2)

            # standard deviation from circle
            SD = np.sqrt(residu_1/len(Ri_1))

            # draw radius and center
            jitter_img = cv2.circle(jitter_img, center =(int(xc_1), int(yc_1)), radius=int(R_1), color=c1, thickness=6)
            jitter_img = cv2.circle(jitter_img, center=(int(xc_1), int(yc_1)), radius=6,thickness = 6, color=c1)

            jitter_img = cv2.circle(jitter_img, center=(int(xc_1), int(yc_1)), radius=int(R_1 + SD), color=c2, thickness=4, lineType=cv2.LINE_8)
            jitter_img = cv2.circle(jitter_img, center=(int(xc_1), int(yc_1)), radius=int(R_1- SD), color=c2, thickness=4, lineType=cv2.LINE_8)


            seq_dic[f"{img_name}"] = {"x": int(xc_1), "y": int(yc_1), "R":int(R_1), "SD": SD}
        cv2.imwrite(os.path.join(save_in, f"{dirs[k]}", f"{img_name}"), jitter_img)
        c += 1

    seq_df = pd.DataFrame(seq_dic).transpose()


    seq_df.to_excel(writer, sheet_name=f"{dirs[k]}")
writer.close()
"""


# this is used to calculate ring center & diameter, then ONE (constant) standard deviation & splits circle ring into segments
# per img (doesnt work yet)
r"""
def fit_circle(xs,ys):
    # calculation of the reduced coordinates
    u = xs - x_m
    v = ys - y_m

    # linear system defining the center in reduced coordinates (uc, vc):
    #    Suu * uc +  Suv * vc = (Suuu + Suvv)/2
    #    Suv * uc +  Svv * vc = (Suuv + Svvv)/2
    Suv = sum(u * v)
    Suu = sum(u ** 2)
    Svv = sum(v ** 2)
    Suuv = sum(u ** 2 * v)
    Suvv = sum(u * v ** 2)
    Suuu = sum(u ** 3)
    Svvv = sum(v ** 3)

    # Solving the linear system
    A = np.array([[Suu, Suv], [Suv, Svv]])
    B = np.array([Suuu + Suvv, Svvv + Suuv]) / 2.0
    uc, vc = np.linalg.solve(A, B)

    xc_1 = x_m + uc
    yc_1 = y_m + vc

    # Calculation of all distances from the center (xc_1, yc_1)
    Ri_1 = np.sqrt((xs - xc_1) ** 2 + (ys - yc_1) ** 2)
    R_1 = np.mean(Ri_1)
    residu_1 = sum((Ri_1 - R_1) ** 2)
    residu2_1 = sum((Ri_1 ** 2 - R_1 ** 2) ** 2)

    # standard deviation from circle
    SD = np.sqrt(residu_1 / len(Ri_1))

    return R_1, SD

root = r"F:\DatenAuswertung_final\S5x"
dirs = os.listdir(os.path.join(root, "finalpreds"))
save_in = r"F:\DatenAuswertung_final\S5x\spatial distribution\rings"

# per sequence k
pit_offsets = []
for k in range(len(dirs)):

    try:
        print("Creating folder")
        new_dir = os.path.join(save_in, f"{dirs[k]}")
        os.mkdir(new_dir)
    except:
        print(f"folder {new_dir} exists already.")



    data_path = os.path.join(root,"finalpreds", dirs[k])
    df = pd.read_json(os.path.join(data_path,"persistence_boxdata_"+f"{dirs[k]}.json"))
    images_list = os.listdir(os.path.join(root,"tif\good", dirs[k]))


    # Make a user-defined colormap.
    cb_r = np.linspace(255,0,len(df))
    cb_g = np.linspace(100,0,len(df))
    cb_b = np.linspace(0,255,len(df))



    colors = np.array(np.random.rand(20,3)*255, dtype = np.uint8)
    # per img
    seq_data = {}
    c = 0
    for img_name in df["names"]:


        curr_dict = {}
        curr_dict["names"] = img_name
        jitter_img = np.ones((2160, 2560, 3), dtype=np.uint8)*255
        jitter_img = np.array(Image.open(os.path.join(root, "tif\good", dirs[k], img_name)).convert("RGB"))

        xs = []
        ys = []
        #color = int(cb_r[c]), int(cb_g[c]), int(cb_b[c])

        im_data = df.loc[df["names"] == img_name]


        for box in list(im_data["Pit"])[0]:
            x = int(box[0] + box[2]/2)
            y = int(box[1] + box[3]/2)

            xs.append(x) # all x
            ys.append(y) # all y centerpoints per curr img

            # draw center of boxes (of pits)
            #jitter_img = cv2.circle(jitter_img, (x,y), radius=3, color=(0,0,255), thickness=6)



        if len(xs) >2:
            # mean centerpoint
            x_m = np.mean(xs)
            y_m = np.mean(ys)

            # calculation of the reduced coordinates
            u = xs - x_m
            v = ys - y_m

            # linear system defining the center in reduced coordinates (uc, vc):
            #    Suu * uc +  Suv * vc = (Suuu + Suvv)/2
            #    Suv * uc +  Svv * vc = (Suuv + Svvv)/2
            Suv = sum(u * v)
            Suu = sum(u ** 2)
            Svv = sum(v ** 2)
            Suuv = sum(u ** 2 * v)
            Suvv = sum(u * v ** 2)
            Suuu = sum(u ** 3)
            Svvv = sum(v ** 3)

            # Solving the linear system
            A = np.array([[Suu, Suv], [Suv, Svv]])
            B = np.array([Suuu + Suvv, Svvv + Suuv]) / 2.0
            uc, vc = np.linalg.solve(A, B)

            xc_1 = x_m + uc
            yc_1 = y_m + vc

            # Calculation of all distances from the center (xc_1, yc_1)
            Ri_1 = np.sqrt((xs - xc_1) ** 2 + (ys - yc_1) ** 2)
            R_1 = np.mean(Ri_1)
            residu_1 = sum((Ri_1 - R_1) ** 2)
            residu2_1 = sum((Ri_1 ** 2 - R_1 ** 2) ** 2)

            # standard deviation from circle
            SD = np.sqrt(residu_1/len(Ri_1))

            # draw radius and center
            jitter_img = cv2.circle(jitter_img, center =(int(xc_1), int(yc_1)), radius=int(R_1), color=(255,255,255), thickness=6)
            jitter_img = cv2.circle(jitter_img, center=(int(xc_1), int(yc_1)), radius=6,thickness = 6, color=(255,255,255))

            jitter_img = cv2.circle(jitter_img, center=(int(xc_1), int(yc_1)), radius=int(R_1 + SD), color=(255,255,255), thickness=4)
            jitter_img = cv2.circle(jitter_img, center=(int(xc_1), int(yc_1)), radius=int(R_1- SD), color=(255,255,255), thickness=4)




            seq_data[f"{img_name}"] = {"x": int(xc_1), "y": int(yc_1), "R":int(R_1), "SD": SD}
            r = int(R_1)
            sliceno = np.int32((np.pi + np.arctan2(np.array(ys)-r, np.array(xs)-r)) * (20 / (2*np.pi)))
            mypoints = np.array([xs,ys], dtype=np.uint8).transpose()
            for ci in np.unique(sliceno):
                if len(mypoints[np.where(sliceno == ci)]) > 0:
                    color = np.random.rand((3))*255
                    passed_points = mypoints[np.where(sliceno == ci)]

                    for pit in passed_points:
                        jitter_img = cv2.circle(jitter_img, center=pit, radius=6, color=color, thickness=4)

                else:
                    a = 2





        cv2.imwrite(os.path.join(save_in, f"{dirs[k]}", f"{img_name}"), jitter_img)
        c += 1
"""

# this is used to calculate ring center & diameter, then ONE (constant) standard deviation & clusters hotspots per img
"""
def fit_circle(xs,ys):
    # calculation of the reduced coordinates
    u = xs - x_m
    v = ys - y_m

    # linear system defining the center in reduced coordinates (uc, vc):
    #    Suu * uc +  Suv * vc = (Suuu + Suvv)/2
    #    Suv * uc +  Svv * vc = (Suuv + Svvv)/2
    Suv = sum(u * v)
    Suu = sum(u ** 2)
    Svv = sum(v ** 2)
    Suuv = sum(u ** 2 * v)
    Suvv = sum(u * v ** 2)
    Suuu = sum(u ** 3)
    Svvv = sum(v ** 3)

    # Solving the linear system
    A = np.array([[Suu, Suv], [Suv, Svv]])
    B = np.array([Suuu + Suvv, Svvv + Suuv]) / 2.0
    uc, vc = np.linalg.solve(A, B)

    xc_1 = x_m + uc
    yc_1 = y_m + vc

    # Calculation of all distances from the center (xc_1, yc_1)
    Ri_1 = np.sqrt((xs - xc_1) ** 2 + (ys - yc_1) ** 2)
    R_1 = np.mean(Ri_1)
    residu_1 = sum((Ri_1 - R_1) ** 2)
    residu2_1 = sum((Ri_1 ** 2 - R_1 ** 2) ** 2)

    # standard deviation from circle
    SD = np.sqrt(residu_1 / len(Ri_1))

    return R_1, SD

root = r"F:\DatenAuswertung_final\preds_Cav3"
preds_folder = "predictions"
save_in = r"F:\DatenAuswertung_final\ringe\Cav3"

dirs = os.listdir(os.path.join(root, preds_folder))

writer = pd.ExcelWriter(os.path.join(save_in, "ring_variables.xlsx"), engine='xlsxwriter')
writer2 = pd.ExcelWriter(os.path.join(save_in, "ring_clusters.xlsx"), engine='xlsxwriter')

# per sequence k
pit_offsets = []
for k in range(len(dirs)):
    cluster_dic = {}
    seq_dic = {}

    try:
        print("Creating folder")
        new_dir = os.path.join(save_in, f"{dirs[k]}")
        os.mkdir(new_dir)
    except:
        print(f"folder {new_dir} exists already.")



    data_path = os.path.join(root,preds_folder, dirs[k])
    df = pd.read_json(os.path.join(data_path,"persistence_boxdata_"+f"{dirs[k]}.json"))
    images_list = os.listdir(os.path.join(root,"tif", dirs[k]))


    # Make a user-defined colormap.
    cb_r = np.linspace(255,0,len(df))
    cb_g = np.linspace(100,0,len(df))
    cb_b = np.linspace(0,255,len(df))



    colors = np.array(np.random.rand(20,3)*255, dtype = np.uint8)
    # per img
    seq_data = {}
    c = 0
    for img_name in df["names"]:

        curr_dict = {}
        curr_dict["names"] = img_name
        jitter_img = np.ones((2160, 2560, 3), dtype=np.uint8)*255
        jitter_img = np.array(Image.open(os.path.join(root, "tif", dirs[k], img_name)).convert("RGB"))

        xs = []
        ys = []
        #color = int(cb_r[c]), int(cb_g[c]), int(cb_b[c])

        im_data = df.loc[df["names"] == img_name]


        for box in list(im_data["Pit"])[0]:
            x = int(box[0] + box[2]/2)
            y = int(box[1] + box[3]/2)

            xs.append(x) # all x
            ys.append(y) # all y centerpoints per curr img

            # draw center of boxes (of pits)
            #jitter_img = cv2.circle(jitter_img, (x,y), radius=3, color=(0,0,255), thickness=6)



        if len(xs) >2:
            # mean centerpoint
            x_m = np.mean(xs)
            y_m = np.mean(ys)

            # calculation of the reduced coordinates
            u = xs - x_m
            v = ys - y_m

            # linear system defining the center in reduced coordinates (uc, vc):
            #    Suu * uc +  Suv * vc = (Suuu + Suvv)/2
            #    Suv * uc +  Svv * vc = (Suuv + Svvv)/2
            Suv = sum(u * v)
            Suu = sum(u ** 2)
            Svv = sum(v ** 2)
            Suuv = sum(u ** 2 * v)
            Suvv = sum(u * v ** 2)
            Suuu = sum(u ** 3)
            Svvv = sum(v ** 3)

            # Solving the linear system
            A = np.array([[Suu, Suv], [Suv, Svv]])
            B = np.array([Suuu + Suvv, Svvv + Suuv]) / 2.0
            uc, vc = np.linalg.solve(A, B)

            xc_1 = x_m + uc
            yc_1 = y_m + vc

            # Calculation of all distances from the center (xc_1, yc_1)
            Ri_1 = np.sqrt((xs - xc_1) ** 2 + (ys - yc_1) ** 2)
            R_1 = np.mean(Ri_1)
            residu_1 = sum((Ri_1 - R_1) ** 2)
            residu2_1 = sum((Ri_1 ** 2 - R_1 ** 2) ** 2)

            # standard deviation from circle
            SD = np.sqrt(residu_1/len(Ri_1))

            # draw radius and center
            jitter_img = cv2.circle(jitter_img, center =(int(xc_1), int(yc_1)), radius=int(R_1), color=(255,255,255), thickness=6)
            jitter_img = cv2.circle(jitter_img, center=(int(xc_1), int(yc_1)), radius=6,thickness = 6, color=(255,255,255))

            jitter_img = cv2.circle(jitter_img, center=(int(xc_1), int(yc_1)), radius=int(R_1 + SD), color=(255,255,255), thickness=4)
            jitter_img = cv2.circle(jitter_img, center=(int(xc_1), int(yc_1)), radius=int(R_1- SD), color=(255,255,255), thickness=4)

            seq_dic[f"{img_name}"] = {"x": int(xc_1), "y": int(yc_1), "R": int(R_1), "SD": round(SD,2)}




            data = np.transpose(np.array([xs, ys]))
            #data = data[np.lexsort((data[:,1], data[:,0]))]

            datas = StandardScaler().fit_transform(data)
            db = DBSCAN(eps=0.25, min_samples=5).fit(datas)
            labels = db.labels_
            # Number of clusters in labels, ignoring noise if present.
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise_ = list(labels).count(-1)

            lab_data = np.transpose(np.array([labels, xs, ys]))
            for label in np.unique(labels):
                passedlabdata = lab_data[np.where(lab_data[:, 0] == label)]
                if label == -1:
                    for j in range(len(passedlabdata)):
                        jitter_img = cv2.circle(jitter_img, (passedlabdata[j,1], passedlabdata[j,2]),
                                                radius=3, color=(0, 0, 255), thickness=6)
                else:

                    # pxmim = passedlabdata[np.argmin(passedlabdata[:,1])][1:]
                    # pymin = passedlabdata[np.argmin(passedlabdata[:, 2])][1:]
                    # pxmax = passedlabdata[np.argmax(passedlabdata[:, 1])][1:]
                    # pymax = passedlabdata[np.argmax(passedlabdata[:, 2])][1:]
                    #
                    # jitter_img = cv2.polylines(jitter_img, [1], isClosed=True, color=(125, 0, 125), thickness=6)
                    try:
                        # ell = EllipseModel()
                        # ell.estimate(passedlabdata[:, 1:])
                        # xc, yc, a, b, theta = ell.params
                        # theta = int(theta*180/np.pi)
                        # axis_len = np.sqrt(a**2 + b**2)
                        # jitter_img = cv2.ellipse(jitter_img, (int(xc),int(yc)), (int(a), int(b)),
                        #         theta, 0, 360, color = (125,0,125), thickness=6)

                        #ellipse = cv2.fitEllipseDirect(passedlabdata[:, 1:])
                        #jitter_img = cv2.ellipse(jitter_img, ellipse, color=(125, 0, 125), thickness=6)

                        # (x,y),radius = cv2.minEnclosingCircle(passedlabdata[:, 1:])
                        # center = (int(x), int(y))
                        # radius = int(radius)
                        # cv2.circle(jitter_img, center, radius, (125,0,125), 6)

                        hull = cv2.convexHull(passedlabdata[:, 1:])
                        jitter_img = cv2.drawContours(jitter_img, [hull], -1, (125,0,125), 6)

                        lines = np.hstack([hull.squeeze(), np.roll(hull.squeeze(), -1, axis=0)])
                        area = 0.5 * abs(sum(x1 * y2 - x2 * y1 for x1, y1, x2, y2 in lines))
                        xmid = int(np.mean(hull.squeeze()[:,0]))
                        ymid = int(np.mean(hull.squeeze()[:,1]))
                        n_pits = len(passedlabdata)

                        jitter_img = cv2.circle(jitter_img, (xmid,ymid),radius = 3,color= (125, 0, 125), thickness=6)
                        cluster_dic[f"{img_name}_{label+1}"] = {"x": xmid, "y": ymid, "n_pits": n_pits, "area_px": area}








                        #hotspot[label] = {"x":xc, "y":yc, "a":a, "b":b, "theta": theta}
                    except:
                        xc, yc, a, b, theta = None, None, None, None, None
                        print("failed drawing hull")


                    # xss = passedlabdata[:,1]
                    # yss = passedlabdata[:,2]
                    # xms = int(np.mean(xss))
                    # yms = int(np.mean(yss))
                    # #rs, sds = fit_circle(xss,yss)
                    # #rs = int(rs)
                    # #sds = int(sds)
                    # pts = passedlabdata[:, 1:]
                    # pts = pts.reshape((-1, 1, 2))
                    # isClosed = False

                    #jitter_img = cv2.polylines(jitter_img, [pts], isClosed,
                    #                        color=(125, 0, 125), thickness=6)
                    for j in range(len(passedlabdata)):
                        jitter_img = cv2.circle(jitter_img, (passedlabdata[j, 1], passedlabdata[j, 2]),
                                                radius=3, color=(0, 0, 255), thickness=6)





                        #jitter_img = cv2.ellipse(jitter_img, (), axesLength,
                           #     angle, startAngle, endAngle, color, thickness)

            print("Estimated number of clusters: %d" % n_clusters_)
            print("Estimated number of noise points: %d" % n_noise_)
        #hotspots[]

        cv2.imwrite(os.path.join(save_in, f"{dirs[k]}", f"{img_name}"), jitter_img)
        c += 1

    seq_df = pd.DataFrame(seq_dic).transpose()
    cluster_df = pd.DataFrame(cluster_dic).transpose()

    seq_df.to_excel(writer, sheet_name=f"{dirs[k]}")
    cluster_df.to_excel(writer2, sheet_name=f"{dirs[k]}")
writer.close()
writer2.close()
"""

# this is to calculate precision and recall (at iou .5) for each image & then plot boxes w. TP,FP,FN flags for each sequence#
r"""
names = ["S51_01", "S52", "S53", "S57_C1", "S58_C1_01", "S59"]
p_seq = []
r_seq = []
names_tot = []
p_full = []
r_full = []

data_list = []
avrgboxsize_tot = []

cp_gt = (20,230,20)
cc_gt = (125,0,125)
cp_pred = (230,20,20)
cc_pred = (20,20,230)


for name in names:
    p_list = []
    r_list = []
    # save_in = rf"F:\DatenAuswertung_180124\Cavi2\labeling\labels"
    # df = pd.read_json(rf"F:\DatenAuswertung_180124\Cavi2\labeling\labels\{name}\labels_{name}.json")

    df_gt = pd.read_json(rf"F:\DatenAuswertung_final\S5x\labels\{name}\labels_{name}.json")

    save_in = r"F:\DatenAuswertung_final\S5x\metrics_system\boxes and flags color"
    df_pred = pd.read_json(rf"F:\DatenAuswertung_final\S5x\finalpreds\{name}\persistence_boxdata_{name}.json")
    #df_pred = pd.read_json(rf"F:\Veysel\DatenAuswertung\S5x\finalpreds\{name}\{name}.json") # pure faster


    im_dir = rf"F:\DatenAuswertung_final\S5x\tif\good\{name}"

    # iou threshold for prec, reca per image
    thresh = 0.5
    pref = "system"

    try:
        save_dir = os.path.join(save_in, f"{name}")
        os.mkdir(save_dir)
    except:
        a = 1

    lens_sequence = []
    img_names = []
    data = {}


    # per img
    for img_name in os.listdir(im_dir):

        gt_data = df_gt[df_gt["names"] == img_name]
        pred_data = df_pred[df_pred["names"] == img_name]

        gt_pits = np.array(gt_data["Pit"].tolist()).squeeze()
        pred_pits = np.array(pred_data["Pit"].tolist()).squeeze()

        if len(gt_pits) > 0 and len(pred_pits) > 0:
            # expand dims if only one box
            if gt_pits.size == 4:
                gt_pits = np.expand_dims(gt_pits, 0)
            if pred_pits.size == 4:
                pred_pits = np.expand_dims(pred_pits, 0)

            # compare ious between a pred box and all gt boxes
            # pick pred box w. highest iou
            # calculate precision and recall per image

            # we have N gt boxes and M pred boxes
            gt_t = box_convert(torch.tensor(gt_pits), in_fmt="xywh", out_fmt="xyxy")
            pred_t = box_convert(torch.tensor(pred_pits), in_fmt="xywh", out_fmt="xyxy")

            # labels are created in matlab with 1 indexing, python has 0 indexing correcting that offset here:
            off_tensor = -torch.ones(np.shape(gt_t))
            gt_t = gt_t + off_tensor


            # tensor of shape NxM
            ious = box_iou(gt_t, pred_t)
            ious_np = ious.numpy()

            pred_tps = np.max(ious_np, axis = 0) > thresh
            gt_tps = np.max(ious_np, axis = 1) > thresh

            gt_labels = [ "TP" if o == True else "FN" for o in gt_tps]
            pred_labels = [ "TP" if o == True else "FP" for o in pred_tps]

            prec = np.sum(np.max(ious_np, axis = 0) > thresh) / ious_np.shape[1]
            reca = np.sum(np.max(ious_np, axis = 1) > thresh) / ious_np.shape[0]

            p_list.append(prec)
            r_list.append(reca)

        else:
            p_list.append(None)
            r_list.append(None)

            gt_labels = None
            pred_labels = None

        img = torch.tensor(np.array(Image.open(os.path.join(im_dir, img_name)).convert("RGB")))
        img = torch.permute(img, (2, 0, 1))

        ii = 0
        for k, c, l, t in zip([gt_data, pred_data], [(cp_gt, cc_gt), (cp_pred, cc_pred)], [gt_labels, pred_labels], ["FN", "FP"]):

            if k["Pit"].size == 4:
                k["Pit"] = np.expand_dims(k["Pit"], 0)
            if k["Cluster"].size == 4:
                k["Cluster"] = np.expand_dims(k["Cluster"], 0)

            if len(k) != 0:
                cl_boxes = k["Cluster"].values[0]
                pit_boxes = k["Pit"].values[0]
                boxes = cl_boxes + pit_boxes
                colors = [c[1]] * len(cl_boxes) + [c[0]] * len(pit_boxes)

                if l is not None:
                    colors = [c[0] if o == t else (20, 20, 230) for o in l]

                if np.array(boxes).size == 4:
                    boxes = np.expand_dims(np.array(boxes), 0)

                if len(boxes) > 0:
                    box_t = torch.tensor(np.squeeze(boxes)).view((len(boxes), 4))
                    box_tc = torchvision.ops.box_convert(box_t, in_fmt="xywh", out_fmt="xyxy")



                    img = draw_bounding_boxes(img, box_tc,
                                              colors=colors, font=r"C:\Users\veyse\Desktop\font\arial")

            pil_img = Image.fromarray(np.transpose(img, (1, 2, 0)).numpy())

            if l is not None:
                I1 = ImageDraw.Draw(pil_img)
                font = ImageFont.truetype("arial.ttf", 64)
                I1.text((50, 50), f"P: {round(prec,2)}", font = font)
                I1.text((50, 150), f"R: {round(reca,2)}", font = font)
            pil_img.save(os.path.join(save_in, name, f"{img_name}"))
            #lens_sequence.append(mean_box_average_len)
            #img_names.append(img_name)
            ii +=1

    names_tot.append(name)
    p_seq.append(np.mean([d for d in p_list if d is not None]))
    r_seq.append(np.mean([d for d in r_list if d is not None]))
    p_full.append(p_list)
    r_full.append(r_list)


df_tot = pd.DataFrame([p_seq, r_seq])
df_tot.columns = names_tot
df_tot.index = [f"AP_{thresh}", f"AR_{thresh}"]

p_fulldf = pd.DataFrame(p_full).transpose()
p_fulldf.columns = names_tot

r_fulldf = pd.DataFrame(r_full).transpose()
p_fulldf.columns = names_tot

try:
    print("Creating metrics folder")
    metrics_dir = os.path.join(save_in, f"metrics_pred{thresh}")
    os.mkdir(metrics_dir)
except:
    print(f"folder {metrics_dir} exists, overwriting old metrics")

df_tot.to_excel(os.path.join(metrics_dir, f"AP_AR_iou_{thresh}_{pref}.xlsx"))
p_fulldf.to_excel(os.path.join(metrics_dir, f"P_iou_{thresh}_{pref}.xlsx"))
r_fulldf.to_excel(os.path.join(metrics_dir, f"R_iou_{thresh}_{pref}.xlsx"))"""

# this to plot pred (red) & gt (green) boxes on an image
r"""
names = ["S51_01", "S52", "S53", "S57_C1", "S58_C1_01", "S59"]
data_list = []
names_tot = []
avrgboxsize_tot = []

cp_gt = (20,230,20)
cc_gt = (125,0,125)
cp_pred = (230,20,20)
cc_pred = (20,20,230)


for name in names:
    #save_in = rf"F:\DatenAuswertung_180124\Cavi2\labeling\labels"
    #df = pd.read_json(rf"F:\DatenAuswertung_180124\Cavi2\labeling\labels\{name}\labels_{name}.json")

    df_gt = pd.read_json(rf"F:\DatenAuswertung_final\S5x\labels\{name}\labels_{name}.json")

    save_in = r"F:\DatenAuswertung_final\S5x\boxes_plotted\system_diaglabels"
    df_pred = pd.read_json(rf"F:\DatenAuswertung_final\S5x\finalpreds\{name}\persistence_boxdata_{name}.json")
    #df_pred = pd.read_json(rf"F:\Veysel\DatenAuswertung\S5x\finalpreds\{name}\{name}.json") # pure faster


    im_dir = rf"F:\DatenAuswertung_final\S5x\tif\good\{name}"

    try:
        save_dir = os.path.join(save_in, f"{name}")
        os.mkdir(save_dir)
    except:
        a = 1


    # per img
    for img_name in df_gt["names"]:
        gt_data = df_gt.loc[df_gt["names"] == img_name]
        pred_data = df_pred.loc[df_pred["names"] == img_name]

        img = torch.tensor(np.array(Image.open(os.path.join(im_dir, img_name)).convert("RGB")))
        img = torch.permute(img, (2, 0, 1))

        for k, c in zip([gt_data, pred_data], [(cp_gt, cc_gt), (cp_pred, cc_pred)]):



            if k["Pit"].size == 4:
                k["Pit"] = np.expand_dims(k["Pit"], 0)
            if k["Cluster"].size == 4:
                k["Cluster"] = np.expand_dims(k["Cluster"], 0)

            if len(k) != 0:
                cl_boxes = k["Cluster"].values[0]
                pit_boxes = k["Pit"].values[0]
                boxes = cl_boxes + pit_boxes
                colors = [c[1]]*len(cl_boxes) + [c[0]]*len(pit_boxes)


                if np.array(boxes).size == 4:
                    boxes = np.expand_dims(np.array(boxes),0)


                if len(boxes) > 0:
                    box_t = torch.tensor(np.squeeze(boxes)).view((len(boxes), 4))
                    if len(boxes) > 1:
                        labels = [str(round(np.sqrt(boxes[k][2]**2 + boxes[k][3]**2),1)) for k in range(len(boxes))]
                    else:
                        labels = ["fu"]
                    box_tc = torchvision.ops.box_convert(box_t, in_fmt="xywh", out_fmt="xyxy")
                    img = draw_bounding_boxes(img, box_tc,
                                              colors=colors, labels=labels, font=r"C:\Users\veyse\Desktop\font\arial", font_size=16)

        pil_img = Image.fromarray(np.transpose(img, (1, 2, 0)).numpy())
        pil_img.save(os.path.join(save_in,name, f"{img_name}"))
"""

# this to plot pred (red) & gt (green) boxes on an image

names = os.listdir(r"F:\DatenAuswertung_final\oldersamples\finalpreds_oldersamples")
data_list = []
names_tot = []
avrgboxsize_tot = []

cp_gt = (20,230,20)
cc_gt = (125,0,125)
cp_pred = (230,20,20)
cc_pred = (20,20,230)


for name in names:
    #save_in = rf"F:\DatenAuswertung_180124\Cavi2\labeling\labels"
    #df = pd.read_json(rf"F:\DatenAuswertung_180124\Cavi2\labeling\labels\{name}\labels_{name}.json")

    #df_gt = pd.read_json(rf"F:\DatenAuswertung_final\S5x\labels\{name}\labels_{name}.json")

    save_in = rf"F:\DatenAuswertung_final\oldersamples\boxes_pred"
    df_pred = pd.read_json(rf"F:\DatenAuswertung_final\oldersamples\finalpreds_oldersamples\{name}\persistence_boxdata_{name}.json")
    im_dir = rf"F:\DatenAuswertung_final\oldersamples\tif\good\{name}"

    try:
        save_dir = os.path.join(save_in, f"{name}")
        os.mkdir(save_dir)
    except:
        a = 1


    # per img
    for img_name in df_pred["names"]:
        pred_data = df_pred.loc[df_pred["names"] == img_name]

        img = torch.tensor(np.array(Image.open(os.path.join(im_dir, img_name)).convert("RGB")))
        img = torch.permute(img, (2, 0, 1))

        k = pred_data
        c = cc_pred



        if k["Pit"].size == 4:
            k["Pit"] = np.expand_dims(k["Pit"], 0)
        if k["Cluster"].size == 4:
            k["Cluster"] = np.expand_dims(k["Cluster"], 0)

        if len(k) != 0:
            cl_boxes = k["Cluster"].values[0]
            pit_boxes = k["Pit"].values[0]
            boxes = cl_boxes + pit_boxes
            colors = [c[1]]*len(cl_boxes) + [c[0]]*len(pit_boxes)


            if np.array(boxes).size == 4:
                boxes = np.expand_dims(np.array(boxes),0)


            if len(boxes) > 0:
                box_t = torch.tensor(np.squeeze(boxes)).view((len(boxes), 4))
                if len(boxes) > 1:
                    labels = [str(round(np.sqrt(boxes[k][2]**2 + boxes[k][3]**2),1)) for k in range(len(boxes))]
                else:
                    labels = ["fu"]
                box_tc = torchvision.ops.box_convert(box_t, in_fmt="xywh", out_fmt="xyxy")
                img = draw_bounding_boxes(img, box_tc,
                                          colors= (255,0,0), font=r"C:\Users\veyse\Desktop\font\arial", font_size=16)

        pil_img = Image.fromarray(np.transpose(img, (1, 2, 0)).numpy())
        pil_img.save(os.path.join(save_in,name, f"{img_name}"))
