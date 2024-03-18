import os
import shutil
import ReadIM
# to install on windows pip install ReadIM --index-url https://pypi.fury.io/alexlib (if pip install ReadIM doesnt work)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

def read_im7(filepath):
    "input filepath - output numpy array of image (currently in native format (16 bit depth resolution)"

    vbuff, vatts = ReadIM.extra.get_Buffer_andAttributeList(filepath)
    v_array, vbuff = ReadIM.extra.buffer_as_array(vbuff)
    del(vbuff)

    # to 8 bit array
    max_val = v_array.max()
    v_array = (v_array / max_val * (2 ** 8)).astype("uint8")
    # manually increase pixel intensity
    #v_array += 50
    return v_array


im7_path = r"E:\Jonas\24\preds_Cav3\im7" # takes name
save_path = r"E:\Jonas\24\preds_Cav3\tif" # automatically creates folder with dir name

#dir_name = path.split("\\")[-1]

#create dir tree

"""if os.path.exists(os.path.join(root_save, dir_name)) == False:
    os.makedirs(os.path.join(root_save, dir_name))
    os.makedirs(os.path.join(root_save, dir_name, dir_name+"_preds"))
    os.makedirs(os.path.join(root_save, dir_name, dir_name+"_tif"))
save_path = os.path.join(root_save, dir_name, dir_name+"_tif")"""

for dir_name in os.listdir(im7_path):

    if os.path.exists(os.path.join(save_path, dir_name)) == False:
        os.makedirs(os.path.join(save_path, dir_name))
    count = 0
    print(f"Starting to convert from directory: {dir_name}. Total {len(os.listdir(os.path.join(im7_path, dir_name)))} images.")
    for file in os.listdir(os.path.join(im7_path, dir_name)):
        count +=1
        v_array = read_im7(os.path.join(im7_path, dir_name, file))
        v_array = v_array[0]
        im_pil = Image.fromarray(v_array, mode= "L")
        file_tif = file.split(".")[0]
        file_tif = file_tif + ".tif"


        im_pil.save(os.path.join(save_path, dir_name,file_tif))
        print(f"[{count}/{len(os.listdir(os.path.join(im7_path, dir_name)))}]Done converting {file_tif} from im7 to tif.")


#v_array = v_array[0]
#v_array8 = v_array.astype("uint8")



#plt.imshow(v_array8, cmap="gray")
#plt.show()



# both uint16 and uint8 look the same...
#im_pil = Image.fromarray(v_array)
#im_pil.save(r"C:\Users\VEYSEL\OneDrive - Universitaet Duisburg-Essen\Desktop\r\im7_tiff_via_python.tif")
