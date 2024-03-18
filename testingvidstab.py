import os

root = r"D:\Trainingsdaten_v07\Trainingsdaten_v07\full\raw"
#save_in = r"D:\DatenAuswertung_180124\Cavi2\preds_05\weitere spielereien\pit_test\img_stab\vidstab\S59"
for c, frame_name in enumerate(os.listdir(root)):
    print(frame_name)