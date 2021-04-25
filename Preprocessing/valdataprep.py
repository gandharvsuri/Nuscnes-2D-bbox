import pandas as pd
import numpy as np
from tqdm import tqdm

print("Loading Metadata............\n\n")
sample_data = pd.read_json("../Nuscenes Data/nuimages-v1.0-all-metadata/v1.0-val/sample_data.json")
category = pd.read_json("../Nuscenes Data/nuimages-v1.0-all-metadata/v1.0-val/category.json")
category["index"] = list(range(len(category)))
category_index = dict(zip(category["token"],category["index"]))

object_ann = pd.read_json("../Nuscenes Data/nuimages-v1.0-all-metadata/v1.0-val/object_ann.json")

print("Creating Label Files............\n\n")
for i, row in tqdm(sample_data.iterrows()):
    filename = row["filename"].split("/")

    if filename[1] == "CAM_FRONT":
        fname = filename[2].replace("."+row["fileformat"], ".txt")
        path =  "../Nuscenes Data/labels/val/" + fname
        f = open(path, "w")

        object_df = object_ann[object_ann["sample_data_token"] == row["token"]]
        flag = len(object_df)
        bbox = object_df["bbox"].values
        
        classes = object_df["category_token"].values
        classes = [category_index[token] for token in classes]
        x_center = []
        y_center = []
        width = []
        height = []

        for x1,y1,x2,y2 in bbox:
            x_center.append(np.mean([x1,x2]))
            y_center.append(np.mean([y1,y2]))
            width.append(abs(x1-x2))
            height.append(abs(y1-y2))

        x_center = np.array(x_center)/1600
        y_center = np.array(y_center)/900
        width = np.array(width)/1600
        height = np.array(height)/900

        assert len(classes) == len(x_center)
        assert len(x_center) == len(y_center)
        assert len(y_center) == len(width)
        assert len(width) == len(height)

        label = list(zip(classes,x_center,y_center,width,height))

        for c,x,y,w,h in label:
            f.write("{} {} {} {} {} ".format(c,x,y,w,h))
        
        f.close()
        
print("Done!\n")