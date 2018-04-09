import os
import sys
import random
import math
import numpy as np
import skimage.io
import PIL.Image as Image
# import matplotlib
# import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("./")
# from cityscape to coco
target_cityscape = [24, 26, 27, 28, 31, 32, 33]
class_map = {24: 1, 33: 2, 26: 3, 32: 4, 28: 6, 31: 7, 27: 8}
# person, car, truck, bus, train, motorcycle, bicycle    

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
import mrcnn.utils as utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
# if not os.path.exists(COCO_MODEL_PATH):
    # utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = '/mnt/brain1/scratch/didoyang/manipulate_exp/pix2pixHD_mani/results/label2city_1024p/val_latest/images'
class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()
# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

files_heads = [x.rstrip('input_label.jpg') for x in os.listdir(IMAGE_DIR) \
        if x.endswith('input_label.jpg')]

# Load a random image from the images folder
count = 0
UoI_accu = [0, 0, 0, 0, 0, 0] 
for head in files_heads:
    count = count + 1
    if count > 2:
        exit()
        print("exit!!")
    file_name = os.path.join(IMAGE_DIR, head + '_synthesized_image.jpg')
    label_name = os.path.join(IMAGE_DIR, head + '_input_label.jpg')
    image = skimage.io.imread(file_name)
    labels = Image.open(label_name)
    labels_array = np.asarray(labels)
    shape = labels_array.shape

    # Run detection
    results = model.detect([image], verbose=1)

    detected_mask = results[0]['masks']
     
    for index, i in enumerate(target_cityscape):
        temp_bool_mask = np.zeros(shape).astype(np.bool)
        temp_bool_mask[labels_array==i] = True
        class_ids = list(results[0]['class_ids'])
        detected_mask_index = [j for j in \
                range(len(class_ids)) if class_map[i] in class_ids]

        aggregated_masks = detected_mask[:, :, detected_mask_index].sum(axis=2)
        intersect = (aggregated_masks & temp_bool_mask).sum()
        union = (aggregated_masks | temp_bool_mask).sum()
        UoI_accu[index] += intersect / float(union)

UoI = [x/float(count) for x in UoI_accu]

print('''UoI: person: %.4f, car: %.4f, truck: %.4f, bus: %.4f,
        , train: %.4f, motorcycle: %.4f, bicycle: %.4f''' % \
                UoI)



