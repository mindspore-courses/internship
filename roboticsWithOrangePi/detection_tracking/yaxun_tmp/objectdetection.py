import os
import sys
import numpy as np
from PIL import Image
import torch
import supervision as sv
#sys.path.append("../GroundingDINO/")
from groundingdino.util.inference import load_model, predict, annotate
import groundingdino.datasets.transforms as T
code_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(code_dir)
CONFIG_PATH = f'{code_dir}/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'
WEIGHTS_PATH= f'{code_dir}/GroundingDINO/weights/groundingdino_swint_ogc.pth'


# CONFIG_PATH = GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py
def lodegroundingDINO():
    model = load_model(CONFIG_PATH, WEIGHTS_PATH)
    return model

def detectobject(prompt,img,model):
    #model = lodegroundingDINO()
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image = Image.fromarray(img)
    image_transformed, _ = transform(image, None)
    BOX_TRESHOLD = 0.35
    TEXT_TRESHOLD = 0.25

    boxes, logits, phrases = predict(
        model=model,
        image=image_transformed,
        caption=prompt,
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD
    )

    annotated_frame = annotate(image_source=img, boxes=boxes, logits=logits, phrases=phrases)

    sv.plot_image(annotated_frame, (16, 16))

    return boxes

def generate_points_in_box(boxe):
    x_c = boxe[0]
    y_c = boxe[1]
    w = boxe[2]
    h = boxe[3]
    x_min =int(x_c - 0.5 * w) 
    y_min = int(y_c - 0.5 * h)
    x_max = int(x_c + 0.5 * w)
    y_max = int(y_c + 0.5 * h)
    points = [(x, y) for x in range(x_min, x_max + 1) for y in range(y_min, y_max + 1)]
    return points

def boxes2masks(boxes,width, height):
    #width, height = img.size
    boxes = boxes * torch.Tensor([width, height, width, height])
    boxes_cpu = boxes.detach().cpu()
    boxes_np = boxes_cpu.numpy()
    print(boxes_np)
    masks = []    
    for boxe in boxes_np :
        temp_mask = np.zeros((height, width), dtype=np.uint8)
        for point in generate_points_in_box(boxe) :
            temp_mask[point[1]][point[0]]=255

        masks.append(temp_mask)
    return masks
  
