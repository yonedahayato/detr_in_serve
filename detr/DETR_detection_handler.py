import math
import matplotlib.pyplot as plt
import numpy as np
import requests

from abc import ABC
import base64
from copy import deepcopy
from io import BytesIO
from PIL import Image
from ts.torch_handler.base_handler import BaseHandler
import torch
import torchvision.transforms as T
from ts.torch_handler.base_handler import BaseHandler

def base64_to_pil(image):

    if isinstance(image, str):
        if "base64," in image:
            # DARA URI の場合、data:[<mediatype>][;base64], を除く
            image = image.split(",")[1]

        image = image.replace("_", "/")
        image = image.replace("-", "+")
        image = base64.b64decode(image)

    if isinstance(image, (bytearray, bytes)):

        try:
            image = Image.open(BytesIO(image))

        except:
            image = image.decode("ascii")
            image = base64.b64decode(image)
            image = Image.open(BytesIO(image))

    return image

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

def plot_results(pil_img, prob, boxes):
    """plot_results func
    Args:
        pil_img(PIL.JpegImagePlugin.JpegImageFile)

    Returns:

    """
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()

    colors = COLORS * 100

    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))

    # plt.axis('off')
    ax.axis('off')

    output = BytesIO()
    plt.savefig(output, format='jpg')
    output.seek(0)
    output = base64.b64encode(output.read())

    return output

class DETRDetectionHandler(BaseHandler, ABC):
    """DETRDetectionHandler class
    """

    def __init__(self):
        print("\n== __init__ ==")
        super(DETRDetectionHandler, self).__init__()
        self.initialized = False

    def initialize(self, ctx):
        print("\n== initialize ==")
        self.device = "cpu"

        self.model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
        self.model.eval()

        self.transform = T.Compose([
            T.Resize(800),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def preprocess(self, requests):
        print("\n== preprocess ==")
        print(requests)

        inputs = requests[0]["body"]
        print("inputs")
        print(inputs)

        self.tags = inputs["tags"]
        image = inputs["image"]
        image = base64_to_pil(image)
        self._image = image

        input_ids_batch = [image]

        return input_ids_batch

    def inference(self, input_batch):
        print("\n== inference ==")
        print(input_batch)

        image = input_batch[0]
        input_image = deepcopy(image)

        # mean-std normalize the input image (batch-size: 1)
        image = self.transform(image).unsqueeze(0)

        # propagate through the model
        outputs = self.model(image)

        # keep only predictions with 0.7+ confidence
        probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > 0.9

        # convert boxes from [0; 1] to image scales
        bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], input_image.size)

        inferences = [input_image, image, probas, keep, bboxes_scaled]

        return inferences

    def postprocess(self, inference_output):
        print("\n== postprocess ==")
        print(inference_output)

        input_image, image, probas, keep, bboxes_scaled = inference_output
        output = plot_results(input_image, probas[keep], bboxes_scaled)

        # use lists to store the outputs via up-values
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []

        hooks = [
            self.model.backbone[-2].register_forward_hook(
                lambda self, input, output: conv_features.append(output)
            ),
            self.model.transformer.encoder.layers[-1].self_attn.register_forward_hook(
                lambda self, input, output: enc_attn_weights.append(output[1])
            ),
            self.model.transformer.decoder.layers[-1].multihead_attn.register_forward_hook(
                lambda self, input, output: dec_attn_weights.append(output[1])
            ),
        ]

        # propagate through the model
        outputs = self.model(image)

        for hook in hooks:
            hook.remove()

        # don't need the list anymore
        conv_features = conv_features[0]
        enc_attn_weights = enc_attn_weights[0]
        dec_attn_weights = dec_attn_weights[0]

        # get the feature map shape
        h, w = conv_features['0'].tensors.shape[-2:]

        # fig, axs = plt.subplots(ncols=len(bboxes_scaled), nrows=2, figsize=(22, 7))
        fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(16, 7))
        colors = COLORS * 100

        for idx, ax_i, (xmin, ymin, xmax, ymax) in zip(keep.nonzero(), axs.T, bboxes_scaled):
            xmin = xmin.to("cpu").detach().numpy().copy()
            ymin = ymin.to("cpu").detach().numpy().copy()
            xmax = xmax.to("cpu").detach().numpy().copy()
            ymax = ymax.to("cpu").detach().numpy().copy()

            ax = ax_i[0]
            feature = dec_attn_weights[0, idx].view(h, w).to("cpu").detach().numpy().copy()
            ax.imshow(feature)
            ax.axis('off')
            ax.set_title(f'query id: {idx.item()}')

            ax = ax_i[1]
            ax.imshow(input_image)

            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                    fill=False, color='blue', linewidth=3))
            ax.axis('off')
            ax.set_title(CLASSES[probas[idx].argmax()])
        fig.tight_layout()

        attention_output = BytesIO()
        plt.savefig(attention_output, format='jpg')
        attention_output.seek(0)
        attention_output = base64.b64encode(attention_output.read())

        output = output.decode()
        attention_output = attention_output.decode()

        return [{"detected": output, "attention": attention_output}]
