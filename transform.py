import random
import torchvision.transforms as t
from torchvision.transforms import functional as F
from src.utils import dboxes300_coco, calc_iou_tensor, Encoder
import torch


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image,clear, target=None):
        for trans in self.transforms:
            image,clear, target = trans(image,clear, target)
        return image, clear,target


class ToTensor(object):
    def __call__(self, image, clear,target):
        image = F.to_tensor(image).contiguous()
        clear=F.to_tensor(clear).contiguous()
        return image, clear,target

class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image,clear, target):
        if random.random() < self.prob:
            image = image.flip(-1)
            clear=clear.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = 1.0 - bbox[:, [2, 0]]
            target["boxes"] = bbox
        return image,clear, target
class SSDCropping(object):
    def __init__(self):
        self.sample_options = (
            # Do nothing
            None,
            # min IoU, max IoU
            (0.1, None),
            (0.3, None),
            (0.5, None),
            (0.7, None),
            (0.9, None),
            # no IoU requirements
            (None, None),
        )
        self.dboxes = dboxes300_coco()

    def __call__(self, image, clear,target):
        # Ensure always return cropped image
        while True:
            mode = random.choice(self.sample_options)
            if mode is None:
                return image,clear, target

            htot, wtot = target['height_width']

            min_iou, max_iou = mode
            min_iou = float('-inf') if min_iou is None else min_iou
            max_iou = float('+inf') if max_iou is None else max_iou

            # Implementation use 5 iteration to find possible candidate
            for _ in range(5):
                # 0.3*0.3 approx. 0.1
                w = random.uniform(0.3, 1.0)
                h = random.uniform(0.3, 1.0)

                if w/h < 0.5 or w/h > 2:
                    continue

                # left 0 ~ wtot - w, top 0 ~ htot - h
                left = random.uniform(0, 1.0 - w)
                top = random.uniform(0, 1.0 - h)

                right = left + w
                bottom = top + h
                bboxes = target["boxes"]
                ious = calc_iou_tensor(bboxes, torch.tensor([[left, top, right, bottom]]))

                if not ((ious > min_iou) & (ious < max_iou)).all():
                    continue

                # discard any bboxes whose center not in the cropped image
                xc = 0.5 * (bboxes[:, 0] + bboxes[:, 2])
                yc = 0.5 * (bboxes[:, 1] + bboxes[:, 3])
                masks = (xc > left) & (xc < right) & (yc > top) & (yc < bottom)
                if not masks.any():
                    continue


                bboxes[bboxes[:, 0] < left, 0] = left
                bboxes[bboxes[:, 1] < top, 1] = top
                bboxes[bboxes[:, 2] > right, 2] = right
                bboxes[bboxes[:, 3] > bottom, 3] = bottom


                bboxes = bboxes[masks, :]
                labels = target['labels']
                labels = labels[masks]

                left_idx = int(left * wtot)
                top_idx = int(top * htot)
                right_idx = int(right * wtot)
                bottom_idx = int(bottom * htot)
                image = image.crop((left_idx, top_idx, right_idx, bottom_idx))
                clear=clear.crop((left_idx, top_idx, right_idx, bottom_idx))

                bboxes[:, 0] = (bboxes[:, 0] - left) / w
                bboxes[:, 1] = (bboxes[:, 1] - top) / h
                bboxes[:, 2] = (bboxes[:, 2] - left) / w
                bboxes[:, 3] = (bboxes[:, 3] - top) / h


                target['boxes'] = bboxes
                target['labels'] = labels

                return image,clear, target


class Resize(object):
    def __init__(self, size=(300, 300)):
        self.resize = t.Resize(size)

    def __call__(self, image, clear,target):
        image = self.resize(image)
        clear=self.resize(clear)
        return image,clear, target


class ColorJitter(object):
    def __init__(self, brightness=0.125, contrast=0.5, saturation=0.5, hue=0.05):
        self.trans = t.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, image, clear,target):
        image = self.trans(image)
        clear=self.trans(clear)
        return image,clear, target


class Normalization(object):

    def __init__(self, mean=None, std=None):
        if mean is None:
            mean = [0.485, 0.456, 0.406]
        if std is None:
            std = [0.229, 0.224, 0.225]
        self.normalize = t.Normalize(mean=mean, std=std)
        self.normalizes=t.Normalize(mean=mean, std=std)
    def __call__(self, image, clear,target):
        image = self.normalize(image)
        clear=self.normalizes(clear)
        return image, clear,target

class DeNormalization(object):
    def __init__(self, mean=None, std=None):
        if mean is None:
            mean = [-0.485/0.229, -0.456/0.224, -0.406/0.225]
        if std is None:
            std = [1/0.229, 1/0.224, 1/0.225]
        self.normalize = t.Normalize(mean=mean, std=std)
        self.normalizes=t.Normalize(mean=mean, std=std)
    def __call__(self, image, clear,target):
        image = self.normalize(image)
        clear=self.normalizes(clear)
        return image, clear,target






class AssignGTtoDefaultBox(object):
    def __init__(self):
        self.default_box = dboxes300_coco()
        self.encoder = Encoder(self.default_box)

    def __call__(self, image, clear,target):
        boxes = target['boxes']
        labels = target["labels"]
        bboxes_out, labels_out = self.encoder.encode(boxes, labels)
        target['boxes'] = bboxes_out
        target['labels'] = labels_out

        return image, clear,target
