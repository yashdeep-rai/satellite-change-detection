
import torch

def iou_score(pred, target, threshold=0.5):

    pred = (pred > threshold).float()

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection

    if union == 0:
        return 0

    return (intersection / union).item()
