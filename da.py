import numpy as np
import torch

class RandomCropPaste(object):
    def __init__(self, size, alpha=1.0, flip_p=0.5):
        """Randomly flip and paste a cropped image on the same image. """
        self.size = size
        self.alpha = alpha
        self.flip_p = flip_p

    def __call__(self, img):
        lam = np.random.beta(self.alpha, self.alpha)
        front_bbx1, front_bby1, front_bbx2, front_bby2 = self._rand_bbox(lam)
        img_front = img[:, front_bby1:front_bby2, front_bbx1:front_bbx2].clone()
        front_w = front_bbx2 - front_bbx1
        front_h = front_bby2 - front_bby1

        img_x1 = np.random.randint(0, high=self.size-front_w)
        img_y1 = np.random.randint(0, high=self.size-front_h)
        img_x2 = img_x1 + front_w
        img_y2 = img_y1 + front_h

        if np.random.rand(1) <= self.flip_p:
            img_front = img_front.flip((-1,))
        if np.random.rand(1) <= self.flip_p:
            img = img.flip((-1,))

        mixup_alpha = np.random.rand(1)
        img[:,img_y1:img_y2, img_x1:img_x2] *= mixup_alpha
        img[:,img_y1:img_y2, img_x1:img_x2] += img_front*(1-mixup_alpha)
        return img

    def _rand_bbox(self, lam):
        W = self.size
        H = self.size
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

class CutMix(object):
  def __init__(self, size, beta):
    self.size = size
    self.beta = beta

  def __call__(self, batch):
    img, label = batch
    rand_img, rand_label = self._shuffle_minibatch(batch)
    lambda_ = np.random.beta(self.beta,self.beta)
    r_x = np.random.uniform(0, self.size)
    r_y = np.random.uniform(0, self.size)
    r_w = self.size * np.sqrt(1-lambda_)
    r_h = self.size * np.sqrt(1-lambda_)
    x1 = int(np.clip(r_x - r_w // 2, a_min=0, a_max=self.size))
    x2 = int(np.clip(r_x + r_w // 2, a_min=0, a_max=self.size))
    y1 = int(np.clip(r_y - r_h // 2, a_min=0, a_max=self.size))
    y2 = int(np.clip(r_y + r_h // 2, a_min=0, a_max=self.size))
    img[:, :, x1:x2, y1:y2] = rand_img[:, :, x1:x2, y1:y2]
    
    lambda_ = 1 - (x2-x1)*(y2-y1)/(self.size*self.size)
    return img, label, rand_label, lambda_

  def _shuffle_minibatch(self, batch):
    img, label = batch
    rand_img, rand_label = img.clone(), label.clone()
    rand_idx = torch.randperm(img.size(0))
    rand_img, rand_label = rand_img[rand_idx], rand_label[rand_idx]
    return rand_img, rand_label

# Code: https://github.com/facebookresearch/mixup-cifar10
class MixUp(object):
  def __init__(self, alpha=0.1):
    self.alpha = alpha

  def __call__(self, batch):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    x, y = batch
    lam = np.random.beta(self.alpha, self.alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam
