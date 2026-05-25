import torch
import matplotlib.pyplot as plt
import numpy as np
import random
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from typing import Sequence, List, Optional
from PIL import Image, ImageEnhance
import os
from pathlib import Path
import cv2
import torch.nn.functional as F
from torchvision.transforms.functional import normalize,rotate
from torchvision.transforms import ColorJitter
import glob
from tqdm import tqdm

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")


def get_files(PATH):
    file_lan = []
    if type(PATH) is str:
        for filepath,dirnames,filenames in os.walk(PATH):
            for filename in filenames:
                file_lan.append(os.path.join(filepath,filename))
    elif type(PATH) is list:
        for path in PATH:
            for filepath,dirnames,filenames in os.walk(path):
                for filename in filenames:
                    file_lan.append(os.path.join(filepath,filename))
    return file_lan


def _swap_dir(image_path: str, source_dirname: str, target_dirname: str) -> Optional[str]:
    """Swap the last `<source_dirname>` directory in `image_path` with `<target_dirname>`."""
    parts = list(Path(image_path).parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == source_dirname:
            parts[i] = target_dirname
            return str(Path(*parts))
    return None


def _find_aux_file(image_path: str, source_dirname: str, target_dirname: str) -> Optional[str]:
    """Locate a sibling file under target_dirname matching image_path's stem.

    Tries the exact filename first, then every common image extension. Returns
    None if no candidate exists on disk.
    """
    swapped = _swap_dir(image_path, source_dirname, target_dirname)
    if swapped is None:
        return None
    if os.path.exists(swapped):
        return swapped
    p = Path(swapped)
    stem = p.stem
    parent = p.parent
    for ext in IMAGE_EXTS:
        candidate = parent / f"{stem}{ext}"
        if candidate.exists():
            return str(candidate)
    return None


def _find_depth(image_path: str, variant_dir: str, fallback_dir: str) -> Optional[str]:
    """Try variant_dir first, then fallback_dir, then return None."""
    for d in (variant_dir, fallback_dir):
        if d is None:
            continue
        path = _find_aux_file(image_path, "images", d)
        if path is not None:
            return path
    return None

class GOSrandomAffine(object):
    def __init__(self, prob=0.5):
        self.prob = prob
        self.transform = transforms.RandomAffine(degrees=30, translate=(0,0.25), scale=(0.8,1.2), shear=15, fill=0) 

    def __call__(self, sample):
        if random.random() < self.prob:
            image, gt = sample['image'], sample['gt']
            
            # Apply random perspective transform to both image and ground truth
            im = torch.cat([image,gt],dim=0)
            im = self.transform(im)
            # gt = self.transform(gt)
            image = im[:3,:,:]
            gt = im[3:,:,:]
            
            sample['image'] = image
            sample['gt'] = gt
        
        return sample

class GOSrandomPerspective(object):
    def __init__(self, prob=0.5, distortion_scale=0.5, p=1.0):
        self.prob = prob
        self.transform = transforms.RandomPerspective(distortion_scale=distortion_scale, p=p)

    def __call__(self, sample):
        if random.random() < self.prob:
            image, gt = sample['image'], sample['gt']
            
            # Apply random perspective transform to both image and ground truth
            im = torch.cat([image,gt],dim=0)
            im = self.transform(im)
            # gt = self.transform(gt)
            image = im[:3,:,:]
            gt = im[3:,:,:]
            
            sample['image'] = image
            sample['gt'] = gt
        
        return sample

class GOSGaussianNoise(object):
    def __init__(self, max_std=0.2, prob=0.5):
        super().__init__()
        self.max_std = max_std
        self.prob = prob
    def __call__(self, sample):
        if random.random() < self.prob:
            image =  sample['image']
            noise = torch.randn(image.shape) * torch.rand(1) * self.max_std
            noisy_img_tensor = image + noise
            sample['image'] = noisy_img_tensor
        return sample

def rotate_and_crop(img, angle):
    rotated_img = transforms.functional.rotate(img, angle)
    _, h_orig, w_orig = img.shape
    theta = abs(torch.tensor(np.radians(angle % 180)))
    if theta > torch.pi/2:
        theta = torch.pi - theta
    new_h = int(h_orig /(torch.cos(theta) + torch.sin(theta)))
    new_w = new_h
    top = (h_orig - new_h) // 2
    left = (h_orig - new_w) // 2
    cropped_img = rotated_img[:, top:top+new_h, left:left+new_w]
    cropped_img = transforms.functional.resize(cropped_img, (h_orig, w_orig))
    return cropped_img

class GOSrandomRotation(object):
    def __init__(self,prob=0.5):
        self.prob = prob

    def __call__(self,sample):
        
        if random.random() < self.prob:
            image, gt, depth =  sample['image'], sample['gt'], sample['depth']
            depth_large = sample['depth_large']
            random_angle = np.random.randint(-30, 30)
            # print(image.shape)
            image = rotate_and_crop(image,random_angle)
            gt = rotate_and_crop(gt,random_angle)
            depth = rotate_and_crop(depth,random_angle)
            depth_large = rotate_and_crop(depth_large,random_angle)
            sample['image'] = image
            sample['gt'] = gt
            sample['depth'] = depth
            sample['depth_large'] = depth_large
        return sample

class GOSColorEnhance(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        if random.random() < self.prob:
            image = sample['image'] * 255.0
            # print(image.max())
            image = Image.fromarray(np.uint8(image.permute(1,2,0))).convert('RGB')

            bright_intensity = random.randint(5, 15) / 10.0
            image = ImageEnhance.Brightness(image).enhance(bright_intensity)
            
            contrast_intensity = random.randint(5, 15) / 10.0
            image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
            
            color_intensity = random.randint(0, 20) / 10.0
            image = ImageEnhance.Color(image).enhance(color_intensity)
            
            sharp_intensity = random.randint(0, 30) / 10.0
            image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)

            image = torch.from_numpy(np.array(image)).permute(2,0,1).float() / 255.0 
            sample['image'] = image
        
        return sample

class GOSColorJitter(object):
    def __init__(self, prob=0.5):
        self.prob = prob
        self.ColorJitter = ColorJitter(0.1,0.1,0.1,0.1)

    def __call__(self,sample):
        
        if random.random() < self.prob:
            image = sample['image']
            image = self.ColorJitter(image)
            sample['image'] = image
        
        return sample
    
class GOSRandomUPCrop(object):
    def __init__(self, prob=0.5, border=30):
        self.prob = prob
        self.border = border
    def __call__(self,sample):
        # flag = 1
        if random.random() < self.prob:
            image = sample['image']
            gt = sample['gt']
            depth = sample['depth']
            depth_large = sample['depth_large']
            
            image_height, image_width = image.shape[-2], image.shape[-1]
            
            crop_win_width = torch.randint(image_width - self.border, image_width, (1,)).item()
            crop_win_height = torch.randint(image_height - self.border, image_height, (1,)).item()

            x_start = (image_width - crop_win_width) // 2
            y_start = (image_height - crop_win_height) // 2
            x_end = x_start + crop_win_width
            y_end = y_start + crop_win_height

            cropped_image = image[..., y_start:y_end, x_start:x_end]
            cropped_gt = gt[..., y_start:y_end, x_start:x_end]
            cropped_depth = depth[..., y_start:y_end, x_start:x_end]
            cropped_depth_large = depth_large[..., y_start:y_end, x_start:x_end]
            image_cropped = F.interpolate(cropped_image[None,...],size=[image.shape[1], image.shape[2]],mode='bilinear',align_corners=True)[0]
            gt_cropped = F.interpolate(cropped_gt[None,...],size=[image.shape[1], image.shape[2]],mode='bilinear',align_corners=True)[0]
            depth_cropped = F.interpolate(cropped_depth[None,...],size=[image.shape[1], image.shape[2]],mode='bilinear',align_corners=True)[0]
            cropped_depth_large = F.interpolate(cropped_depth_large[None,...],size=[image.shape[1], image.shape[2]],mode='bilinear',align_corners=True)[0]
            sample['image'] = image_cropped
            sample['gt'] = gt_cropped
            sample['depth'] = depth_cropped
            sample['depth_large'] = cropped_depth_large
        return sample



class GOSNormalize(object):
    def __init__(self, mean=[0.485,0.456,0.406,0], std=[0.229,0.224,0.225,1.0]):
        self.mean = mean
        self.std = std

    def __call__(self,sample):
        image = sample['image']
        image = normalize(image,self.mean,self.std)
        sample['image'] = image
        return sample

class GOSMAXNormalize(object):
    def __init__(self):
        pass
    def __call__(self,sample):
        image =  sample['image']
        image = (image-image.min()) / (image.max() - image.min())
        sample['image'] = image
        return sample

class GOSRandomHFlip(object):
    def __init__(self,prob=0.5):
        self.prob = prob
    def __call__(self,sample):
        # random horizontal flip
        if random.random() <= self.prob:
            image, gt =  sample['image'], sample['gt']
            depth, depth_large = sample['depth'],sample['depth_large']
            image = torch.flip(image,dims=[2])
            gt = torch.flip(gt,dims=[2])
            depth = torch.flip(depth,dims=[2])
            depth_large = torch.flip(depth_large,dims=[2])
            sample['image'] = image
            sample['gt'] = gt
            sample['depth'] = depth
            sample['depth_large'] = depth_large
        return sample

class GOSRandomimg2Grayedge(object):
    def __init__(self,prob=0.5):
        self.prob = prob
    def __call__(self,sample):
        if random.random() <= self.prob:
            gt = sample['gt'][None,...]

            kernel_x = torch.tensor([[-1, 0, 1],
                                    [-2, 0, 2],
                                    [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            kernel_y = torch.tensor([[-1, -2, -1],
                                    [0, 0, 0],
                                    [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)


            edges_x = torch.nn.functional.conv2d(gt, kernel_x, padding=1)
            edges_y = torch.nn.functional.conv2d(gt, kernel_y, padding=1)


            edges = torch.sqrt(edges_x ** 2 + edges_y ** 2).repeat(1,3,1,1)
            sample['image'] = edges[0]
        return sample

class GOSRandombackground2same(object):
    def __init__(self,prob=0.5):
        self.prob = prob
    def __call__(self,sample):
        if random.random() <= self.prob:
            im = sample['image']
            gt = sample['gt']
            objects = im*gt
            objects_mean_rgb = objects.mean(dim=(1,2),keepdim=True)
            same_back_ground = objects_mean_rgb*(1-gt)
            sample['image'] = objects + same_back_ground
        return sample
    
class GOSRandombackground2edgesame(object):
    def __init__(self,prob=0.5):
        self.prob = prob
    def __call__(self,sample):
        if random.random() <= self.prob:
            im = sample['image']
            gt = sample['gt']
            edges = abs(gt[None,...]-F.avg_pool2d(gt[None,...],kernel_size=31,stride=1,padding=15))[0]
            edges = (edges - edges.min()) / (edges.max() - edges.min())
            avg_im = F.avg_pool2d(im[None,...],kernel_size=31,stride=1,padding=15)[0]
            sample['image'] = im * (1-edges) + avg_im * edges
        return sample

class GOSRandomGray(object):
    def __init__(self,prob=0.5):
        self.prob = prob
        self.transform = transforms.Grayscale(num_output_channels=3)
    def __call__(self,sample):
        if random.random() <= self.prob:
            image = sample['image']
            image = self.transform(image)
            sample['image'] = image
        return sample

class GOSTorchRandomCrop(object):
    def __init__(self, prob=0.5, border=30):
        self.prob = prob
        self.border = border

    def __call__(self, sample):
        if random.random() < self.prob:
            image = sample['image']
            gt = sample['gt']


            image_height, image_width = image.shape[1], image.shape[2]

            crop_win_width = np.random.randint(image_width - self.border, image_width)
            crop_win_height = np.random.randint(image_height - self.border, image_height)


            top = (image_height - crop_win_height) // 2
            left = (image_width - crop_win_width) // 2
            bottom = top + crop_win_height
            right = left + crop_win_width

            image_cropped = image[:, top:bottom, left:right]
            gt_cropped = gt[:, top:bottom, left:right]

            sample['image'] = image_cropped
            sample['gt'] = gt_cropped

        return sample

class MyDataset(Dataset):
    """DIS-style segmentation dataset with RGB + pseudo-depth.

    Expected on-disk layout under each `root` (directory passed in):
        <root>/.../<name>.<ext>                    (image)
        ../masks/<name>.png                         (GT mask; only if use_gt)
        ../{depth_small,depth_base,depth_large}/<name>.png  (RGB-paired depth variants)
        ../depth_large_1024/<name>.png             (training-only depth GT for SiLog loss)

    Any missing depth variant falls back to a single `depth/` directory next to
    `images/`. Missing files raise clear errors at __getitem__ time.

    Args:
        depth_variants: dir names to sample from for the RGB-paired depth input
            during training. Defaults to ('depth_large','depth_base','depth_small').
            At eval time only the first variant (or `depth_fallback_dir`) is used.
        depth_gt_dir: dir name holding the supervision target for the SiLog
            depth-head loss (training only). Defaults to 'depth_large_1024'.
        depth_fallback_dir: dir name used when variant/GT lookups miss.
            Default 'depth'.
        labels_from_filename: if True (default), one-hot labels are parsed from
            '#'-separated filename tokens (DIS-5K convention). If False, labels
            are zeros (use this for custom data without DIS naming).
    """
    _depth_synth_warned = False  # class-level: print warning once across instances

    def __init__(self,root=None,transform=[],chached=False,size=[224,224],stoi=None,
                 istrain=0,use_gt=True,
                 depth_variants=('depth_large','depth_base','depth_small'),
                 depth_gt_dir='depth_large_1024',
                 depth_fallback_dir='depth',
                 labels_from_filename=True,
                 pair_list=None,
                 synthesize_missing_depth=False):
        self.istrain = istrain
        # pair_list = [{'image': ..., 'mask': ..., 'depth': ... (optional)}]
        self.pairs = pair_list
        if pair_list is not None:
            self.imlists = [p['image'] for p in pair_list]
        else:
            if root is None:
                raise ValueError("Either `root` or `pair_list` must be provided.")
            self.imlists = get_files(root)
        self.transforms = transforms.Compose(transform)
        self.chached = chached
        self.size = size
        self.use_gt = use_gt
        self.depth_variants = tuple(depth_variants) if depth_variants else ()
        self.depth_gt_dir = depth_gt_dir
        self.depth_fallback_dir = depth_fallback_dir
        self.labels_from_filename = labels_from_filename and use_gt
        self.synthesize_missing_depth = synthesize_missing_depth
        if use_gt and self.labels_from_filename:
            if stoi is None:
                label_chache = []
                for i in range(len(self.imlists)):
                    label_chache.append(''.join(self.imlists[i].split('/')[-1].split('#')[0:3]))
                    label_chache = sorted(list(set(label_chache)))
                self.stoi = { ch:i for i,ch in enumerate(label_chache) }
            else:
                self.stoi = stoi
        else:
            self.stoi = {"0":0}
        if self.chached:
            self.imlists_chache = []
            self.gt_chache = []
            self.raw_size = []
            for im in tqdm(range(len(self.imlists))):
                tmpimg = cv2.cvtColor(cv2.imread(self.imlists[im]),cv2.COLOR_BGR2RGB)
                self.raw_size.append(tmpimg.shape)

                tmpimg = F.interpolate(torch.from_numpy(tmpimg).permute(2,0,1)[None,...],size=size,mode='bilinear',align_corners=True)[0]
                self.imlists_chache.append(tmpimg)
                if use_gt:
                    tmplabel = cv2.cvtColor(cv2.imread(self.imlists[im].replace('/images','/masks').replace('.jpg','.png')),cv2.COLOR_BGR2GRAY)
                    tmplabel = F.interpolate(torch.from_numpy(tmplabel)[None,None,...],size=size,mode='bilinear', align_corners=True)[0][0]
                else:
                    tmplabel = torch.zeros([1,size[0],size[1]])
                self.gt_chache.append(tmplabel)

    def __getitem__(self, index):
        
        if self.chached:
            im = self.imlists_chache[index]
            gt = self.gt_chache[index]
            raw_size = self.raw_size[index][:2]
            one_hot_label = torch.zeros([len(self.stoi)])
            one_hot_label[self.stoi[''.join(self.imlists[index].split('/')[-1].split('#')[0:3])]] = 1
            label =  one_hot_label
        else:
            image_path = self.imlists[index]
            pair = self.pairs[index] if self.pairs is not None else None

            im = cv2.cvtColor(cv2.imread(image_path),cv2.COLOR_BGR2RGB)
            raw_size = im.shape[:2]
            im_orig_gray = None  # populated lazily if we need to synthesize depth
            im = F.interpolate(torch.from_numpy(im).permute(2,0,1)[None,...],size=self.size,mode='bilinear',align_corners=True)[0]

            if self.use_gt:
                mask_path = (pair.get('mask') if pair is not None else None) \
                            or _find_aux_file(image_path, "images", "masks")
                if mask_path is None:
                    raise FileNotFoundError(
                        f"No mask found for {image_path} "
                        f"(searched sibling 'masks/' directory)."
                    )
                gt = cv2.cvtColor(cv2.imread(mask_path),cv2.COLOR_BGR2GRAY)
                gt = F.interpolate(torch.from_numpy(gt)[None,None,...],size=self.size,mode='nearest')[0][0]
            else:
                gt = torch.zeros([1,self.size[0],self.size[1]])

            one_hot_label = torch.zeros([len(self.stoi)])
            if self.use_gt and self.labels_from_filename:
                key = ''.join(image_path.split('/')[-1].split('#')[0:3])
                if key in self.stoi:
                    one_hot_label[self.stoi[key]] = 1
            label =  one_hot_label

            # ---- depth input ----
            # Priority: explicit `pair['depth']`, then sibling variant dir, then fallback dir,
            # then (if enabled) synthesize from grayscale of the input image.
            if self.istrain and len(self.depth_variants) > 1:
                variant = random.choice(self.depth_variants)
            elif len(self.depth_variants) >= 1:
                variant = self.depth_variants[0]
            else:
                variant = None
            depth_path = (pair.get('depth') if pair is not None else None) \
                         or _find_depth(image_path, variant, self.depth_fallback_dir)
            if depth_path is not None:
                depth = cv2.cvtColor(cv2.imread(depth_path),cv2.COLOR_BGR2GRAY)
                depth = F.interpolate(torch.from_numpy(depth)[None,None,...],size=self.size,mode='bilinear',align_corners=True)[0]
            elif self.synthesize_missing_depth:
                if not MyDataset._depth_synth_warned:
                    print(f"[MyDataset] No depth files found; synthesizing pseudo-depth "
                          f"from RGB grayscale. Generate real depth maps via "
                          f"DAM_V2/Depth-prepare.ipynb for best results.")
                    MyDataset._depth_synth_warned = True
                # Grayscale of the (already resized) image as a stand-in depth map.
                im_orig_gray = (0.299 * im[0] + 0.587 * im[1] + 0.114 * im[2])[None, ...]
                depth = im_orig_gray.clone()
            else:
                raise FileNotFoundError(
                    f"No depth map found for {image_path}. Tried sibling "
                    f"'{variant}/' and '{self.depth_fallback_dir}/'. Pass "
                    f"synthesize_missing_depth=True (or --synthesize_depth) to use "
                    f"grayscale of the RGB image, or generate real depth maps via "
                    f"DAM_V2/Depth-prepare.ipynb."
                )

            if self.istrain:
                gt_depth_path = _find_depth(image_path, self.depth_gt_dir, self.depth_fallback_dir)
                if gt_depth_path is not None:
                    large_depth = cv2.cvtColor(cv2.imread(gt_depth_path),cv2.COLOR_BGR2GRAY)
                    large_depth = F.interpolate(torch.from_numpy(large_depth)[None,None,...],size=self.size,mode='bilinear',align_corners=True)[0]
                    large_depth = torch.divide(large_depth,255.0)
                else:
                    # No depth GT available: reuse the input depth (degenerates SiLog auxiliary loss).
                    # depth is still 0-255 here; the divide further below applies only to `depth`.
                    large_depth = torch.divide(depth.clone(), 255.0)
        # depth = torch.zeros_like(im)
        im = torch.divide(im,255.0)
        gt = torch.divide(gt,255.0)
        depth = torch.divide(depth,255.0)
        sample = {
            'image_name':self.imlists[index],
            'image_size':raw_size,
            "image": im.float(),
            "gt": gt.float()[None,...],
            "depth": depth.float(),
            "depth_large": large_depth.float() if self.istrain else depth.float(),
            "label": label,
        }
        if self.transforms:
            sample = self.transforms(sample)
        return sample
    
    def __len__(self):
        return self.imlists.__len__()
    
def build_dataset(is_train,args):
    if is_train:
        train_data_path = [
            args.data_path+'/DIS-TR/images',
                           ]
        return MyDataset(train_data_path,transform=[
            #方位
            GOSRandomHFlip(0.5),
            GOSrandomRotation(0.5),
            #颜色
            GOSColorEnhance(0.5),
            GOSRandomGray(0.25),
            #放大
            GOSRandomUPCrop(0.5),
            #标准化
            GOSNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ],chached=args.chached,size=[args.input_size,args.input_size],istrain=True)
    else:
        valid_data_path = args.data_path+'/DIS-VD/images'
        return MyDataset(valid_data_path,transform=[GOSNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])],chached=args.chached,size=[args.input_size,args.input_size])
def _load_pair_list_from_csv(csv_path: str, data_root: str):
    """Read CSV with columns image_path, mask_path, depth_path (optional), dataset.

    Paths are joined with `data_root` if they aren't absolute.
    """
    import csv as _csv
    pairs = []
    with open(csv_path, newline='') as f:
        for row in _csv.DictReader(f):
            img = row['image_path']
            msk = row.get('mask_path') or None
            dep = row.get('depth_path') or None
            if not os.path.isabs(img):
                img = os.path.join(data_root, img)
            if msk and not os.path.isabs(msk):
                msk = os.path.join(data_root, msk)
            if dep and not os.path.isabs(dep):
                dep = os.path.join(data_root, dep)
            pairs.append({
                'image': img,
                'mask': msk,
                'depth': dep,
                'dataset': row.get('dataset', ''),
            })
    return pairs


def _split_pairs(pairs, val_frac: float, seed: int):
    """Shuffle deterministically by seed and split into (train, val)."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(pairs))
    rng.shuffle(idx)
    n_val = int(round(len(pairs) * val_frac))
    val_idx = set(idx[:n_val].tolist())
    train, val = [], []
    for i, pair in enumerate(pairs):
        (val if i in val_idx else train).append(pair)
    return train, val


def build_csv_dataset(is_train, args):
    """Dataset builder that reads (image, mask, depth?) pairs from a CSV
    and applies an 80/20 (configurable) train/val split.

    Required args:
        --csv_path: path to the CSV (see build_dataset_csv.py)
        --data_path: prepended to non-absolute paths in the CSV (default ./data)
        --val_split: validation fraction (default 0.2)
        --csv_split_seed: RNG seed for the shuffle (default 42)
        --synthesize_depth: if True, fall back to grayscale of the RGB image
            when no depth file is found (good for getting started before
            DAM-V2 has been run).
    """
    csv_path = getattr(args, 'csv_path', 'data/index.csv')
    val_frac = float(getattr(args, 'val_split', 0.2))
    seed = int(getattr(args, 'csv_split_seed', 42))
    synthesize_depth = bool(getattr(args, 'synthesize_depth', False))
    labels_from_filename = bool(getattr(args, 'labels_from_filename', False))

    pairs = _load_pair_list_from_csv(csv_path, args.data_path)
    train_pairs, val_pairs = _split_pairs(pairs, val_frac, seed)
    chosen = train_pairs if is_train else val_pairs
    print(f"[csv-dataset] {csv_path}: {len(pairs)} total -> "
          f"train={len(train_pairs)} val={len(val_pairs)} (val_frac={val_frac}, seed={seed})")

    transform = [
        GOSRandomHFlip(0.5),
        GOSrandomRotation(0.5),
        GOSColorEnhance(0.5),
        GOSRandomGray(0.25),
        GOSRandomUPCrop(0.5),
        GOSNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ] if is_train else [
        GOSNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

    return MyDataset(
        pair_list=chosen,
        transform=transform,
        chached=args.chached,
        size=[args.input_size, args.input_size],
        istrain=is_train,
        depth_variants=getattr(args, 'depth_variants',
                               ('depth_large', 'depth_base', 'depth_small')),
        depth_gt_dir=getattr(args, 'depth_gt_dir', 'depth_large_1024'),
        depth_fallback_dir=getattr(args, 'depth_fallback_dir', 'depth'),
        labels_from_filename=labels_from_filename,
        synthesize_missing_depth=synthesize_depth,
    )


def build_finetune_dataset(is_train, args):
    """Dataset builder for fine-tuning on a user-provided directory layout.

    Expects:
        <args.data_path>/<args.train_subdir>/images/...
        <args.data_path>/<args.train_subdir>/masks/...
        <args.data_path>/<args.train_subdir>/depth/...           (single depth dir is fine)
        <args.data_path>/<args.val_subdir>/images/...
        <args.data_path>/<args.val_subdir>/masks/...
        <args.data_path>/<args.val_subdir>/depth/...

    Optional per-variant depth dirs (depth_small/depth_base/depth_large/
    depth_large_1024) are auto-detected; missing ones fall back to depth/.
    """
    train_subdir = getattr(args, 'train_subdir', 'train')
    val_subdir = getattr(args, 'val_subdir', 'val')
    depth_variants = getattr(args, 'depth_variants',
                             ('depth_large','depth_base','depth_small'))
    depth_gt_dir = getattr(args, 'depth_gt_dir', 'depth_large_1024')
    depth_fallback_dir = getattr(args, 'depth_fallback_dir', 'depth')
    labels_from_filename = getattr(args, 'labels_from_filename', False)

    subdir = train_subdir if is_train else val_subdir
    images_root = os.path.join(args.data_path, subdir, 'images')
    if not os.path.isdir(images_root):
        raise FileNotFoundError(
            f"Expected images at {images_root}. Layout under {args.data_path} should be "
            f"{subdir}/images/, {subdir}/masks/, {subdir}/depth/ (or per-variant depth dirs)."
        )

    transform = []
    if is_train:
        transform = [
            GOSRandomHFlip(0.5),
            GOSrandomRotation(0.5),
            GOSColorEnhance(0.5),
            GOSRandomGray(0.25),
            GOSRandomUPCrop(0.5),
            GOSNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    else:
        transform = [GOSNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]

    return MyDataset(
        images_root,
        transform=transform,
        chached=args.chached,
        size=[args.input_size, args.input_size],
        istrain=is_train,
        depth_variants=depth_variants,
        depth_gt_dir=depth_gt_dir,
        depth_fallback_dir=depth_fallback_dir,
        labels_from_filename=labels_from_filename,
    )


def keep_n_files(directory, n=3):
    files = [(file_path, os.path.getmtime(file_path)) for file_path in glob.glob(os.path.join(directory, '*'))]
    
    files.sort(key=lambda x: x[1], reverse=True)
    
    for file_path, _ in files[n:]:

        os.remove(file_path)
