import os
from segm.model.factory import load_model
from segm.data.factory import create_dataset
from segm.model.factory import create_segmenter 
import segm.utils.torch as ptu
from segm.flops import get_dataset_validation_path, dataset_prepare, InferenceDataset
from torchvision import transforms
from  segm.model.factory import load_model
from segm.data.utils import STATS
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from typing import List, Tuple
import numpy as np
from PIL import Image
import random
import torch
import torch.nn.functional as F
import cv2


try:
    from scipy.ndimage import binary_erosion
except ImportError:
    pass  # Don't fail if scipy is not installed. It's only necessary for this one file.




class InferenceDataset(Dataset):

    def __init__(self, root_dir, transform=None, vis_transform=None, txt_file=None):
        self.root_dir = root_dir
        self.transform = transform
        self.vis_transform = vis_transform

        # If txt_file is provided, read image names from it
        if txt_file:
            with open(txt_file, 'r') as file:
                self.image_files = [os.path.join(root_dir, line.strip() + '.jpg') for line in file.readlines()]
        else:
            # Otherwise, load all image paths from root_dir and subfolders
            self.image_files = self._load_image_paths(root_dir)

    def _load_image_paths(self, root_dir):
        image_files = []
        for dirpath, _, filenames in os.walk(root_dir):
            for file in filenames:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                    image_files.append(os.path.join(dirpath, file))
        return image_files

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")
        image_vis = None
        output = ()
        if self.transform:
            image_tensor = self.transform(image)
        if self.vis_transform:
            image_vis = self.vis_transform(image)
        return image_tensor, image_vis



def overlay_img(image1, image2, alpha, output_path='overlayed_img.png'):
    image1 = np.array(image1)
    image2 = np.array(image2)
    image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2BGR)
    image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2BGR)

    overlay = cv2.addWeighted(image2, 1- alpha, image1, alpha, 0)
    cv2.imwrite(output_path, overlay)
    return overlay


def generate_colormap(N: int, seed: int = 0) -> List[Tuple[float, float, float]]:
    """Generates a equidistant colormap with N elements."""
    random.seed(seed)

    def generate_color():
        return (random.random(), random.random(), random.random())

    return [generate_color() for _ in range(N)]


def make_visualization(
    ori_img: Image, source: torch.Tensor, patch_size: int = 16, class_token: bool = True, output_path='overlayed_img.png'
) -> Image:
    """
    Create a visualization like in the paper.

    Args:
     -

    Returns:
     - A PIL image the same size as the input.
    """

    img = np.array(ori_img.convert("RGB")) / 255.0
    source = source.detach().cpu()

    h, w, _ = img.shape
    ph = h // patch_size
    pw = w // patch_size

    if class_token:
        source = source[:, :, 1:]

    vis = source.argmax(dim=1)
    num_groups = vis.max().item() + 1

    cmap = generate_colormap(num_groups)
    vis_img = 0
    print(num_groups)

    for i in range(num_groups):
        mask = (vis == i).float().view(1, 1, ph, pw)
        mask = F.interpolate(mask, size=(h, w), mode="nearest")
        mask = mask.view(h, w, 1).numpy()

        color = (mask * img).sum(axis=(0, 1)) / mask.sum()
        mask_eroded = binary_erosion(mask[..., 0])[..., None]
        mask_edge = mask - mask_eroded

        if not np.isfinite(color).all():
            color = np.zeros(3)

        vis_img = vis_img + mask_eroded * color.reshape(1, 1, 3)
        vis_img = vis_img + mask_edge * np.array(cmap[i]).reshape(1, 1, 3)

    # Convert back into a PIL image
    vis_img = Image.fromarray(np.uint8(vis_img * 255))
    return overlay_img(ori_img, vis_img, 0.25, output_path=output_path)



if __name__ == '__main__':
    import algm
    import argparse
 
    parser = argparse.ArgumentParser(description='Visualize segmentation model output.')
    parser.add_argument('--model_size', type=str, default='ti', choices=['ti', 's', 'b', 'l'], help='Model size to use.')
    parser.add_argument('--algo', type=str, default='pitome', help='algo to use')
    args = parser.parse_args()

    model_size = args.model_size

    model_dir = '/media/caduser/MyBook/chau/algm-segmenter/runs'
    model_path =  f'{model_dir}/vit_{model_size}_16/checkpoint.pth'
    root_dir = os.getenv('DATASET')

    model, variant = load_model(model_path)
    root_dir = os.getenv('DATASET')
    input_size = variant['dataset_kwargs']['crop_size']
    normalization = variant['dataset_kwargs']['normalization']

    dataset = 'ade20k'
    dataset_path, dataset_txt_path =  get_dataset_validation_path(dataset, root_dir)
    stats = STATS[normalization]
    batch_size = 1 
    validation_loader = dataset_prepare(dataset_path, dataset_txt_path, stats, batch_size, input_size)

    vis_transforms = transforms.Compose([
        transforms.Resize((input_size, input_size)),  # Resize the image to the input size
    ])

    tensor_transforms = transforms.Compose([
        transforms.Resize((input_size, input_size)),  # Resize the image to the input size
        transforms.ToTensor(),          # Convert image to tensor
        transforms.Normalize(stats["mean"], stats["std"]) # Normalize with mean and std
    ])

    dataset = InferenceDataset(root_dir=dataset_path, transform=tensor_transforms, vis_transform=vis_transforms)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # index = 1000 
    #    index = 1500 
    #    index = 950 
    index = 1350 
    #    index = 1950 
    if args.algo == 'algm':
        algm.patch.algm_segmenter_patch(model, [1,5], trace_source=True)
        model.encoder.window_size = [2,2] 
        model.encoder.threshold = 0.88     
    elif args.algo == 'pitome':
       algm.patch.pitome_segmenter_patch(model, [1,2], trace_source=True, num_merge_step=2)
       model.encoder.margin = 0.925
    


    sample_idx = index 
    image_tensor, image_vis = dataset[sample_idx]
    sample = image_tensor.repeat(1,1,1,1).to(device)
    model = model.to(device)
    model(sample)
    source = model.encoder._turbo_info['source']

    make_visualization(image_vis, source, output_path=f'{args.algo}.png')
    
