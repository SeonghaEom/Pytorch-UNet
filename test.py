import argparse
import logging
import os
from os.path import splitext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask
from evaluate import evaluate
from utils.dice_score import multiclass_dice_coeff, dice_coeff

def preprocess(pil_img, scale, is_mask):
    w, h = pil_img.size
    newW, newH = int(scale * w), int(scale * h)
    assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
    pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
    img_ndarray = np.asarray(pil_img)

    if img_ndarray.ndim == 2 and not is_mask:
        img_ndarray = img_ndarray[np.newaxis, ...]
    elif not is_mask:
        img_ndarray = img_ndarray.transpose((2, 0, 1))

    if not is_mask:
        img_ndarray = img_ndarray / 255

    return img_ndarray

def load(filename):
    ext = splitext(filename)[1]
    if ext in ['.npz', '.npy']:
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)


def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor, is_mask=False))[:3,:,:]
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)[0]
            output = F.one_hot(output.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
        else:
            probs = torch.sigmoid(output)[0]

        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((full_img.size[1], full_img.size[0])),
            transforms.ToTensor()
        ])
        full_mask = tf(probs.cpu()).squeeze()

    if net.n_classes == 1:
        return (full_mask > out_threshold).numpy(), output
    else:
        return F.one_hot(full_mask.argmax(dim=0), net.n_classes).permute(2, 0, 1).numpy(), output


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Directory of input images (.png files)', required=True)
    parser.add_argument('--label', '-l', metavar='INPUT', nargs='+', help='Directory of label images (.npy files)', required=True)
    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+', help='Directory of output images that will be produced (.png, .npy files)', required=True)
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')

    return parser.parse_args()

def get_output_filenames(in_files):
    # def _generate_name(fn):
    #     split = os.path.splitext(fn)
    #     return f'{split[0]}_OUT{split[1]}'

    return list(map('./result/', in_files))


def mask_to_image(mask: np.ndarray):
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))


if __name__ == '__main__':
    args = get_args()
    in_files = list(Path(args.input[0]).glob('*.png'))
    label_files = Path(args.label[0])
    out_path = Path(args.output[0])

    net = UNet(n_channels=3, n_classes=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Loading model {args.model}')
    print(f'Using device {device}')


    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))
    net.eval()

    
    dice_score = 0

    # logging.info('Model loaded!')
    print ('Model loaded!')

    test_set = BasicDataset(args.input[0], args.label[0], 0.5)
    loader_args = dict(batch_size=1, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, shuffle=False, drop_last=True, **loader_args)
    num_test_batches = len(test_loader)

    for i, batch in enumerate(test_loader):
        image, mask_true = batch['image'][:,:3], batch['mask']
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # predict the mask
            mask_pred = net(image)

            # convert to one-hot format
            if net.n_classes == 1:
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=False)

    net.train()

    # Fixes a potential division by zero error
    if num_test_batches == 0:
        print('Dice score: {}'.format(dice_score))
        print('Jaccard Index: {}'.format(dice_score /(2 - dice_score)))
    else:
        dice_score = dice_score / num_test_batches
        print('Dice score: {}'.format(dice_score))
        print('Jaccard Index: {}'.format(dice_score /(2 - dice_score)))

    for i, filename in enumerate(in_files):
        print (f'\nPredicting image {filename} ...')
        
        img = Image.open(filename)
        mask_pred, mask_pred_tensor = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)

        # label = Image.fromarray(np.load(label_files[i]))

        if not args.no_save:
            # print (out_path)
            
            result = mask_to_image(mask_pred)
            basename = os.path.basename(filename)
            split= os.path.splitext(basename)
            # print (basename)
            result.save('{}/{}'.format(out_path,basename))

            # mask_np = mask_pred.cpu()
            np.save('{}/{}'.format(out_path,split[0]), mask_pred)
            print ('Saved segmentation results ({}.png,{}.npy) in result directory..'.format(split[0], split[0]))