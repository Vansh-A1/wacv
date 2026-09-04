import os, sys, cv2
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image, ImageOps, ImageFilter
import numpy as np
import random
try:
    from natsort import os_sorted
except ImportError:  # optional; only DistTestImageDataset needs it
    def os_sorted(iterable):
        return sorted(iterable)

def get_training_set(root_dir):
    return ImageDataset(root_dir,
                             transform=transform())

def crop_image(img, crop_size):
    # Define the desired crop size
    crop_size = (crop_size, crop_size)

    # Get the image size
    width, height = img.size

    # Calculate the maximum x and y coordinates for the top-left corner of the crop
    max_x = width - crop_size[0]
    max_y = height - crop_size[1]

    # Generate a random x and y coordinate for the top-left corner of the crop
    x = random.randint(0, max_x)
    y = random.randint(0, max_y)

    # Crop the image
    img_cropped = img.crop((x, y, x+crop_size[0], y+crop_size[1]))

    return img_cropped

def center_crop(img, size):
    """
    Crop the given PIL Image at the center to the given size.
    Args:
        img (PIL.Image): Image to be cropped.
        size (tuple): Desired output size of the crop.
    Returns:
        PIL.Image: Cropped image.
    """
    width, height = img.size  # Get the dimensions of the image
    # print(img.size)
    left = (width - size) / 2
    top = (height - size) / 2
    right = (width + size) / 2
    bottom = (height + size) / 2
    # Crop the image at the center
    return img.crop((left, top, right, bottom))

def add_noise(img):
  
    # Getting the dimensions of the image
    # print(img.shape)
    row , col, ch = img.shape
      
    # Randomly pick some pixels in the
    # image for coloring them white
    # Pick a random number between 300 and 10000
    number_of_pixels = random.randint(300, 10000)
    for i in range(number_of_pixels):
        
        # Pick a random y coordinate
        y_coord=random.randint(0, row - 1)
          
        # Pick a random x coordinate
        x_coord=random.randint(0, col - 1)
          
        # Color that pixel to white
        img[y_coord][x_coord] = 255
          
    # Randomly pick some pixels in
    # the image for coloring them black
    # Pick a random number between 300 and 10000
    number_of_pixels = random.randint(300 , 10000)
    for i in range(number_of_pixels):
        
        # Pick a random y coordinate
        y_coord=random.randint(0, row - 1)
          
        # Pick a random x coordinate
        x_coord=random.randint(0, col - 1)
          
        # Color that pixel to black
        img[y_coord][x_coord] = 0
          
    return img

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, train_size, crop=None, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.crop_size = crop
        self.images = []
        self.image_count = train_size
        # self.count = 0

        """***********************************************************************"""
        # This part of code is to make list of image for REDS dataset
        for folder in sorted(os.listdir(self.root_dir)):
            self.count = 0
            for file in sorted(os.listdir(os.path.join(self.root_dir, folder))):                
                if file.endswith('.jpg') or file.endswith('.png') or file.endswith('.bmp'):
                    self.images.append(os.path.join(self.root_dir,folder, file))
                    self.count += 1
                    # print(self.count, os.path.join(self.root_dir,folder, file))
                    if self.count >= self.image_count:
                        break
        """***********************************************************************""" 
        # sys.exit()

        # for file in sorted(os.listdir(self.root_dir)):
        #     if file.endswith('.jpg') or file.endswith('.png') or file.endswith('.bmp'):
        #         self.images.append(os.path.join(self.root_dir, file))
        #         self.count += 1
        #         if self.count >= self.image_count:
        #             break

        # if self.image_count <=  len(self.images):
        #     self.images = self.images[0:self.image_count]

        print('===>Training data size :',len(self.images))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        
        if self.crop_size is not None:
            # image = center_crop(image, self.crop_size)
            image = image.resize((self.crop_size, self.crop_size))
        

        # Normalize to the range 0-1
        image = np.asarray(image)
        image = (image/255.0).astype('float32')

        if self.transform:
            image = self.transform(image)

        
        return image

class ValidationImageDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, val_size, crop=None, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.crop_size = crop
        self.images = []
        self.image_count = val_size
        self.count = 0

        for folder in sorted(os.listdir(self.root_dir)):
            self.count = 0
            for file in sorted(os.listdir(os.path.join(self.root_dir, folder))):                
                if file.endswith('.jpg') or file.endswith('.png') or file.endswith('.bmp'):
                    self.images.append(os.path.join(self.root_dir,folder, file))
                    self.count += 1
                    # print(self.count, os.path.join(self.root_dir,folder, file))
                    if self.count >= self.image_count:
                        break


        # for file in sorted(os.listdir(self.root_dir)):
        #     if file.endswith('.jpg') or file.endswith('.png') or file.endswith('.bmp'):
        #         self.images.append(os.path.join(self.root_dir, file))
        #         self.count += 1
        #         if self.count >= self.image_count:
        #             break

        # if self.image_count <=  len(self.images):
        #     self.images = self.images[0:self.image_count]

        print('===>Validation data size :',len(self.images))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        
        if self.crop_size is not None:
            # image = center_crop(image, self.crop_size)  
            image = image.resize((self.crop_size, self.crop_size))      

        # Normalize to the range 0-1
        image = np.asarray(image)
        image = (image/255.0).astype('float32')

        if self.transform:
            image = self.transform(image)

        
        return image

class TestImageDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, test_size, crop=None, scale= 1, noise=False, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.crop_size = crop
        self.scale = scale
        self.images = []
        self.image_count = test_size
        self.noise = noise

        for folder in sorted(os.listdir(self.root_dir)):
            self.count = 0
            for file in sorted(os.listdir(os.path.join(self.root_dir, folder))):                
                if file.endswith('.jpg') or file.endswith('.png') or file.endswith('.bmp'):
                    self.images.append(os.path.join(self.root_dir,folder, file))
                    self.count += 1
                    # print(self.count, os.path.join(self.root_dir,folder, file))
                    if self.count >= self.image_count:
                        break


        # for file in os_sorted(os.listdir(self.root_dir)):
        #     if file.endswith('.jpg') or file.endswith('.png') or file.endswith('.bmp'):
        #         self.images.append(os.path.join(self.root_dir, file))
                # self.count += 1
                # if self.count >= self.image_count:
                #     break
        # if self.image_count <=  len(self.images):
        #     self.images = self.images[0:self.image_count]
        
        print('===>Testing data size :',len(self.images))
        if self.scale > 1:
            print('===> Scaling input image to ', self.scale, 'x')
        if self.noise:
            print('===> Adding random noise to image...')



    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')

        width, height = image.size

        # print(os.path.split(self.images[idx])[1])
        basename = os.path.basename(os.path.split(self.images[idx])[1])

        # if width < self.crop_size or height < self.crop_size:
        #     image = image.resize((self.crop_size, self.crop_size))
        if self.crop_size is not None:
            image = center_crop(image, self.crop_size)   
        
        
        if self.scale > 1:
            # width, height = image.size
            downsampled_im = image.resize((image.width//self.scale, image.height//self.scale))
            # downsampled_im.save('downn.png')
            upsampled_im = downsampled_im.resize((image.width, image.height))
            # upsampled_im.save('upsample.png')
            image = downsampled_im


        
        # print(image.size)

        if self.noise:
            # Apply Gaussian blur filter with radius 2
            image = image.filter(ImageFilter.GaussianBlur(radius=8)) 
            # image.save('blur.png')  

        image = np.asarray(image)


        # Normalize to the range 0-1
        image = (image/255.0).astype('float32')

        # image.save('norm.png')
        
        if self.transform:
            image = self.transform(image)

        # transform_val = transforms.ToPILImage()
        # tf_imgae = transform_val(image)
        # tf_imgae.save('denorm.png')

        
        # if self.noise:
        #     noise = torch.randn_like(image) * 1 + 0
        #     image = image + noise
        
        return image, basename

class DistTestImageDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, test_size, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.image_count = test_size


        for file in os_sorted(os.listdir(self.root_dir)):
            if file.endswith('.jpg') or file.endswith('.png') or file.endswith('.bmp'):
                self.images.append(os.path.join(self.root_dir, file))

        if self.image_count <=  len(self.images):
            self.images = self.images[0:self.image_count]
        
        print('===>Testing data size :',len(self.images))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        
        if self.transform:
            image = self.transform(image)

    
        return image


def _resize_short_side(img, target):
    """If any side < target, scale short side up to target (preserve aspect)."""
    w, h = img.size
    if w >= target and h >= target:
        return img
    scale = float(target) / float(min(w, h))
    new_w = max(target, int(round(w * scale)))
    new_h = max(target, int(round(h * scale)))
    return img.resize((new_w, new_h), Image.BICUBIC)


def _label_from_basename(basename):
    name = basename.lower()
    if '_hr' in name or name.endswith('hr.png') or name.endswith('hr.bmp') or name.endswith('hr.jpg'):
        return 'HR'
    if '_lr' in name or 'x8_lr' in name or name.endswith('lr.png') or name.endswith('lr.bmp'):
        return 'LR'
    if 'gblur' in name or ('blur' in name and 'sharpen' not in name):
        return 'gblur'
    if 'jpeg' in name or 'jpg_q' in name:
        return 'jpeg'
    if 'sharpen' in name:
        return 'sharpen'
    if 'color_f' in name or '_color_' in name:
        return 'color'
    if 'pixelate' in name:
        return 'pixelate'
    return 'other'


class ManifestImageDataset(torch.utils.data.Dataset):
    """Training dataset from an explicit absolute-path file list (manifest).
    Random-crop to crop_size, random hflip, normalize to [0,1].
    If an image is smaller than crop_size on any side, resize short side up then crop.
    """
    def __init__(self, file_list, crop_size=256, hflip=True, transform=None):
        self.images = list(file_list)
        self.crop_size = crop_size
        self.hflip = hflip
        self.transform = transform
        print('===>Training data size :', len(self.images))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        image = _resize_short_side(image, self.crop_size)
        image = crop_image(image, self.crop_size)
        if self.hflip and random.random() < 0.5:
            image = ImageOps.mirror(image)

        image = np.asarray(image)
        image = (image / 255.0).astype('float32')

        if self.transform:
            image = self.transform(image)

        return image


class EvalImageDataset(torch.utils.data.Dataset):
    """Fixed eval/plot set: center-crop (or resize-short then center-crop), [0,1],
    returns (image, basename, label) labeled by filename (HR/LR/gblur/jpeg).
    """
    def __init__(self, file_list, crop_size=256, transform=None):
        self.images = list(file_list)
        self.crop_size = crop_size
        self.transform = transform
        print('===>Eval data size :', len(self.images))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path = self.images[idx]
        basename = os.path.basename(path)
        label = _label_from_basename(basename)

        image = Image.open(path).convert('RGB')
        image = _resize_short_side(image, self.crop_size)
        image = center_crop(image, self.crop_size)

        image = np.asarray(image)
        image = (image / 255.0).astype('float32')

        if self.transform:
            image = self.transform(image)

        return image, basename, label
