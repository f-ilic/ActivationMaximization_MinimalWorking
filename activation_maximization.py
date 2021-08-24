from torch.optim import SGD, Adam, AdamW
from util.metrics import spatial_tv_norm, temporal_tv_norm
import copy
import numpy as np
from PIL import Image, ImageFilter

import torch
from torch.autograd import Variable
def init_input(num_frames, init_blur=0.1, mean=[0.5, 0.5, 0.5], std=[0,0,0], img_height=160, img_width=160):
    created_images = []
    for i in range(0, num_frames):
        np.random.seed(1337)
        x = np.uint8(np.random.uniform(0, 255, (img_height, img_width, 3)))
        created_images.append(x)

    created_images = [recreate_image(blur_image(ci, init_blur, mean, std).cpu()) for ci in created_images]
    return created_images

def init_flownet_input(init_blur=0.1, mean=[0.5, 0.5, 0.5], std=[0,0,0], img_height=160, img_width=160):
    created_images = []
    for i in range(0, 2):
        np.random.seed(1337)
        x = np.uint8(np.random.uniform(0, 255, (img_height, img_width, 3)))
        created_images.append(x)

    created_images = [recreate_image(blur_image(ci, init_blur, mean, std).cpu()) for ci in created_images]
    return created_images

class SaveFeatures():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        # self.features = torch.tensor(output,requires_grad=True).cuda()
        self.features = output
    def close(self):
        self.hook.remove()

def maximize_activation(model, target_class, init, iterations=550, blur_rad=1, wd=0.00001,
                        lr=10, kappa=0, gamma=0, mean=[0.5, 0.5, 0.5], std=[0,0,0], featuremap=0, layer=0):
    """
    This function maximized the target class (last fc neuron), or a single layer given a specific model.
    @param init is the initial conditions of the input
    note: init is a list of tensors, which are in this loop transformed to the correct tensor shape (if its video data)
          this means that if you're doing activation maximization for a model that only uses a single frame as input
          still pass it in as a list that contains the singular tensor. This is handled correctly.

          Also image is blurred ever iteration with a gaussian of sigma=@param blur_rad
          Spatial and Temporal TV available so the loss is computed as

        loss = class_loss + (TV_spatial * kappa) + (TV_temporal * gamma)

        returns list of tensors [(img), (img) ...] img: w,h,ch
    """
    model = model.cuda()
    model.eval()
    clipping_value = 0.1
    num_frames = len(init)
    created_images = init
    activations = SaveFeatures(list(model.children())[layer])  # register hook

    # ---------------------------------- Iterations in Maximization Activation ----------------------------------
    for i in range(0, iterations):
        processed_images = [blur_image(ci, blur_rad, mean, std) for ci in created_images]
        processed_images = [pi.cuda() for pi in processed_images]

        # CHOSE WHICH OPTIMIZER YOU WANT TO USE HERE!
        # optimizer = SGD(processed_images, lr=lr, weight_decay=wd)
        optimizer = AdamW(processed_images, lr=lr, weight_decay=wd)

        # Reshape the input variable to have correct dimensions, i.e. if its a temporal stack or a single frame
        if len(processed_images) == 1:
            input_var = torch.cat(processed_images, 0).permute(1, 0, 2, 3).squeeze().unsqueeze(0)
        else:
            input_var = torch.cat(processed_images, 0).permute(1, 0, 2, 3).unsqueeze(0)

        output = model(input_var)

        # v--- Use this if you want to do activation  maximization on the classification neuron
        # class_loss = -output[0, target_class]

        # v--- Use this for AM on individual layers
        activation_loss = -(activations.features[0, featuremap].mean())

        tensor_stack = torch.cat(processed_images, dim=0)
        spat_tv_loss = spatial_tv_norm(tensor_stack)
        temp_tv_loss = temporal_tv_norm(tensor_stack)

        spat_tv_loss *= kappa
        temp_tv_loss *= gamma

        loss = activation_loss + spat_tv_loss + temp_tv_loss

        if i % 10 == 0:
            print(f'it:{str(i)} \t | loss:{loss:.2f} \t | '
                  f'activation loss:{activation_loss:.2f} \t | '
                  f'spatial tv:{spat_tv_loss:.8f} \t | '
                  f'temporal tv:{temp_tv_loss:.8f}')

        model.zero_grad()
        loss.backward()

        if clipping_value:
            torch.nn.utils.clip_grad_norm(model.parameters(), clipping_value)
        optimizer.step()

        # Recreate image: [(img), (img) ...] img: w,h,ch
        created_images = [recreate_image(processed_images[i].cpu(), mean, std) for i in range(0, num_frames)]

    return created_images


def preprocess_image(pil_im, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Args:
        PIL_img (PIL_img): PIL Image or numpy array to process
        resize_im (bool): Resize to 224 or not
    returns:
        im_as_var (torch variable): Variable that contains processed float tensor
    """

    if type(pil_im) != Image.Image:
        try:
            pil_im = Image.fromarray(pil_im)
        except Exception as e:
            print("could not transform PIL_img to a PIL Image object. Please check input.")

    im_as_arr = np.float32(pil_im)
    im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
    # Normalize the channels
    # z = (x-mean)/sigma
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
        im_as_arr[channel] -= mean[channel]
        im_as_arr[channel] /= std[channel]
    # Convert to float tensor
    im_as_ten = torch.from_numpy(im_as_arr).float()
    # Add one more channel to the beginning. Tensor shape = 1,3,224,224
    im_as_ten.unsqueeze_(0)
    # Convert to Pytorch variable
    im_as_var = Variable(im_as_ten, requires_grad=True)
    return im_as_var # 1, 3, height, width


def recreate_image(im_as_var, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
        create numpy image from torch tensor: output dim = h,w,ch
    """
    reverse_mean = [-m/s for m, s in zip(mean, std)]
    reverse_std = [1/s for s in std]

    recreated_im = copy.copy(im_as_var.data.numpy()[0])
    for c in range(3):
        recreated_im[c] -= reverse_mean[c]
        recreated_im[c] /= reverse_std[c]
    recreated_im[recreated_im > 1] = 1
    recreated_im[recreated_im < 0] = 0
    recreated_im = np.round(recreated_im * 255)

    recreated_im = np.uint8(recreated_im).transpose(1, 2, 0)
    return recreated_im


def blur_image(pil_im, blur_rad=None, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    if type(pil_im) != Image.Image:
        try:
            pil_im = Image.fromarray(pil_im)
        except Exception as e:
            print("could not transform PIL_img to a PIL Image object. Please check input.")

    if blur_rad:
        pil_im = pil_im.filter(ImageFilter.GaussianBlur(blur_rad))

    im_as_arr = np.float32(pil_im)
    im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H

    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
        im_as_arr[channel] -= mean[channel]
        im_as_arr[channel] /= std[channel]
    im_as_ten = torch.from_numpy(im_as_arr).float()
    # Add one more channel to the beginning. Tensor shape = 1,3,224,224
    im_as_ten.unsqueeze_(0)
    im_as_var = Variable(im_as_ten.cuda(), requires_grad=True)
    return im_as_var  # 1,3, height, width

