import torch

def spatial_tv_norm(x):
    """
    x: tensor with dimensions (t,c,x,y)
    """
    x_diff = x - torch.roll(x, -1, dims=3)
    y_diff = x - torch.roll(x, -1, dims=2)
    grad_norm2 = torch.pow(x_diff, 2) + torch.pow(y_diff, 2)
    norm = torch.sum(grad_norm2)
    t_img, c_img, h_img, w_img = x.size()
    return norm / (h_img * w_img)


def temporal_tv_norm(x):
    """
    x: tensor with dimensions (t,c,x,y)
    """
    t_diff = x - torch.roll(x, -1, dims=0)
    grad_norm2 = torch.pow(t_diff, 2)
    norm = torch.sum(grad_norm2)
    t_img, c_img, h_img, w_img = x.size()
    return norm / t_img


