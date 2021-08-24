# import cv2
import torchvision.transforms as T

def save_tensor_list_as_gif(tensors, path='out.gif', duration=200, loop=0):
    """
    @param tensor: list of C,H,W tensors
    @param path: where to save the gif
    @param duration: duration of each frame in ms
    @param loop: 0 if it should loop forever
    """
    PIL_img_list = []

    for t in tensors:
        PIL_img_list.append(T.ToPILImage()(t))

    PIL_img_list[0].save(path, append_images=PIL_img_list[1:],save_all=True, duration=duration, loop=loop, allow_mixed=False)

def save_tensor_as_img(tensor, path='out.png'):
    """
    @param tensor: C,H,W tensor
    """
    img = T.ToPILImage()(tensor)
    img.save(path)

def save_tensor_list_as_avi(tensors, path='out.mp4', fps=10):
    """
    @param tensor: list of C,H,W tensors
    @param path: where to save the gif
    @param fps: frame rate to save at
    """
    ch, h, w = tensors[0].shape

    videodims = (h, w)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(f"{path}",fourcc, fps, videodims)

    for t in tensors:
        video.write(cv2.cvtColor(t.permute(1,2,0).numpy(), cv2.COLOR_RGB2BGR))
    video.release()