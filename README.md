

# Minimal working(?) Activation Maximization for individual layers, and output neurons

Set up the environment:
```conda create --name <env> --file environment.yaml```

To do a dry run `python do_activation_maximization.py`.


### Flownet
* FlowNet partially from https://github.com/ClementPinard/FlowNetPytorch
* Needs: https://pypi.org/project/spatial-correlation-sampler/ (gcc8 doesnt work, needs ```$CUDA_HOME``` to be set)
* Get weights from https://drive.google.com/file/d/1H_5WE-Lrx5arD0-X801yRzdSAuBZQmXh/view


### C3D
* C3D from https://github.com/DavideA/c3d-pytorch.
* Get weights from http://imagelab.ing.unimore.it/files/c3d_pytorch/c3d.pickle