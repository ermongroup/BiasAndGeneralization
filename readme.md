This repository contains code related to numerosity and pie experiments in the paper 
["Bias and Generalization in Deep Generative Models"](to appear). 

For code and instructions on the dataset, please refer to the 
[dataset](https://github.com/ShengjiaZhao/BiasAndGeneralization/tree/master/dataset) folder. 

To train the model, use ``` python train.py [options] ```

Possible arguments are:

- ``--objective=str`` where ``str`` can be ``vae`` or ``gan``.
- ``--architecture=str`` where ``str`` can be ``conv``, ``small`` of ``fc``. ``conv`` and ``small`` use a convnet, while ``fc`` uses a fully connected network. ``small`` contains approximately 1/4 as many parameters as ``conv``. 
- ``--dataset=str`` where 
- ``--gpu=int`` specifies which GPU to launch the job.
- ``--z_dim=int`` is the dimensionality of latent variables ``z``. 
