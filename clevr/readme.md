This repository contains code related to clevr experiments in the paper 
[Bias and Generalization in Deep Generative Models](https://arxiv.org/abs/1811.03259). 

![concept](https://github.com/ermongroup/BiasAndGeneralization/blob/master/DotsAndPie/img/concept_illustration.png)

For code and instructions on the dataset, please refer to the 
[dataset](https://github.com/hyren/clevr/tree/master/clevr) folder. 

## GAN
### Numerosity
```
python num.py --model_name WGAN-GP --base_name CNN \
    --input_height=128 --output_height=128 --train \
    --dataset one --batch_size 64 \
    --data_str=../dataset/clevr-dataset-gen/dataset/ --d_iter=5 --g_iter=1
```
### Shape-Color
```
python sc.py --model_name WGAN-GP --base_name CNN \
    --input_height=128 --output_height=32 --train \
    --dataset red_cone.blue_cone --batch_size 64 \
    --cors color --universal red.blue-cone.torus \
    --data_str=../dataset/clevr-dataset-gen/dataset/ --d_iter=5 --g_iter=1
```
```
python sc.py --model_name WGAN-GP --base_name CNN \
    --input_height=128 --output_height=32 --train \
    --dataset red_cone.blue_cylinder-blue_cone.green_cylinder --batch_size 64 \
    --cors shape --universal red.green.blue-cone.cylinder.sphere \
    --data_str=../dataset/clevr-dataset-gen/dataset/ --d_iter=5 --g_iter=1
```
## PixelCNN
### Shape-Color
```
python train.py --nr_gpu 1 --dataset red_cone.blue_cone --cors color --universal red.blue-cone.torus
```

## Important Arguments
- ``--universal`` denotes the universal set of shape and color, we assume all images contain two objects. For example, ``red.green.blue-cone.cylinder.sphere`` means the two object may have color in ``[red, green, blue]`` and shape in ``[cone, cylinder, sphere]``.
- ``--cors``, i.e. "color or shape", do experiment with color or shape. If choose shape, it means that a given shape (specified by argument ``--dataset``) will only appear on some colors while other shapes will appear on all possible colors.
- ``--dataset`` denotes the exceptions. For example, ``--cors=shape --dataset=red_cone.blue_cylinder-blue_cone.green_cylinder`` means that one certain shape configuration, which is the left object is a ``cone`` and the right object is a ``cylinder``, does not appear on any color configurations except when the left object is ``red`` and the right object is ``blue`` or the left object is ``blue`` and the right object is ``green``, please refer to the figure below as the final configurations in the training set. (so another implicit requirement is that if ``--cors=shape``, the ``--dataset`` argument may only contain one shape configuration, for example, ``--dataset=red_cone.blue_cylinder-blue_sphere.green_cylinder`` does not meet the requirement, and may cause bugs).

<img src="https://github.com/ermongroup/BiasAndGeneralization/blob/master/clevr/clevr/samples/nine.png" width="200" height="200" />

- ``--data_str`` denotes the directory where the trained images are saved, the organization should look like the following. The [dataset generation code](https://github.com/hyren/clevr/tree/master/clevr) will organize the images in the exact way.
```
$data_str/
$data_str/red_cone/
$data_str/red_cylinder/
...
```

### Disclaimers
Some code is borrowed from the GAN [repo](https://github.com/carpedm20/DCGAN-tensorflow) and PixelCNN [repo](https://github.com/openai/pixel-cnn). 

