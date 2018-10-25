# Generating Samples in CLEVR Dataset

Souce code to generate the CLEVR dataset in our NIPS'18 paper ["Bias and Generalization in Deep Generative Models: An Empirical Study"](/).

If you find it helpful, please consider citing our paper.

**Disclaimers**: some code is borrowed from the original CLEVR Generation [repo](https://github.com/facebookresearch/clevr-dataset-gen).

## Steps

1. Download zip file on [blender website](https://www.blender.org/download/) and unzip it to any path you want, say $BLENDER.

2. git clone this repo into any path you want.

3. ```cd clevr```

4. ```export PATH=$BLENDER:$PATH```

5. ```echo $PWD/image_generation >> $BLENDER/$VERSION/python/lib/python3.5/site-packages/clevr.pth```

6. ```cd image_generation```

7. Synthetic the dataset based on your need, note that the generation process can be done in parallel.

## Numerosity
### Generate One Object
```
blender --background --python render_images.py -- --num_images 10 --min_dist 2.5 \
  --margin 2.5 --min_pixels_per_object 700 --max_retries 5 --min_objects 1 --max_objects 1 --width 128 \
  --height 128 --start_idx 0 --use_gpu 1 --properties_json data/numerosity.json \
  --output_image_dir ../dataset/one/images/ --output_scene_dir ../dataset/one/scenes \
  --output_scene_file ../dataset/one/CLEVR_scenes.json
```

Some samples

![one1](https://github.com/hyren/clevr/blob/master/clevr/samples/one1.jpg)
![one2](https://github.com/hyren/clevr/blob/master/clevr/samples/one2.jpg)
![one3](https://github.com/hyren/clevr/blob/master/clevr/samples/one3.jpg)

### Generate Two Objects
```
blender --background --python render_images.py -- --num_images 10 --min_dist 2.5 \
  --margin 2.5 --min_pixels_per_object 700 --max_retries 5 --min_objects 2 --max_objects 2 --width 128 \
  --height 128 --start_idx 0 --use_gpu 1 --properties_json data/numerosity.json \
  --output_image_dir ../dataset/two/images/ --output_scene_dir ../dataset/two/scenes \
  --output_scene_file ../dataset/two/CLEVR_scenes.json
```

Some samples

![two1](https://github.com/hyren/clevr/blob/master/clevr/samples/two1.jpg)
![two2](https://github.com/hyren/clevr/blob/master/clevr/samples/two2.jpg)
![two3](https://github.com/hyren/clevr/blob/master/clevr/samples/two3.jpg)

### Generate Three Objects
```
blender --background --python render_images.py -- --num_images 10 --min_dist 0.4 \
  --min_pixels_per_object 700 --max_retries 5 --min_objects 3 --max_objects 3 --width 128 --height 128 \
  --start_idx 0 --use_gpu 1 --properties_json data/numerosity.json \
  --output_image_dir ../dataset/three/images/ --output_scene_dir ../dataset/three/scenes\
  --output_scene_file ../dataset/three/CLEVR_scenes.json
```

Some samples

![three1](https://github.com/hyren/clevr/blob/master/clevr/samples/three1.jpg)
![three2](https://github.com/hyren/clevr/blob/master/clevr/samples/three2.jpg)
![three3](https://github.com/hyren/clevr/blob/master/clevr/samples/three3.jpg)

## Combinations
### Generate Red Cone and Green Cylinder
```
blender --background --python generate_combinations.py -- --num_images 10 --min_dist 0.5 --margin 0.5 \
  --min_pixels_per_object 700 --max_retries 5 --min_objects 2 --max_objects 2 --width 128 --height 128 \
  --start_idx 0 --use_gpu 1 --gen_list red_cone.green_cylinder
```

Some samples

![comb1](https://github.com/hyren/clevr/blob/master/clevr/samples/comb1.jpg)
![comb2](https://github.com/hyren/clevr/blob/master/clevr/samples/comb2.jpg)
![comb3](https://github.com/hyren/clevr/blob/master/clevr/samples/comb3.jpg)

All the generated images will be in the `../dataset` directory. 

