# Only works with python 2
import matplotlib
matplotlib.use('Agg')
import time
import os
import numpy as np
import argparse
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--bn', type=int, default=0)
parser.add_argument('--bs', type=int, default=128)
parser.add_argument('--dest', type=str, default='/data/dots/')
parser.add_argument('--noisy', type=bool, default='True')
parser.add_argument('--count', type=int, default=3, help='number of dots to generate')
args = parser.parse_args()

args.dest = os.path.join(args.dest, '%d_dots' % args.count)


def fig2data(fig):
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf


def gen_image_count(num_object=3, overlap=False):
    radius = 0.08
    while True:
        shifts = np.random.uniform(radius, 1.0-radius, size=(num_object, 2))
        dist1 = np.tile(np.expand_dims(shifts, axis=0), (num_object, 1, 1))
        dist2 = np.tile(np.expand_dims(shifts, axis=1), (1, num_object, 1))
        dist = np.sqrt(np.sum(np.square(dist1 - dist2), axis=2))
        np.fill_diagonal(dist, 1.0)
        if not overlap and np.min(dist) > 2.1 * radius:
            break
        if overlap and np.min(dist) > 2 * radius * 0.9:
            break

    margin = 5
    fig = plt.figure(figsize=((64+2*margin)/10.0, (64+2*margin)/10.0), dpi=10)
    ax = plt.gca()
    for i in range(num_object):
        random_color = np.random.uniform(0, 0.9, size=(3,))
        circle = plt.Circle(shifts[i], radius, color=random_color)
        ax.add_artist(circle)
    plt.axis('off')
    plt.tight_layout()

    arr = fig2data(fig)
    arr = arr[margin:64+margin, margin:64+margin, :3]

    plt.close(fig)
    return arr


images = []
start_time = time.time()
for i in range(args.bs):
    new_img = gen_image_count(args.count).astype(np.float32) / 255.0

    if args.noisy:
        new_img += np.random.normal(loc=0, scale=0.03, size=new_img.shape)
        new_img = 1.0 - np.abs(1.0 - new_img)
        new_img = np.abs(new_img)

    images.append(new_img)
    if (i+1) % 1000 == 0:
        print("Generating %d-th image, time used: %f" % (i+1, time.time() - start_time))

plt.imshow(images[0])
plt.show()

batch_img = np.stack(images, axis=0)
if not os.path.isdir(args.dest):
    os.makedirs(args.dest)
np.savez(os.path.join(args.dest, 'batch%d.npz' % args.bn), images=batch_img)

