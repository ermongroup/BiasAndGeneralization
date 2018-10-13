import math
if __name__ == '__main__':
    from abstract_dataset import *
else:
    from dataset.abstract_dataset import *


def gen_pie(params):
    count = int(params[0])

    # 0.35 + k/20: 0.4-0.8, range 1-9
    size = int(params[1])
    if size == 0:
        size = np.random.randint(1, 10)
    size = size / 20.0 + 0.35
    print(size)

    # (k-5)/20: -0.2 - 0.2
    locx = int(params[2])
    if locx == 0:
        locx = np.random.randint(1, 10)
    locx = (locx - 5) / 20.0

    locy = int(params[3])
    if locy == 0:
        locy = np.random.randint(1, 10)
    locy = (locy - 5) / 20.0

    # k/10, 0.0-1.0
    color = int(params[4])
    if color == 0:
        color = np.random.randint(1, 10)
    color = color / 10.0

    resolution = 1000
    lutx = []
    luty = []
    lut2 = []
    for i in range(64):
        for j in range(64):
            x = (i - 32.0) / 32.0 - locx
            y = (j - 32.0) / 32.0 - locy
            if x ** 2 + y ** 2 <= size ** 2:
                lutx.append(i)
                luty.append(j)
                lut2.append(int((math.atan2(y, x) / 2.0 / math.pi + 0.5) * (resolution - 1)))

    random_color = np.random.uniform(0, 0.9, size=(count, 3))
    random_color[0, 0] = np.random.uniform(0.8, 0.9)
    random_color[0, 1:] = 0.0
    random_color[1:, 0] = 0.0
    random_weights = np.random.uniform(0.01, 1, size=count - 1)
    random_weights = np.concatenate([[color], (1 - color) * random_weights / np.sum(random_weights)])

    color_band = np.zeros(shape=(resolution, 3), dtype=np.float32)
    color_weights = np.cumsum(random_weights)
    color_weights = np.insert(color_weights, 0, 0.0)
    for c in range(len(random_color)):
        color_band[int(resolution * color_weights[c]):int(resolution * color_weights[c + 1])] = random_color[c]

    for i in range(3):
        swap_start, swap_end = 0, 0
        while swap_end - swap_start < 10:
            swap_start = np.random.randint(0, resolution // 2 - 10)
            swap_end = np.random.randint(10, resolution // 2)
        recv_start = np.random.randint(resolution // 2, resolution - swap_end + swap_start)
        recv_end = recv_start + swap_end - swap_start
        buffer = color_band[swap_start:swap_end].copy()
        color_band[swap_start:swap_end] = color_band[recv_start:recv_end]
        color_band[recv_start:recv_end] = buffer

    canvas = np.ones(shape=(64, 64, 3), dtype=np.float32)
    canvas[lutx, luty] = color_band[lut2]
    return canvas


def compute_radius(img):
    img = np.reshape(img, [-1, 3])
    size = float(len(np.argwhere(np.sum(img, axis=1) < 2.7)))
    radius = math.sqrt(size / math.pi)
    return radius / 32.0


def compute_proportion(img):
    img = np.reshape(img, [-1, 3])
    colors = img[np.argwhere(np.sum(img, axis=1) < 2.7)[:, 0]]
    reds = np.argwhere(colors[:, 0] > np.max(colors[:, 1:], axis=1))
    return float(reds.shape[0]) / colors.shape[0]


def compute_location(img):
    colors = np.argwhere(np.sum(img, axis=2) < 2.7)
    location = np.mean(colors, axis=0) / 32.0 - 1.0
    return location


class PieDataset(Dataset):
    def __init__(self, params=('40000',)):
        """ params is a tuple of strings, each string contain five integers,
        representing [num-of-color, size, x-location, y-location, proportion-of-red];
        each value is from 1-9, if value is 0, than this dimension is randomly selected.
        For data point selects a random param"""
        Dataset.__init__(self)
        self.data_dims = [64, 64, 3]
        self.name = "pie"
        self.batch_size = 100
        self.params = params

        self.train_ptr = 0
        self.train_cache = []
        self.max_size = 200000

        self.range = [0.0, 1.0]

    def next_batch(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        prev_ptr = self.train_ptr
        self.train_ptr += batch_size
        if self.train_ptr > self.max_size:
            prev_ptr = 0
            self.train_ptr = batch_size
        while self.train_ptr > len(self.train_cache):
            self.train_cache.append(gen_pie(np.random.choice(self.params)))
        return np.stack(self.train_cache[prev_ptr:self.train_ptr], axis=0)

    def reset(self):
        self.train_ptr = 0

    @staticmethod
    def eval_size(arr):
        return np.array([compute_radius(img) for img in arr])

    @staticmethod
    def eval_color_proportion(arr):
        return np.array([compute_proportion(img) for img in arr])

    @staticmethod
    def eval_location(arr):
        return np.stack([compute_location(img) for img in arr], axis=0)


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    dataset = PieDataset()
    images = dataset.next_batch(100)
    plt.figure(figsize=(6, 6))
    for i in range(0, 16):
        plt.subplot(4, 4, i+1)
        plt.imshow(images[i])
        plt.gca().xaxis.set_visible(False)
        plt.gca().yaxis.set_visible(False)
    plt.tight_layout()
    plt.savefig('img/pie_example.png')
    plt.show()
    # plt.show()
    # plt.subplot(2, 2, 1)
    # plt.hist(dataset.eval_location(images)[:, 0], range=(-0.5, 0.5), bins=30)
    # plt.subplot(2, 2, 2)
    # plt.hist(dataset.eval_location(images)[:, 1], range=(-0.5, 0.5), bins=30)
    # plt.subplot(2, 2, 3)
    # plt.hist(dataset.eval_color_proportion(images), range=(0, 1), bins=30)
    # plt.subplot(2, 2, 4)
    # plt.hist(dataset.eval_size(images), range=(0, 1), bins=30)
    # plt.show()
    # for i in range(0, 16):
    #     plt.subplot(4, 4, i+1)
    #     dataset.plot_colors(plt.gca(), labels[i])
    # plt.show()
    #
    # for i in range(0, 16):
    #     plt.subplot(4, 4, i+1)
    #     print(dataset.eval_colors(images[i]))
    #     dataset.plot_colors(plt.gca(), dataset.eval_colors(images[i]))
    # plt.show()
