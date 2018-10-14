from dataset import *
import time
from models import *
from vae import VAE
from gan import GAN
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--z_dim', type=int, default=100, help='z dimension')
parser.add_argument('--dataset', type=str, default='pie_30', help='e.g. dots_3, pie_30')
parser.add_argument('--objective', type=str, default='gan', help='vae or gan')
parser.add_argument('--architecture', type=str, default='conv', help='Current options are conv, small, fc')
parser.add_argument('--data_path', type=str, default='/data/dots/')
parser.add_argument('--log_path', type=str, default='log')

parser.add_argument('--run', type=int, default=0, help='Index of the run')
parser.add_argument('--lr', type=float, default=-4.0, help='log_10 of initial learning rate')
parser.add_argument('--beta', type=float, default=1.0, help='Coefficient of KL(q(z|x)||p(z)), only useful for VAE')
parser.add_argument('--drep', type=int, default=2, help='Number of times to train discriminator for each generator update, only useful for GAN')
args = parser.parse_args()


batch_size = 100

# data_root = '/data/pie'
# log_root = '/data/empirical_gm'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
name = '%s/%s/model=%s-zdim=%d-lr=%.2f-beta=%.2f-drep=%d-run=%d' % \
       (args.dataset, args.objective, args.architecture, args.z_dim, args.lr, args.beta, args.drep, args.run)
log_path = os.path.join(args.log_path, name)
make_model_path(log_path)


# Parse args.dataset to create the dataset object
assert 'dots' in args.dataset or 'pie' in args.dataset
if 'dots' in args.dataset:
    splited = args.dataset.split('_')
    db_path = []
    for item in splited[1:]:
        db_path.append(os.path.join(args.data_path, '%s_dots' % item))
    dataset = DotsDataset(db_path=db_path)
else:
    splited = args.dataset.split('_')
    num_params = int(splited[1])

    fixed_dim = -1
    if len(splited) > 2:
        fixed_dim = int(splited[2])
        fixed_options = [int(item) for item in list(splited[3])]
        cur_option = 0

    params = []
    size_list = [1, 3, 5, 7, 9]
    locx_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    locy_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    color_list = [1, 3, 5, 7, 9]

    np.random.seed(args.run)
    for i in range(num_params):
        param = None
        while param is None or param in params:
            if fixed_dim != 0:
                size = np.random.choice(size_list)
            else:
                size = fixed_options[cur_option]
            if fixed_dim != 1:
                locx = np.random.choice(locx_list)
            else:
                locx = fixed_options[cur_option]
            if fixed_dim != 2:
                locy = np.random.choice(locy_list)
            else:
                locy = fixed_options[cur_option]
            if fixed_dim != 3:
                color = np.random.choice(color_list)
            else:
                color = fixed_options[cur_option]
            param = '4%d%d%d%d' % (size, locx, locy, color)
        if fixed_dim >= 0:
            cur_option = (cur_option + 1) % len(fixed_options)
        params.append(param)

    dataset = PieDataset(params=params)

# Create model object
assert args.objective in ['vae', 'gan']
if args.objective == 'vae':
    model = VAE(args, dataset, log_path)
else:
    model = GAN(args, dataset, log_path)

# Training
start_time = time.time()
for idx in range(1, 200001):
    bx = dataset.next_batch(batch_size)
    model.train_step(bx)

    if idx % 100 == 0:
        print("Iteration: [%6d] time: %4.2f" % (idx, time.time() - start_time))

    if idx % 10000 == 0:
        bxg = model.sample(20480)
        bx_list = []
        for rep in range(20):
            bx_list.append(dataset.next_batch(1024))
        bx = np.concatenate(bx_list, axis=0)
        np.savez(os.path.join(log_path, 'samples%d.npz' % (idx // 10000)), g=bxg, x=bx)
        model.save()
