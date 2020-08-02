import argparse
from utils import ImagePool, load_test_data, save_images
import matplotlib.pyplot as plt

# (unused) import matplotlib.image as mpimg
# (unused) from glob import glob

parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset_dir', dest='dataset_dir', default='gta', help='path of the dataset')
parser.add_argument('--epoch', dest='epoch', type=int, default=25, help='# of epoch')
parser.add_argument('--epoch_step', dest='epoch_step', type=int, default=100, help='# of epoch to decay lr')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='# images in batch')
parser.add_argument('--train_size', dest='train_size', type=int, default=1e8, help='# images used to train')
parser.add_argument('--img_height', dest='img_height', type=int, default=256, help='image height')
parser.add_argument('--img_width', dest='img_width', type=int, default=512, help='image width')
parser.add_argument('--ngf', dest='ngf', type=int, default=64, help='# of gen filters in first conv layer')
parser.add_argument('--ndf', dest='ndf', type=int, default=64, help='# of discri filters in first conv layer')
parser.add_argument('--input_nc', dest='input_nc', type=int, default=3, help='# of input image channels')
parser.add_argument('--output_nc', dest='output_nc', type=int, default=3, help='# of output image channels')
parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--which_direction', dest='which_direction', default='AtoB', help='AtoB or BtoA')
parser.add_argument('--phase', dest='phase', default='train', help='train, test')
parser.add_argument('--save_freq', dest='save_freq', type=int, default=1000, help='save a model every save_freq iterations')
parser.add_argument('--print_freq', dest='print_freq', type=int, default=100, help='print the debug information every print_freq iterations')
parser.add_argument('--continue_train', dest='continue_train', type=bool, default=False, help='if continue training, load the latest model: 1: true, 0: false')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='sample are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='./test', help='test sample are saved here')
parser.add_argument('--L1_lambda', dest='L1_lambda', type=float, default=10.0, help='weight on L1 term in objective')
parser.add_argument('--Lg_lambda', dest='Lg_lambda', type=float, default=5.0, help='weight on gradloss term in objective')
parser.add_argument('--use_resnet', dest='use_resnet', type=bool, default=False, help='generation network using residule block')
parser.add_argument('--use_lsgan', dest='use_lsgan', type=bool, default=True, help='gan loss defined in lsgan')
parser.add_argument('--max_size', dest='max_size', type=int, default=50, help='max size of image pool, 0 means do not use image pool')
parser.add_argument('--segment_class', dest='segment_class', type=int, default=8, help='number of segmentation classes')
args = parser.parse_args()

pool = ImagePool(10)

sample_file = "test/real_00007.png"
sample_image = load_test_data(sample_file, args.img_width, args.img_height)

# for epoch in range(args.epoch):
#     dataA = glob('./datasets/{}/*.*'.format(args.dataset_dir + '/trainA'))
#     dataB = glob('./datasets/{}/*.*'.format(args.dataset_dir + '/trainB'))
#     np.random.shuffle(dataA)
#     np.random.shuffle(dataB)
#     batch_idxs = min(min(len(dataA), len(dataB)), args.train_size) // args.batch_size
#     lr = args.lr if epoch < args.epoch_step else args.lr*(args.epoch-epoch)/(args.epoch-args.epoch_step)

#     for idx in range(0, batch_idxs):
#         batch_files = list(zip(dataA[idx * args.batch_size:(idx + 1) * args.batch_size],
#                                 dataB[idx * args.batch_size:(idx + 1) * args.batch_size]))
#         batch_images = []
#         batch_segs = []
#         batch_seg_mask_A = []
#         batch_seg_mask_B = []
#         for batch_file in batch_files:
#             tmp_image, tmp_seg, tmp_seg_mask_A, tmp_seg_mask_B = load_train_data(batch_file, args.img_width, args.img_height, num_seg_masks=args.segment_class)

print(sample_image.shape)
imgplot = plt.imshow(sample_image)
plt.show()

save_images(sample_image, [1, 1], "test/holis.png")
