import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import vgg19
import numpy as np
import scipy
from PIL import Image, ImageOps



# prepare the data
# Let's create methods that will allow us to load and preprocess our images easily.
# We perform the same preprocessing process as are expected according to the VGG training process.
# VGG networks are trained on images
# with each channel normalized by mean = [103.939, 116.779, 123.68]and with channels BGR.
def load_img(img_path, img_height, img_width):
    # note that the target_size param is (img_height, img_width)
    img = image.load_img(img_path, target_size=(img_height, img_width))
    img = image.img_to_array(img)
    # We need to broadcast the image array such that it has a batch dimension
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img

def save_img(img, img_height, img_width, img_path):
    x = img.numpy().copy().reshape((img_height, img_width, 3))
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    scipy.misc.imsave(img_path, x)
    return x


def generate_noise_image(content_path, height, width, noise_ratio=0.5):
    content_image = load_img(content_path, height, width)
    noise_image = np.random.uniform(-20, 20, (1, height, width, 3)).astype(np.float32)
    return noise_image * noise_ratio + content_image * (1 - noise_ratio)

def remove_earlier_checkpoints(checkpoint_folder_path):
    folder = checkpoint_folder_path
    checkpoint_path = os.path.join(folder, 'checkpoint')
    checkpoint = open(checkpoint_path, 'r')
    line = checkpoint.readline()
    parts = line.split(':')
    latest = parts[1][2:-2]
    print('latest checkpoint: ', latest)

    index = 0
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                name = file_path.split('/')[2]
                print('checkpoint file name: ', name)
                if((latest not in name) and ('checkpoint' != name)):
                    os.unlink(file_path)
                    index += 1
        except Exception as e:
            print(e)

    print('cleaned {} files'.format(index))
