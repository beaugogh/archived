import tensorflow as tf
from style_transfer import StyleTransfer
import utils

# enable eager execution
tf.enable_eager_execution()
print('Eager execution: {}'.format(tf.executing_eagerly()))

painter = StyleTransfer()
threshold = painter.miss_percentage_threshold
while True:
    best_img, miss_percentage = painter.run()
    if best_img is not None:
        utils.remove_earlier_checkpoints('./checkpoints')
    if miss_percentage > threshold:
        painter.learning_rate = painter.learning_rate * painter.lr_decay_rate
