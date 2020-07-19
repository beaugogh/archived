import os
import time
import utils
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe


class StyleTransfer():
    def __init__(self):
        print('initializing......')
        self.img_height = 600
        self.img_width = 600
        self.content_pic_name = 'dogs.jpg'
        self.style_pic_name = 'vege.jpg'
        self.content_path = './content/' + self.content_pic_name
        self.style_path = './style/' + self.style_pic_name
        self.output_path = './output'

        self.content_layers = ['block5_conv2']
        self.style_layers = ['block1_conv1',
                             'block2_conv1',
                             'block3_conv1',
                             'block4_conv1',
                             'block5_conv1']
        self.style_layer_weights = [0.5, 1.0, 1.5, 3.0, 4.0]
        self.num_iterations = 500
        self.miss_percentage_threshold = 0.4
        self.lr_decay_rate = 0.8
        self.content_weight = 1
        self.style_weight = 1
        self.variation_weight = 0.01
        self.best_loss = float('inf')

        # params for adam optimizer
        lr = 6
        self.learning_rate = \
            tfe.Variable(lr, name='lr', dtype=tf.float32, trainable=False)  # default: 0.001
        self.beta1 = 0.9  # default: 0.9,
        self.beta2 = 0.999  # default: 0.999
        self.epsilon = 1e-8  # default: 1e-08

        # create functor model for eager execution
        # keep the layers of the model off training,
        # since it is the input image itself getting 'trained' in back propagation
        self.model = self._get_model()
        for layer in self.model.layers:
            layer.trainable = False
        # pass the content and style images through the model
        # to get the content and style features in batch
        content_image = \
            utils.load_img(self.content_path, self.img_height, self.img_width)
        style_image = \
            utils.load_img(self.style_path, self.img_height, self.img_width)
        stack_images = np.concatenate([style_image, content_image], axis=0)
        model_outputs = self.model(stack_images)
        num_style_layers = len(self.style_layers)
        self.content_features = \
            [content_layer[1] for content_layer in model_outputs[num_style_layers:]]
        self.style_features = \
            [style_layer[0] for style_layer in model_outputs[:num_style_layers]]
        self.gram_style_features = \
            [self._gram_matrix(style_feature) for style_feature in self.style_features]

        # set initial image
        self.init_image = \
            utils.load_img(self.content_path, self.img_height, self.img_width)
        self.init_image = \
            tfe.Variable(self.init_image, dtype=tf.float32)

    def _get_model(self):
        vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
        vgg.trainable = False
        # Get output layers corresponding to style and content layers
        style_outputs = [vgg.get_layer(name).output for name in self.style_layers]
        content_outputs = [vgg.get_layer(name).output for name in self.content_layers]
        model_outputs = style_outputs + content_outputs
        # Build model
        model = tf.keras.Model(vgg.input, model_outputs)
        return model

    def _content_loss(self, content, output_content):
        return tf.reduce_mean(tf.square(output_content - content))

    def _gram_matrix(self, input_tensor):
        channels = int(input_tensor.shape[-1])
        a = tf.reshape(input_tensor, [-1, channels])
        n = tf.shape(a)[0]
        gram = tf.matmul(a, a, transpose_a=True)
        return gram / tf.cast(n, tf.float32)

    def _style_loss(self, gram_style, output_style):
        output_gram_style = self._gram_matrix(output_style)
        return tf.reduce_mean(tf.square(output_gram_style - gram_style))

    def _variation_loss(self, x):
        a = tf.square(
            x[:, :self.img_height - 1, :self.img_width - 1, :] -
            x[:, 1:, :self.img_width - 1, :])
        b = tf.square(
            x[:, :self.img_height - 1, :self.img_width - 1, :] -
            x[:, :self.img_height - 1, 1:, :])
        return tf.reduce_sum(tf.pow(a + b, 1.25))

    def _loss(self):
        # Feed our init image through our model. This will give us the content and
        # style representations at our desired layers. Since we're using eager
        # our model is callable just like any other function!
        model_outputs = self.model(self.init_image)

        num_style_layers = len(self.style_layers)
        num_content_layers = len(self.content_layers)

        style_output_features = model_outputs[:num_style_layers]
        content_output_features = model_outputs[num_style_layers:]

        style_loss = 0
        content_loss = 0

        # Accumulate style losses from all layers
        # Here, we equally weight each contribution of each loss layer
        # weight_per_style_layer = 1.0 / float(num_style_layers)
        index = 0
        for gram_style, output_style in zip(self.gram_style_features, style_output_features):
            # style_loss += weight_per_style_layer * self._style_loss(gram_style, output_style[0])
            style_loss += self.style_layer_weights[index] * self._style_loss(gram_style, output_style[0])
            index += 1

        # Accumulate content losses from all layers
        weight_per_content_layer = 1.0 / float(num_content_layers)
        for content, output_content in zip(self.content_features, content_output_features):
            content_loss += weight_per_content_layer * self._content_loss(content, output_content[0])

        variation_loss = self._variation_loss(self.init_image)

        style_loss *= self.style_weight
        content_loss *= self.content_weight
        variation_loss *= self.variation_weight

        # Get total loss
        loss = style_loss + content_loss + variation_loss
        return loss, style_loss, content_loss, variation_loss

    def compute_grads(self):
        with tf.GradientTape() as tape:
            all_loss = self._loss()
        # Compute gradients wrt input image
        total_loss = all_loss[0]
        return tape.gradient(total_loss, self.init_image), all_loss

    def run(self):
        print('running style transfer... lr = ', self.learning_rate)
        print('-----------------------------------------------------------')

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                beta1=self.beta1,
                                                beta2=self.beta2,
                                                epsilon=self.epsilon)
        # store the best result
        start_time = time.time()
        global_start = time.time()
        best_img = None
        best_loss = self.best_loss * 1.01

        # meta training params
        immediate_save = False
        num_iterations_with_increased_loss = 0.0
        miss_percentage = 0.0

        # checkpoint variables
        check_num = 50
        check_path = './checkpoints'
        check_prefix = os.path.join(check_path, 'ckpt')
        checkpoint = tfe.Checkpoint(optimizer=self.optimizer, model=self.model)
        checkpoint_status = \
            tf.train.get_checkpoint_state(os.path.dirname(check_path+'/checkpoint'))
        checkpoint_restore = None
        if checkpoint_status:
            print('Found checkpoint: ', checkpoint_status.model_checkpoint_path)
            # checkpoint.init_image = self.init_image
            checkpoint_restore = checkpoint.restore(tf.train.latest_checkpoint(check_path))
        else:
            print('No checkpoint found...')
            print('')

        # optimizing
        for i in range(self.num_iterations + 1):
            print('......', i)
            save_to_path = self.output_path + '/out_' + str(i) + '.jpg'
            grads, all_loss = self.compute_grads()
            loss, style_loss, content_loss, variation_loss = all_loss
            # update the initial image with gradients
            self.optimizer.apply_gradients([(grads, self.init_image)])
            # if i== 0 and checkpoint_restore is not None:
            #     checkpoint_restore.assert_consumed().run_restore_ops()

            if miss_percentage > self.miss_percentage_threshold:
                break

            if i == 0:
                utils.save_img(self.init_image, self.img_height, self.img_width, save_to_path)
                checkpoint.init_image = self.init_image
                checkpoint.save(file_prefix=check_prefix + '-' + str(i))

            if loss < best_loss:
                # Update best loss and best image from total loss.
                best_loss = loss
                if best_loss < self.best_loss:
                    self.best_loss = best_loss
                best_img = self.init_image

                if (i > 0 and i % check_num == 0) or immediate_save:
                    print('Iteration: {}, learning_rate: {}'.format(i, self.learning_rate))
                    print('Total loss: {:.4e} vs. best_loss: {:.4e} vs. self.best_loss: {:.4e} '
                          .format(loss, best_loss, self.best_loss))
                    print('Total loss: {:.4e}, '
                          'style loss: {:.4e}, '
                          'content loss: {:.4e}, '
                          'variation loss: {:.4e}, '
                          'time: {:.4f}s'.format(loss,
                                                 style_loss,
                                                 content_loss,
                                                 variation_loss,
                                                 time.time() - start_time))
                    start_time = time.time()
                    utils.save_img(self.init_image, self.img_height, self.img_width, save_to_path)
                    immediate_save = False
                    checkpoint.init_image = self.init_image
                    checkpoint.save(file_prefix=check_prefix + '-' + str(i))

            else:
                print('...... {}: loss increased, miss-percentage = {:.1f}%'
                      .format(i, miss_percentage*100))
                print('learning rate = ', self.learning_rate)
                print('Total loss: {:.4e} vs. best_loss: {:.4e} vs. self.best_loss: {:.4e} '
                      .format(loss, best_loss, self.best_loss))
                num_iterations_with_increased_loss += 1.0
                miss_percentage = num_iterations_with_increased_loss / self.num_iterations
                immediate_save = True

        print('Total time: {:.4f}s'.format(time.time() - global_start))

        return best_img, miss_percentage
