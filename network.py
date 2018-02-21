import tensorflow as tf
from copy import deepcopy
import matplotlib.pyplot as plt
import warnings
from six.moves import cPickle as Pk
import numpy as np
from random import randint
from data_utils import form_api_dataset
import os


class Network(object):
    def __init__(self, inputs, num_class):
        self.inputs = inputs
        self.num_class = num_class

    # Operations -------------------------------------------------------------------------------------------------------
    def make_var(self, name, shape):
        return tf.get_variable(name, shape, trainable=True)

    def conv(self, inp, kernel_size, stride, c_o, name, relu=True, biased=True, padding='SAME'):

        # get num channel of input
        c_i = inp.get_shape()[-1]

        with tf.variable_scope(name) as scope:
            kernel = self.make_var('weights', shape=[kernel_size,
                                                     kernel_size,
                                                     c_i,
                                                     c_o])

            # Convolution for a given input and kernel
            output = tf.nn.conv2d(inp,
                                  kernel,
                                  [1, stride, stride, 1],
                                  padding=padding)
            # Add the biases
            if biased:
                # biases = self.make_var('biases', [c_o])
                biases = self.make_var('bi_'+name, [c_o])
                output = tf.nn.bias_add(output, biases)
            if relu:
                # ReLU non-linearity
                output = tf.nn.relu(output, name=scope.name)
            return output

    def atrous_conv(self, inp, kernel_size, c_o, dilation, name, relu=True, padding='SAME', biased=True):

        # Get the number of channels in the input
        c_i = inp.get_shape()[-1]

        with tf.variable_scope(name) as scope:
            kernel = self.make_var('weights', shape=[kernel_size, kernel_size, c_i, c_o])

            output = tf.nn.atrous_conv2d(inp, kernel, dilation, padding=padding)

            # Add the biases
            if biased:
                # biases = self.make_var('biases', [c_o])
                biases = self.make_var('bi_'+name, [c_o])
                output = tf.nn.bias_add(output, biases)
            if relu:
                # ReLU non-linearity
                output = tf.nn.relu(output, name=scope.name)
            return output

    def tran_conv(self, inp, kernel_size, out_shape, stride, c_o, name, relu=True, padding='SAME', biased=True):

        # Get the number of channels in the input
        c_i = inp.get_shape()[-1]

        with tf.variable_scope(name) as scope:
            kernel = self.make_var('weights', shape=[kernel_size, kernel_size, c_o, c_i])
            output = tf.nn.conv2d_transpose(inp, kernel, out_shape, [1, stride, stride, 1], padding=padding)

            # Add the biases
            if biased:
                biases = self.make_var('bias', [c_o])
                output = tf.nn.bias_add(output, biases)
            if relu:
                # ReLU non-linearity
                output = tf.nn.relu(output, name=scope.name)

            return output

    def relu(self, inp, name):
        return tf.nn.relu(inp, name=name)

    def max_pool(self, inp, kernal_size, stride, name, padding='SAME'):
        return tf.nn.max_pool(inp,
                              ksize=[1, kernal_size, kernal_size, 1],
                              strides=[1, stride, stride, 1],
                              padding=padding,
                              name=name)

    def batch_normalization(self, inp, name, is_training=True, activation_fn=None, scale=True):
        with tf.variable_scope(name) as scope:
            output = tf.contrib.slim.batch_norm(
                inp,
                activation_fn=activation_fn,
                is_training=is_training,
                updates_collections=None,
                scale=scale,
                scope=scope)
            return output

    def mul_ratio(self, inp, name):
        with tf.variable_scope(name) as scope:
            ratio = self.make_var('ratio', shape=inp.get_shape())

            # Multiply input and ratio
            output = tf.multiply(inp, ratio, name=scope.name)
            return output

    def dropout(self, inp, keep_prob, name):
        return tf.nn.dropout(inp, keep_prob, name=name)

    def add(self, inputs, name):
        return tf.add_n(inputs, name=name)

    def train_operation(self, loss, trainable_var, learning_rate=1e-15, name=None):
        if name is None:
            optimizer = tf.train.AdamOptimizer(learning_rate)
        else:
            optimizer = tf.train.AdamOptimizer(learning_rate, name=name)
        return optimizer.minimize(loss, var_list=trainable_var)

    def saveModel(self, saver, session, info):
        saver.save(session, './saved_model/model/'+self.__class__.__name__+'/'+self.__class__.__name__)
        try:
            with open('./saved_model/model/'+self.__class__.__name__+'/'+'info.pk', 'wb') as f:
                Pk.dump(info, f, Pk.HIGHEST_PROTOCOL)
            print('     Saved model...')
        except Exception as e:
            print('Unable to save info file,', e)

    def segmentize(self, features, batch_size):
        if not os.path.exists('./segmentized_samples/'):
            os.makedirs('./segmentized_samples/')

        tf.reset_default_graph()
        turns = features.shape[0]//batch_size

        j = 0

        with tf.Session() as session:
            loader = tf.train.import_meta_graph('./saved_model/model/'+self.__class__.__name__+'/'+self.__class__.__name__+'.meta')
            loader.restore(session, './saved_model/model/'+self.__class__.__name__+'/'+self.__class__.__name__)

            inputs = tf.get_collection("inputs")[0]

            s_predictions = tf.get_collection("s_predictions")[0]

            for t in range(turns):

                fd = {inputs: features[t*batch_size:t*batch_size+batch_size, :, :, :]}

                preds = session.run(s_predictions, feed_dict=fd)

                for i in range(batch_size):
                    plt.ion()
                    pred_raw = deepcopy(preds[i, :, :])
                    pred_img = pred_raw

                    in_img = np.uint8(features[j, :, :, :])

                    plt.subplot(1, 2, 1)
                    plt.imshow(in_img)
                    plt.axis('off')
                    plt.subplot(1, 2, 2)
                    plt.imshow(pred_img)
                    plt.axis('off')
                    plt.savefig('./segmentized_samples/'+str(j+1)+'.png')
                    j += 1
                    plt.pause(1.0)

    def compute_pixel_error(self, preds, labels):
        assert preds.shape == labels.shape

        sub = np.subtract(np.reshape(preds, [-1]), np.reshape(labels, [-1]))
        num_err = np.count_nonzero(sub)

        error = 100.00*(num_err/(len(sub)))

        # error = np.mean(np.multiply(np.divide(num_err, self.image_size*self.image_size), 100.0))
        return error

    def compute_iou(self, pred, lab, c):
        indlab = np.where(lab.flatten() == c)[0]
        if len(indlab) == 0:
            return -1

        indpred = np.where(pred.flatten() == c)[0]
        tp = np.intersect1d(indlab, indpred)
        fp = np.setdiff1d(indlab, indpred)
        fn = np.setdiff1d(indpred, indlab)

        iou = len(tp)/(len(fp)+len(tp)+len(fn))

        return iou

    def cal_miou(self, preds, labs):
        assert preds.shape == labs.shape
        ps = np.reshape(preds, [-1])
        ls = np.reshape(labs, [-1])

        miou = 0
        n = 0
        for i in range(0, self.num_class):
            iou = self.compute_iou(ps, ls, i)
            if iou >= 0:
                miou += iou
                n += 1
        return 100.00*(miou/n)


    def train(self, s_tr_fname, s_va_fname, batch_size, learning_rate, num_record, val_point=2):
        # Create directory for saved model
        if not os.path.exists('./saved_model/model/'):
            os.makedirs('./saved_model/model/')

        with tf.Session() as session:
            if os.path.exists('./saved_model/model/'+self.__class__.__name__+'/'):
                loader = tf.train.import_meta_graph('./saved_model/model/'+self.__class__.__name__+'/'+self.__class__.__name__+'.meta')
                loader.restore(session, './saved_model/model/'+self.__class__.__name__+'/'+self.__class__.__name__)

                inputs = tf.get_collection("inputs")[0]
                s_labels = tf.get_collection("s_labels")[0]
                s_loss = tf.get_collection("s_loss")[0]
                s_train_op = tf.get_collection("s_train_op")[0]
                s_predictions = tf.get_collection("s_predictions")[0]
                s_tfrec_fname = tf.get_collection("s_tfrec_fname")[0]

                print('Picked up saved model', self.__class__.__name__)
                try:
                    with open('./saved_model/model/'+self.__class__.__name__+'/'+'info.pk', 'rb') as f:
                        info = Pk.load(f)
                    step = info[0]+1
                except FileNotFoundError:
                    print('Cannot load info file')
                    step = 0
            else:
                inputs = tf.placeholder(
                    tf.float32,
                    shape=(batch_size, self.image_size, self.image_size, 3),
                    name='inputs'
                )

                s_labels = tf.placeholder(
                    tf.int32,
                    shape=(batch_size, self.image_size, self.image_size),
                    name='s_labels'
                )

                s_logits, s_predictions = self.model(inputs)

                s_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=s_labels, logits=s_logits))

                trainable_var = tf.trainable_variables()
                s_train_op = self.train_operation(s_loss, trainable_var, learning_rate, name='s_train_op')

                s_tfrec_fname = tf.placeholder(tf.string, shape=[None], name='s_tfrec_fname')
                tf.add_to_collection("s_tfrec_fname", s_tfrec_fname)

                # add operations to collection to restore the model later
                tf.add_to_collection("inputs", inputs)
                tf.add_to_collection("s_labels", s_labels)
                tf.add_to_collection('s_loss', s_loss)
                tf.add_to_collection("s_train_op", s_train_op)

                print('Initialized.')
                tf.global_variables_initializer().run()
                step = 0

            s_iterator, s_next_element = form_api_dataset(s_tfrec_fname, batch_size)

            # Supress the warning of matplotlib
            warnings.filterwarnings("ignore", ".*GUI is implemented.*")

            # Create a graph writer
            writer = tf.summary.FileWriter('./saved_model/graph/'+self.__class__.__name__+'/', session.graph)

            # Create a saver object which will save all the variables
            saver = tf.train.Saver(tf.global_variables())

            session.run(s_iterator.initializer,
                        feed_dict={
                            s_tfrec_fname: s_tr_fname
                        })

            while True:
                try:
                    s_elem = session.run(s_next_element)
                    fds = {inputs: s_elem[0], s_labels: s_elem[1]}

                    _, lss = session.run([s_train_op, s_loss], feed_dict=fds)

                    if step % np.ceil(num_record/batch_size) == 0:
                        epoch = int(step / (num_record/batch_size))
                        print('Start epoch #', epoch, ' - Segmentation loss is: %f' % lss)
                        summary = tf.Summary()
                        summary.value.add(tag='Segmentation Loss', simple_value=lss)

                        if epoch % val_point == 0:
                            # Validation -------------------------------------------------------------------------------
                            s_va_iter, s_va_ne = form_api_dataset(s_tfrec_fname, batch_size, 1)
                            session.run(s_va_iter.initializer,
                                        feed_dict={s_tfrec_fname: s_va_fname})

                            # segmentation validation process ----------------------------------------------------------
                            print('     Validating segmentation -------------------')
                            valid_input_s = None
                            valid_preds_s = None
                            valid_labs_s = None
                            while True:
                                try:
                                    va_s_elem = session.run(s_va_ne)
                                    va_s_feature = va_s_elem[0]
                                    va_fds = {inputs: va_s_feature}

                                    if valid_preds_s is None:
                                        valid_input_s = va_s_elem[0]
                                        valid_preds_s = session.run(s_predictions, feed_dict=va_fds)
                                        valid_labs_s = va_s_elem[1]
                                    else:
                                        valid_input_s = np.vstack((valid_input_s, va_s_elem[0]))
                                        valid_preds_s = np.vstack((valid_preds_s, session.run(s_predictions, feed_dict=va_fds)))
                                        valid_labs_s = np.vstack((valid_labs_s, va_s_elem[1]))
                                except tf.errors.OutOfRangeError:
                                    # Randomly get a single image from the training data
                                    rand_index = randint(0, valid_preds_s.shape[0]-1)
                                    pred_raw = deepcopy(valid_preds_s[rand_index, :, :])
                                    lab_raw = deepcopy(valid_labs_s[rand_index, :, :])

                                    in_img = np.uint8(deepcopy(valid_input_s[rand_index, :, :, :]))

                                    pred_img = np.uint8(pred_raw)
                                    lab_img = np.uint8(lab_raw)

                                    plt.ion()
                                    plt.subplot(1, 3, 1)
                                    plt.imshow(in_img)
                                    plt.title('Image')
                                    plt.axis('off')
                                    plt.subplot(1, 3, 2)
                                    plt.imshow(lab_img)
                                    plt.title('Label')
                                    plt.axis('off')
                                    plt.subplot(1, 3, 3)
                                    plt.imshow(pred_img)
                                    plt.title('Prediction')
                                    plt.axis('off')
                                    plt.pause(0.05)

                                    assert valid_preds_s.shape[0] == valid_labs_s.shape[0]

                                    # Compute mean pixel error
                                    err = self.compute_pixel_error(valid_preds_s, valid_labs_s)
                                    print('     Pixel error rate is: ', err)
                                    summary.value.add(tag='Pixel Error Rate', simple_value=err)
                                    # Compute Mean Intersection over Union
                                    MIoU = self.cal_miou(valid_preds_s, valid_labs_s)
                                    print('     Mean Intersection over Union: ', MIoU)
                                    summary.value.add(tag='MIoU', simple_value=MIoU)
                                    break
                                # --------------------------------------------------------------------------------------

                        writer.add_summary(summary, epoch)
                        writer.flush()

                        self.saveModel(saver, session, [step])

                        # if epoch == 500:
                        #     break

                    step += 1
                except tf.errors.OutOfRangeError:
                    break


