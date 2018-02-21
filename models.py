import tensorflow as tf
from network import Network


class KNetS(Network):
    def __init__(self, image_size, num_class, keep_prob=0.7):
        self.image_size = image_size
        self.num_class = num_class
        self.keep_prob = keep_prob

    def model(self, inp):
        batch_size = inp.get_shape().as_list()[0]

        conv1 = self.conv(inp, 7, 2, 32, name='conv1', relu=False, biased=True)
        bn_conv1 = self.batch_normalization(conv1, name='bn_conv1', activation_fn=None)

        # --------------------------------------------------------------------------------------------------------------

        conv2 = self.conv(bn_conv1, 4, 1, 64, name='conv2', relu=False, biased=True)
        bn_conv2 = self.batch_normalization(conv2, name='bn_conv2', activation_fn=None)

        # --------------------------------------------------------------------------------------------------------------

        conv3 = self.conv(bn_conv2, 3, 1, 128, name='conv3', relu=False, biased=True)
        bn_conv3 = self.batch_normalization(conv3, name='bn_conv3', activation_fn=tf.nn.relu)

        tran_conv1 = self.tran_conv(bn_conv3, 2, [batch_size, 150, 150, 128], 1, 128, name='tran_conv1', relu=False)

        pool1 = self.max_pool(tran_conv1, 3, 2, name='pool1')

        # --------------------------------------------------------------------------------------------------------------

        c2a_b1 = self.conv(pool1, 1, 1, 256, name='c2a_b1', relu=False, biased=False)
        bn2a_b1 = self.batch_normalization(c2a_b1, name='bn2a_b1', activation_fn=None)

        c2a_b2a = self.conv(pool1, 1, 1, 64, name='c2a_b2a', relu=False, biased=False)
        bn2a_b2a = self.batch_normalization(c2a_b2a, name='bn2a_b2a', activation_fn=tf.nn.relu)

        c2a_b2b = self.conv(bn2a_b2a, 3, 1, 64, name='c2a_b2b', relu=False)
        bn2a_b2b = self.batch_normalization(c2a_b2b, name='bn2a_b2b', activation_fn=tf.nn.relu)

        c2a_b2c = self.conv(bn2a_b2b, 1, 1, 256, name='c2a_b2c', biased=False, relu=False)
        bn2a_b2c = self.batch_normalization(c2a_b2c, name='bn2a_b2c', activation_fn=None)

        c2a = self.add([bn2a_b1, bn2a_b2c], name='c2a')
        drpt1 = self.dropout(c2a, self.keep_prob, name='drpt1')
        c2a_relu = self.relu(drpt1, name='c2a_relu')

        # --------------------------------------------------------------------------------------------------------------

        c2b_b2a = self.conv(c2a_relu, 1, 1, 64, name='c2b_b2a', biased=False, relu=False)
        bn2b_b2a = self.batch_normalization(c2b_b2a, name='bn2b_b2a', activation_fn=tf.nn.relu)

        c2b_b2b = self.conv(bn2b_b2a, 3, 1, 64, biased=False, relu=False, name='c2b_b2b')
        bn2b_b2b = self.batch_normalization(c2b_b2b, activation_fn=tf.nn.relu, name='bn2b_b2b')

        c2b_b2c = self.conv(bn2b_b2b, 1, 1, 256,  biased=False, relu=False, name='c2b_b2c')
        bn2b_b2c = self.batch_normalization(c2b_b2c, activation_fn=None, name='bn2b_b2c')

        c2b = self.add([c2a_relu, bn2b_b2c], name='c2b')
        drpt2 = self.dropout(c2b, self.keep_prob, name='drpt2')
        c2b_relu = self.relu(drpt2, name='c2b_relu')

        # --------------------------------------------------------------------------------------------------------------

        c2c_b2a = self.conv(c2b_relu, 1, 1, 64, biased=False, relu=False, name='c2c_b2a')
        bn2c_b2a = self.batch_normalization(c2c_b2a, activation_fn=tf.nn.relu, name='bn2c_b2a')

        c2c_b2b = self.conv(bn2c_b2a, 3, 1, 64, biased=False, relu=False, name='c2c_b2b')
        bn2c_b2b = self.batch_normalization(c2c_b2b, activation_fn=tf.nn.relu, name='bn2c_b2b')

        c2c_b2c = self.conv(bn2c_b2b, 1, 1, 256, biased=False, relu=False, name='c2c_b2c')
        bn2c_b2c = self.batch_normalization(c2c_b2c, activation_fn=None, name='bn2c_b2c')

        c2c = self.add([c2b_relu, bn2c_b2c], name='c2c')
        drpt3 = self.dropout(c2c, self.keep_prob, name='drpt3')
        c2c_relu = self.relu(drpt3, name='c2c_relu')

        # --------------------------------------------------------------------------------------------------------------

        c3a_b1 = self.conv(c2c_relu, 1, 2, 512, biased=False, relu=False, name='c3a_b1')
        bn3a_b1 = self.batch_normalization(c3a_b1, activation_fn=None, name='bn3a_b1')

        c3a_b2a = self.conv(c2c_relu, 1, 2, 128, biased=False, relu=False, name='c3a_b2a')
        bn3a_b2a = self.batch_normalization(c3a_b2a, activation_fn=tf.nn.relu, name='bn3a_b2a')

        c3a_b2b = self.conv(bn3a_b2a, 3, 1, 128, biased=False, relu=False, name='c3a_b2b')
        bn3a_b2b = self.batch_normalization(c3a_b2b, activation_fn=tf.nn.relu, name='bn3a_b2b')

        c3a_b2c = self.conv(bn3a_b2b, 1, 1, 512, biased=False, relu=False, name='c3a_b2c')
        bn3a_b2c = self.batch_normalization(c3a_b2c, activation_fn=None, name='bn3a_b2c')

        c3a = self.add([bn3a_b1, bn3a_b2c], name='c3a')
        drpt4 = self.dropout(c3a, self.keep_prob, name='drpt4')
        c3a_relu = self.relu(drpt4, name='c3a_relu')

        # --------------------------------------------------------------------------------------------------------------

        c3b1_b2a = self.conv(c3a_relu, 1, 1, 128, biased=False, relu=False, name='c3b1_b2a')
        bn3b1_b2a = self.batch_normalization(c3b1_b2a, activation_fn=tf.nn.relu, name='bn3b1_b2a')

        c3b1_b2b = self.conv(bn3b1_b2a, 3, 1, 128, biased=False, relu=False, name='c3b1_b2b')
        bn3b1_b2b = self.batch_normalization(c3b1_b2b, activation_fn=tf.nn.relu, name='bn3b1_b2b')

        c3b1_b2c = self.conv(bn3b1_b2b, 1, 1, 512, biased=False, relu=False, name='c3b1_b2c')
        bn3b1_b2c = self.batch_normalization(c3b1_b2c, activation_fn=None, name='bn3b1_b2c')

        c3b1 = self.add([c3a_relu, bn3b1_b2c], name='c3b1')
        drpt5 = self.dropout(c3b1, self.keep_prob, name='drpt5')
        c3b1_relu = self.relu(drpt5, name='c3b1_relu')

        # --------------------------------------------------------------------------------------------------------------

        c3b2_b2a = self.conv(c3b1_relu, 1, 1, 128, biased=False, relu=False, name='c3b2_b2a')
        bn3b2_b2a = self.batch_normalization(c3b2_b2a, activation_fn=tf.nn.relu, name='bn3b2_b2a')

        c3b2_b2b = self.conv(bn3b2_b2a, 3, 1, 128, biased=False, relu=False, name='c3b2_b2b')
        bn3b2_b2b = self.batch_normalization(c3b2_b2b, activation_fn=tf.nn.relu, name='bn3b2_b2b')

        c3b2_b2c = self.conv(bn3b2_b2b, 1, 1, 512, biased=False, relu=False, name='c3b2_b2c')
        bn3b2_b2c = self.batch_normalization(c3b2_b2c, activation_fn=None, name='bn3b2_b2c')

        c3b2 = self.add([c3b1_relu, bn3b2_b2c], name='c3b2')
        drpt6 = self.dropout(c3b2, self.keep_prob, name='drpt6')
        c3b2_relu = self.relu(drpt6, name='c3b2_relu')

        # --------------------------------------------------------------------------------------------------------------

        c3b3_b2a = self.conv(c3b2_relu, 1, 1, 128, biased=False, relu=False, name='c3b3_b2a')
        bn3b3_b2a = self.batch_normalization(c3b3_b2a, activation_fn=tf.nn.relu, name='bn3b3_b2a')

        c3b3_b2b = self.conv(bn3b3_b2a, 3, 1, 128, biased=False, relu=False, name='c3b3_b2b')
        bn3b3_b2b = self.batch_normalization(c3b3_b2b, activation_fn=tf.nn.relu, name='bn3b3_b2b')

        c3b3_b2c = self.conv(bn3b3_b2b, 1, 1, 512, biased=False, relu=False, name='c3b3_b2c')
        bn3b3_b2c = self.batch_normalization(c3b3_b2c, activation_fn=None, name='bn3b3_b2c')

        c3b3 = self.add([c3b2_relu, bn3b3_b2c], name='c3b3')
        drpt7 = self.dropout(c3b3, self.keep_prob, name='drpt7')
        c3b3_relu = self.relu(drpt7, name='c3b3_relu')

        # --------------------------------------------------------------------------------------------------------------

        c4a_b1 = self.conv(c3b3_relu, 1, 1, 1024, biased=False, relu=False, name='c4a_b1')
        bn4a_b1 = self.batch_normalization(c4a_b1, activation_fn=None, name='bn4a_b1')

        c4a_b2a = self.conv(c3b3_relu, 1, 1, 256, biased=False, relu=False, name='c4a_b2a')
        bn4a_b2a = self.batch_normalization(c4a_b2a, activation_fn=tf.nn.relu, name='bn4a_b2a')

        c4a_b2b = self.atrous_conv(bn4a_b2a, 3, 256, 2, padding='SAME', biased=False, relu=False, name='c4a_b2b')
        bn4a_b2b = self.batch_normalization(c4a_b2b, activation_fn=tf.nn.relu, name='bn4a_b2b')

        c4a_b2c = self.conv(bn4a_b2b, 1, 1, 1024, biased=False, relu=False, name='c4a_b2c')
        bn4a_b2c = self.batch_normalization(c4a_b2c, activation_fn=None, name='bn4a_b2c')

        # New additions ------------------------------------------------------------------------------------------------

        c4a = self.add([bn4a_b1, bn4a_b2c], name='c4a')
        drpt8 = self.dropout(c4a, self.keep_prob, name='drpt8')
        c4a_relu = self.relu(drpt8, name='c4a_relu')

        c4b1_b2a = self.conv(c4a_relu, 1, 1, 256, biased=False, relu=False, name='c4b1_b2a')
        bn4b1_b2a = self.batch_normalization(c4b1_b2a, activation_fn=tf.nn.relu, name='bn4b1_b2a')

        c4b1_b2b = self.atrous_conv(bn4b1_b2a, 3, 256, 2, padding='SAME', biased=False, relu=False, name='c4b1_b2b')
        bn4b1_b2b = self.batch_normalization(c4b1_b2b, activation_fn=tf.nn.relu, name='bn4b1_b2b')

        c4b1_b2c = self.conv(bn4b1_b2b, 1, 1, 1024, biased=False, relu=False, name='c4b1_b2c')
        bn4b1_b2c = self.batch_normalization(c4b1_b2c, activation_fn=None, name='bn4b1_b2c')

        # --------------------------------------------------------------------------------------------------------------

        c4b1 = self.add([c4a_relu, bn4b1_b2c], name='c4b1')
        drpt9 = self.dropout(c4b1, self.keep_prob, name='drpt9')
        c4b1_relu = self.relu(drpt9, name='c4b1_relu')

        c4b2_b2a = self.conv(c4b1_relu, 1, 1, 256, biased=False, relu=False, name='c4b2_b2a')
        bn4b2_b2a = self.batch_normalization(c4b2_b2a, activation_fn=tf.nn.relu, name='bn4b2_b2a')

        c4b2_b2b = self.atrous_conv(bn4b2_b2a, 3, 256, 2, padding='SAME', biased=False, relu=False, name='c4b2_b2b')
        bn4b2_b2b = self.batch_normalization(c4b2_b2b, activation_fn=tf.nn.relu, name='bn4b2_b2b')

        c4b2_b2c = self.conv(bn4b2_b2b, 1, 1, 1024, biased=False, relu=False, name='c4b2_b2c')
        bn4b2_b2c = self.batch_normalization(c4b2_b2c, activation_fn=None, name='bn4b2_b2c')

        # --------------------------------------------------------------------------------------------------------------

        c4b2 = self.add([c4b1_relu, bn4b2_b2c], name='c4b2')
        drpt10 = self.dropout(c4b2, self.keep_prob, name='drpt10')
        c4b2_relu = self.relu(drpt10, name='c4b2_relu')

        c4b3_b2a = self.conv(c4b2_relu, 1, 1, 256, biased=False, relu=False, name='c4b3_b2a')
        bn4b3_b2a = self.batch_normalization(c4b3_b2a, activation_fn=tf.nn.relu, name='bn4b3_b2a')

        c4b3_b2b = self.atrous_conv(bn4b3_b2a, 3, 256, 2, padding='SAME', biased=False, relu=False, name='c4b3_b2b')
        bn4b3_b2b = self.batch_normalization(c4b3_b2b, activation_fn=tf.nn.relu, name='bn4b3_b2b')

        c4b3_b2c = self.conv(bn4b3_b2b, 1, 1, 1024, biased=False, relu=False, name='c4b3_b2c')
        bn4b3_b2c = self.batch_normalization(c4b3_b2c, activation_fn=None, name='bn4b3_b2c')

        # --------------------------------------------------------------------------------------------------------------

        c4b3 = self.add([c4b2_relu, bn4b3_b2c], name='c4b3')
        drpt11 = self.dropout(c4b3, self.keep_prob, name='drpt11')
        c4b3_relu = self.relu(drpt11, name='c4b3_relu')

        c4b4_b2a = self.conv(c4b3_relu, 1, 1, 256, biased=False, relu=False, name='c4b4_b2a')
        bn4b4_b2a = self.batch_normalization(c4b4_b2a, activation_fn=tf.nn.relu, name='bn4b4_b2a')

        c4b4_b2b = self.atrous_conv(bn4b4_b2a, 3, 256, 2, padding='SAME', biased=False, relu=False, name='c4b4_b2b')
        bn4b4_b2b = self.batch_normalization(c4b4_b2b, activation_fn=tf.nn.relu, name='bn4b4_b2b')

        c4b4_b2c = self.conv(bn4b4_b2b, 1, 1, 1024, biased=False, relu=False, name='c4b4_b2c')
        bn4b4_b2c = self.batch_normalization(c4b4_b2c, activation_fn=None, name='bn4b4_b2c')

        # --------------------------------------------------------------------------------------------------------------
        c4b4 = self.add([c4b3_relu, bn4b4_b2c], name='c4b4')
        drpt12 = self.dropout(c4b4, self.keep_prob, name='drpt12')
        c4b4_relu = self.relu(drpt12, name='c4b4_relu')

        c4b5_b2a = self.conv(c4b4_relu, 1, 1, 256, biased=False, relu=False, name='c4b5_b2a')
        bn4b5_b2a = self.batch_normalization(c4b5_b2a, activation_fn=tf.nn.relu, name='bn4b5_b2a')

        c4b5_b2b = self.atrous_conv(bn4b5_b2a, 3, 256, 2, padding='SAME', biased=False, relu=False, name='c4b5_b2b')
        bn4b5_b2b = self.batch_normalization(c4b5_b2b, activation_fn=tf.nn.relu, name='bn4b5_b2b')

        c4b5_b2c = self.conv(bn4b5_b2b, 1, 1, 1024, biased=False, relu=False, name='c4b5_b2c')
        bn4b5_b2c = self.batch_normalization(c4b5_b2c, activation_fn=None, name='bn4b5_b2c')

        # --------------------------------------------------------------------------------------------------------------

        c4b5 = self.add([c4b4_relu, bn4b5_b2c], name='c4b5')
        drpt13 = self.dropout(c4b5, self.keep_prob, name='drpt13')
        c4b5_relu = self.relu(drpt13, name='c4b5_relu')

        c4b6_b2a = self.conv(c4b5_relu, 1, 1, 256, biased=False, relu=False, name='c4b6_b2a')
        bn4b6_b2a = self.batch_normalization(c4b6_b2a, activation_fn=tf.nn.relu, name='bn4b6_b2a')

        c4b6_b2b = self.atrous_conv(bn4b6_b2a, 3, 256, 2, padding='SAME', biased=False, relu=False, name='c4b6_b2b')
        bn4b6_b2b = self.batch_normalization(c4b6_b2b, activation_fn=tf.nn.relu, name='bn4b6_b2b')

        c4b6_b2c = self.conv(bn4b6_b2b, 1, 1, 1024, biased=False, relu=False, name='c4b6_b2c')
        bn4b6_b2c = self.batch_normalization(c4b6_b2c, activation_fn=None, name='bn4b6_b2c')

        # --------------------------------------------------------------------------------------------------------------

        c5c = self.add([bn4b5_b2c, bn4b6_b2c], name='c5c')
        drpt14 = self.dropout(c5c, self.keep_prob, name='drpt14')
        c5c_relu = self.relu(drpt14, name='c5c_relu')

        fc1_c0 = self.atrous_conv(c5c_relu, kernel_size=3, c_o=self.num_class, 
                                  dilation=1, padding='SAME', relu=False, name='fc1_c0')

        fc1_c1 = self.atrous_conv(c5c_relu, 3, self.num_class, 2, padding='SAME', relu=False, name='fc1_c1')

        fc1_c2 = self.atrous_conv(c5c_relu, 3, self.num_class, 4, padding='SAME', relu=False, name='fc1_c2')

        fc1_c3 = self.atrous_conv(c5c_relu, 3, self.num_class, 8, padding='SAME', relu=False, name='fc1_c3')

        fc1 = self.add([fc1_c0, fc1_c1, fc1_c2, fc1_c3], name='fc1')

        # Upsampling - -------------------------------------------------------------------------------------------------

        tran_conv2 = self.tran_conv(fc1, 2,
                                [batch_size, 75, 75, 512],
                                2, 512, name='tran_conv2', relu=False)

        tran_conv2_bn = self.batch_normalization(tran_conv2, activation_fn=None, name='tran_conv2_bn')

        conv4 = self.conv(tran_conv2_bn, 2, 1, 256, name='conv4', relu=True, biased=True)
        drpt15 = self.dropout(conv4, self.keep_prob, name='drpt15')

        tran_conv3 = self.tran_conv(drpt15, 2,
                                [batch_size, 150, 150, 128],
                                2, 128, name='tran_conv3', relu=False)
        tran_conv3_bn = self.batch_normalization(tran_conv3, activation_fn=None, name='tran_conv3_bn')

        conv5 = self.conv(tran_conv3_bn, 2, 1, 64, name='conv5', relu=True, biased=True)
        drpt16 = self.dropout(conv5, self.keep_prob, name='drpt16')

        tran_conv4 = self.tran_conv(drpt16, 2,
                                [batch_size, self.image_size, self.image_size, self.num_class],
                                2, self.num_class, name='tran_conv4', relu=False)

        s_logits = self.batch_normalization(tran_conv4, activation_fn=tf.nn.relu, name='s_logits')
        tf.add_to_collection("s_logits", s_logits)

        s_predictions = tf.argmax(s_logits, axis=3, name='s_predictions')
        tf.add_to_collection("s_predictions", s_predictions)

        # --------------------------------------------------------------------------------------------------------------

        return s_logits, s_predictions


class FCN(Network):
    keep_prob = 0.7

    def __init__(self, image_size, num_class):
        self.image_size = image_size
        self.num_class = num_class

    def model(self, inp):
        # 1st Round
        conv1_1 = self.conv(inp, kernel_size=3, stride=1, c_o=64, name='conv1_1')
        conv1_2 = self.conv(conv1_1, 3, 1, 64, name='conv1_2')
        pool1 = self.max_pool(conv1_2, 2, 2, name='pool1')

        # 2nd Round
        conv2_1 = self.conv(pool1, 3, 1, 128, name='conv2_1')
        conv2_2 = self.conv(conv2_1, 3, 1, 128, name='conv2_2')
        pool2 = self.max_pool(conv2_2, 2, 2, name='pool2')

        # 3rd Round
        conv3_1 = self.conv(pool2, 3, 1, 256, name='conv3_1')
        conv3_2 = self.conv(conv3_1, 3, 1, 256, name='conv3_2')
        conv3_3 = self.conv(conv3_2, 3, 1, 256, name='conv3_3')
        pool3 = self.max_pool(conv3_3, 2, 2, name='pool3')

        # 4th Round
        conv4_1 = self.conv(pool3, 3, 1, 512, name='conv4_1')
        conv4_2 = self.conv(conv4_1, 3, 1, 512, name='conv4_2')
        conv4_3 = self.conv(conv4_2, 3, 1, 512, name='conv4_3')
        pool4 = self.max_pool(conv4_3, 2, 2, name='pool4')

        # 5th Round
        conv5_1 = self.conv(pool4, 3, 1, 512, name='conv5_1')
        conv5_2 = self.conv(conv5_1, 3, 1, 512, name='conv5_2')
        conv5_3 = self.conv(conv5_2, 3, 1, 512, name='conv5_3')
        pool5 = self.max_pool(conv5_3, 2, 2, name='pool5')

        # 6th Round
        conv6 = self.conv(pool5, 7, 1, 4096, name='conv6')
        dropout6 = self.dropout(conv6, self.keep_prob, name='dropout6')

        # 7th Round
        conv7 = self.conv(dropout6, 1, 1, 4096, name='conv7')
        dropout7 = self.dropout(conv7, self.keep_prob, name='dropout7')

        # 8th Round
        conv8 = self.conv(dropout7, 1, 1, self.num_class, name='conv8')

        # Deconvolution 1st Layer
        deconv_1 = self.tran_conv(conv8, kernel_size=4, out_shape=tf.shape(pool4), stride=2,
                                c_o=pool4.get_shape().as_list()[3], name='deconv_1', relu=False)
        fuse_1 = self.add([deconv_1, pool4], name='fuse_1')

        # Deconvolution 2nd Layer
        deconv_2 = self.tran_conv(fuse_1, 4, tf.shape(pool3), 2, pool3.get_shape().as_list()[3], name='deconv_2', relu=False)
        fuse_2 = self.add([deconv_2, pool3], name='fuse_2')

        # Deconvolution 3rd Layer
        out_shape3 = tf.stack([fuse_2.get_shape().as_list()[0], self.image_size, self.image_size, self.num_class])

        deconv_3 = self.tran_conv(fuse_2, 16, out_shape3, 8, self.num_class, name='deconv_3', relu=False)

        s_logits = self.batch_normalization(deconv_3, activation_fn=None, name='s_logits')

        tf.add_to_collection("s_logits", s_logits)

        s_predictions = tf.argmax(s_logits, axis=3, name='s_predictions')
        tf.add_to_collection("s_predictions", s_predictions)

        return s_logits, s_predictions
