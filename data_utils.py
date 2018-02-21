import numpy as np
import tensorflow as tf
import sys
import matplotlib.pyplot as plt
from scipy import io
import warnings
import os
import scipy.misc
from scipy import ndimage


class PrepData:
    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def writeTFRecord(self, inputs, labels, record_file_name, compress='GZIP'):
        if not os.path.exists('./tfrecords/'):
            os.makedirs('./tfrecords/')

        num_sample = inputs.shape[0]
        # open the TFRecords file
        if compress == 'GZIP':
            opt = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
        elif compress == 'ZLIB':
            opt = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
        else:
            opt = None

        writer = tf.python_io.TFRecordWriter('./tfrecords/' +
                                             str(num_sample) + '_' + record_file_name + '_'
                                             + compress + '.tfrecords', options=opt)

        for i in range(num_sample):
            # print how many images are saved every 1000 images
            if not i % 1000:
                print('Processing data: {}/{}'.format(i, num_sample))
                sys.stdout.flush()

            # Load the image
            inp = inputs[i, :, :, :]

            if len(labels.shape) > 2:
                lab = labels[i, :, :]
            else:
                lab = labels[i, :]

            # Create a feature
            feature = {'input': self._bytes_feature(inp.tostring()),
                       'label': self._bytes_feature(lab.tostring())}

            # Create an example protocol buffer
            example = tf.train.Example(features=tf.train.Features(feature=feature))

            # Serialize to string and write on the file
            writer.write(example.SerializeToString())

        writer.close()
        sys.stdout.flush()


def parser_s(record):
    img_size = 300
    keys_to_features = {
        "input": tf.FixedLenFeature((), tf.string),
        "label": tf.FixedLenFeature((), tf.string)
    }
    parsed = tf.parse_single_example(record, keys_to_features)

    # Perform additional preprocessing on the parsed data.
    inp = tf.decode_raw(parsed["input"], tf.float32) # Need to tune the datatype to get parse the right value
    inp = tf.reshape(inp, [img_size, img_size, 3])
    lab = tf.decode_raw(parsed["label"], tf.int32)
    lab = tf.reshape(lab, [img_size, img_size])

    return inp, lab


def form_api_dataset(ts_fname, batch_size, num_epoch=0, compression_type='GZIP'):
    dataset = tf.data.TFRecordDataset(ts_fname, compression_type=compression_type)
    dataset = dataset.map(parser_s)  # Parse the record into tensors.

    if num_epoch > 0:
        dataset = dataset.repeat(num_epoch)
    else:
        dataset = dataset.repeat()  # Repeat the input indefinitely.

    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()
    return iterator, next_element


def test_segmentize(net, directory_folder, image_size):
    pics = load_pics(directory_folder, image_size=image_size)
    net.segmentize(pics)


def load_pics(folder, image_size):
    image_files = os.listdir(folder)
    # image_files.sort()

    if image_files[0] == '.DS_Store':
        image_files.pop(0)

    content = np.ndarray(
        shape=(len(image_files),
               image_size,
               image_size,
               3),
        dtype=np.float32)

    index = 0
    for image in image_files:
        image_file = os.path.join(folder, image)
        try:
            image_data = ndimage.imread(image_file, mode='RGB')
            image_data = scipy.misc.imresize(image_data, (image_size, image_size))
            content[index, :, :, :] = image_data
            index = index + 1
        except IOError as e:
            print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')

    return content


def load_mat(matfile, key):
    dictionary = io.loadmat(matfile)
    content = dictionary.get(key)

    return content


# Use to iterate and show data in the tfrecord file
def showing_data_from_tfrecord(s_tr_fname, batch_size=1):
    with tf.Session() as session:

        s_tfrec_fname = tf.placeholder(tf.string, shape=[None], name='s_tfrec_fname')

        tf.global_variables_initializer().run()
        print('Initialized variable.')

        s_iterator, s_next_element = form_api_dataset(s_tfrec_fname, batch_size, num_epoch=1)

        # Supress the warning of matplotlib
        warnings.filterwarnings("ignore", ".*GUI is implemented.*")

        session.run(s_iterator.initializer, feed_dict={s_tfrec_fname: s_tr_fname})

        while True:
            try:
                s_elem = session.run(s_next_element)
                img = np.uint8(np.squeeze(s_elem[0]))
                lab = np.uint8(np.squeeze(s_elem[1]))
                lab_img = lab

                plt.ion()
                plt.subplot(1, 2, 1)
                plt.imshow(img)
                plt.title('Imgage')
                plt.axis('off')
                plt.subplot(1, 2, 2)
                plt.title('Label')
                plt.imshow(lab_img)
                plt.axis('off')
                plt.pause(0.05)
            except tf.errors.OutOfRangeError:
                break


def write_dataset_file(np_inputs, np_labels, wfname):
    prep_dat = PrepData()
    prep_dat.writeTFRecord(np_inputs, np_labels, wfname)


def manage_mat(feature_mat_file, label_mat_file, key):
    feas = np.float32(load_mat(feature_mat_file, key))
    labs = np.uint32(load_mat(label_mat_file, key+'_lab'))

    return feas, labs


def main():

    tr_fea, tr_lab = manage_mat('./mat_files/train.mat', './mat_files/train_lab.mat', 'train')

    write_dataset_file(tr_fea, tr_lab, 'trainS')

    va_fea, va_lab = manage_mat('./mat_files/val.mat', './mat_files/val_lab.mat', 'val')

    write_dataset_file(va_fea, va_lab, 'validS')


if __name__ == "__main__":
    main()
