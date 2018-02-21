from models import KNetS


def main():
    s_tr_fname = ['./tfrecords/561_trainS_GZIP.tfrecords']
    s_va_fname = ['./tfrecords/140_validS_GZIP.tfrecords']
    nr = 561

    knets = KNetS(image_size=300, num_class=32)
    knets.train(s_tr_fname, s_va_fname, batch_size=1, learning_rate=.0001, num_record=nr)


if __name__ == "__main__":
    main()
