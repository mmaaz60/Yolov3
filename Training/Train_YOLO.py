"""
MODIFIED FROM keras-yolo3 PACKAGE, https://github.com/qqwweee/keras-yolo3
Retrain the YOLO model for your own dataset.
"""

import os
import argparse
import numpy as np
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from time import time
from Utils.Train_Utils import get_classes, get_anchors, create_model, create_tiny_model, \
    data_generator_wrapper, ChangeToOtherMachine


def parse_args():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)

    parser.add_argument(
        "--annotation_file", type=str,  default="/home/maaz/Desktop/Visitor_Tracking/TrainingPipeline_TF2.0/"
                                                "TrainYourOwnYOLO/Data/ch04_yolo.txt",
        help="Path to annotation file for Yolo."
    )
    parser.add_argument(
        "--classes_file", type=str,  default="/home/maaz/Desktop/Visitor_Tracking/TrainingPipeline_TF2.0/"
                                             "TrainYourOwnYOLO/Data/person.names",
        help="Path to YOLO class names."
    )
    parser.add_argument(
        "--log_dir", type=str,  default="/home/maaz/Desktop/Visitor_Tracking/TrainingPipeline_TF2.0/"
                                        "TrainYourOwnYOLO/checkpoints/logs",
        help="Folder to save training logs and trained weights to."
    )
    parser.add_argument(
        "--anchors_path", type=str,  default="/home/maaz/Desktop/Visitor_Tracking/TrainingPipeline_TF2.0/"
                                             "TrainYourOwnYOLO/Data/yolo-tiny_anchors.txt",
        help="Path to YOLO anchors."
    )
    parser.add_argument(
        "--weights_path", type=str,  default="/home/maaz/Desktop/Visitor_Tracking/TrainingPipeline_TF2.0/"
                                             "TrainYourOwnYOLO/checkpoints/yolov3-tiny.h5",
        help="Path to pre-trained YOLO weights."
    )
    parser.add_argument(
        "--val_split", type=float, default=0.2,
        help="Percentage of training set to be used for validation. Default is 20%."
    )
    parser.add_argument(
        "--is_tiny", type=bool, default=True,
        help="Use the tiny Yolo version for better performance and less accuracy. Default is True."
    )
    parser.add_argument(
        "--network_input_shape", type=float, default=416,
        help="The input size of the network. The number must be a multiple of 32."
    )
    parser.add_argument(
        "--batch_size", type=float, default=32,
        help="The size of mini batches used for training."
    )
    parser.add_argument(
        "--random_seed", type=float, default=None,
        help="Random seed value to make script deterministic. Default is 'None', i.e. non-deterministic."
    )
    parser.add_argument(
        "--epochs", type=float, default=50,
        help="Number of epochs for training last layers and number of epochs for fine-tuning layers. Default is 50."
    )

    return parser.parse_args()


if __name__ == '__main__':
    # Parser command line arguments
    args = parse_args()

    np.random.seed(args.random_seed)
    log_dir = args.log_dir

    class_names = get_classes(args.classes_file)
    num_classes = len(class_names)

    anchors = get_anchors(args.anchors_path)
    weights_path = args.weights_path

    input_shape = (args.network_input_shape, args.network_input_shape)

    BATCH_SIZE = args.batch_size

    epoch1, epoch2 = args.epochs, 4*args.epochs

    # is_tiny_version = (len(anchors) == 6)  # default setting
    if args.is_tiny:
        # Create tiny yolov3 model
        model = create_tiny_model(input_shape, anchors, num_classes,
                                  freeze_body=2, weights_path=weights_path)
    else:
        # Create full yolov3 model
        model = create_model(input_shape, anchors, num_classes,
                             freeze_body=2, weights_path=weights_path)  # make sure you know what you freeze

    log_dir_time = os.path.join(log_dir, '{}'.format(int(time())))
    logging = TensorBoard(log_dir=log_dir_time)
    checkpoint = ModelCheckpoint(os.path.join(log_dir, 'checkpoint.h5'),
                                 monitor='val_loss', save_weights_only=True, save_best_only=True, period=5)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    val_split = args.val_split
    with open(args.annotation_file) as f:
        lines = f.readlines()

    # # This step makes sure that the path names correspond to the local machine
    # # This is important if annotation and training are done on different machines (e.g. training on AWS)
    # lines = ChangeToOtherMachine(lines, remote_machine='')
    np.random.shuffle(lines)
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val

    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a decent model.
    if True:
        model.compile(optimizer=Adam(lr=1e-3), loss={
            # use custom yolo_loss Lambda layer.
            'yolo_loss': lambda y_true, y_pred: y_pred})

        # batch_size = 32
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, BATCH_SIZE))
        history = model.fit_generator(
            data_generator_wrapper(lines[:num_train], BATCH_SIZE, input_shape, anchors, num_classes),
            steps_per_epoch=max(1, num_train // BATCH_SIZE),
            validation_data=data_generator_wrapper(lines[num_train:], BATCH_SIZE, input_shape, anchors, num_classes),
            validation_steps=max(1, num_val // BATCH_SIZE),
            epochs=epoch1,
            initial_epoch=0,
            callbacks=[logging, checkpoint])
        model.save_weights(os.path.join(log_dir, 'trained_weights_stage_1.h5'))

        step1_train_loss = history.history['loss']

        file = open(os.path.join(log_dir_time, 'step1_loss.npy'), "w")
        with open(os.path.join(log_dir_time, 'step1_loss.npy'), 'w') as f:
            for item in step1_train_loss:
                f.write("%s\n" % item)
        file.close()

        step1_val_loss = np.array(history.history['val_loss'])

        file = open(os.path.join(log_dir_time, 'step1_val_loss.npy'), "w")
        with open(os.path.join(log_dir_time, 'step1_val_loss.npy'), 'w') as f:
            for item in step1_val_loss:
                f.write("%s\n" % item)
        file.close()

    # Unfreeze and continue training, to fine-tune.
    # Train longer if the result is unsatisfactory.
    if True:
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        model.compile(optimizer=Adam(lr=1e-4),
                      loss={'yolo_loss': lambda y_true, y_pred: y_pred})  # recompile to apply the change
        # print('Unfreeze all layers.')

        # batch_size = 4  # note that more GPU memory is required after unfreezing the body
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, BATCH_SIZE))
        history = model.fit_generator(
            data_generator_wrapper(lines[:num_train], BATCH_SIZE, input_shape, anchors, num_classes),
            steps_per_epoch=max(1, num_train // BATCH_SIZE),
            validation_data=data_generator_wrapper(lines[num_train:], BATCH_SIZE, input_shape, anchors, num_classes),
            validation_steps=max(1, num_val // BATCH_SIZE),
            epochs=epoch1 + epoch2,
            initial_epoch=epoch1,
            callbacks=[logging, checkpoint, reduce_lr, early_stopping])
        model.save_weights(os.path.join(log_dir, 'trained_weights_final.h5'))
        step2_train_loss = history.history['loss']

        file = open(os.path.join(log_dir_time, 'step2_loss.npy'), "w")
        with open(os.path.join(log_dir_time, 'step2_loss.npy'), 'w') as f:
            for item in step2_train_loss:
                f.write("%s\n" % item)
        file.close()

        step2_val_loss = np.array(history.history['val_loss'])

        file = open(os.path.join(log_dir_time, 'step2_val_loss.npy'), "w")
        with open(os.path.join(log_dir_time, 'step2_val_loss.npy'), 'w') as f:
            for item in step2_val_loss:
                f.write("%s\n" % item)
        file.close()
