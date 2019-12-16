import os
import argparse
from yolo import YOLO, detect_video
from timeit import default_timer as timer
from Utils.utils import detect_object
import pandas as pd
import numpy as np
from Utils.Get_File_Paths import GetFileList

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def parse_args():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)

    parser.add_argument(
        "--input_path", type=str, default="/home/maaz/Desktop/Visitor_Tracking/TrainingPipeline_TF2.0"
                                          "/TrainYourOwnYOLO/Data/cam15/test_videos",
        help="Path to image/video directory. All subdirectories will be included."
    )
    parser.add_argument(
        "--output", type=str, default="/home/maaz/Desktop/Visitor_Tracking/TrainingPipeline_TF2.0/TrainYourOwnYOLO"
                                      "/Data/cam15/test_videos_results",
        help="Output path for detection results."
    )
    parser.add_argument(
        "--save_img", default=True, action="store_true",
        help="Save bounding box coordinates and output images with annotated boxes."
    )
    parser.add_argument(
        "--file_types", '--names-list', nargs='*', default=[],
        help="Specify list of file types to include. Default is --file_types .jpg .jpeg .png .mp4"
    )
    parser.add_argument(
        '--yolo_model', type=str, dest='model_path', default="/home/maaz/Desktop/Visitor_Tracking"
                                                             "/TrainingPipeline_TF2.0/TrainYourOwnYOLO/checkpoints"
                                                             "/cam15/logs/trained_weights_final.h5",
        help="Path to pre-trained weight files."
    )
    parser.add_argument(
        '--anchors', type=str, dest='anchors_path', default="/home/maaz/Desktop/Visitor_Tracking/TrainingPipeline_TF2"
                                                            ".0/TrainYourOwnYOLO/Data/yolo-tiny_anchors.txt",
        help="Path to YOLO anchors."
    )
    parser.add_argument(
        '--classes', type=str, dest='classes_path', default="/home/maaz/Desktop/Visitor_Tracking/TrainingPipeline_TF2"
                                                            ".0/TrainYourOwnYOLO/Data/person.names",
        help="Path to YOLO class specifications"
    )
    parser.add_argument(
        '--gpu_num', type=int, default=1,
        help='Number of GPU to use. Default is 1'
    )
    parser.add_argument(
        '--confidence', type=float, dest='score', default=0.7,
        help='Threshold for YOLO object confidence score to show predictions. Default is 0.25.'
    )
    parser.add_argument(
        '--box_file', type=str, dest='box', default="/home/maaz/Desktop/Visitor_Tracking/TrainingPipeline_TF2.0"
                                                    "/TrainYourOwnYOLO/Data/test/test_videos_results/output_box.txt",
        help="File to save bounding box results to."
    )

    parser.add_argument(
        '--postfix', type=str, dest='postfix', default='_catface',
        help='Specify the postfix for images with bounding boxes. Default is "_catface"'
    )

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    save_img = args.save_img
    file_types = args.file_types

    if file_types:
        input_paths = GetFileList(args.input_path, endings=file_types)
    else:
        input_paths = GetFileList(args.input_path)

    # Split images and videos
    img_endings = ('.jpg', '.jpg', '.png')
    vid_endings = ('.mp4', '.mpeg', '.mpg', '.avi')

    input_image_paths = []
    input_video_paths = []
    for item in input_paths:
        if item.endswith(img_endings):
            input_image_paths.append(item)
        elif item.endswith(vid_endings):
            input_video_paths.append(item)

    output_path = args.output
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # define YOLO detector
    yolo = YOLO(**{"model_path": args.model_path,
                   "anchors_path": args.anchors_path,
                   "classes_path": args.classes_path,
                   "score": args.score,
                   "gpu_num": args.gpu_num,
                   "model_image_size": (416, 416),
                   }
                )

    # Make a dataframe for the prediction outputs
    out_df = pd.DataFrame(
        columns=['image', 'image_path', 'xmin', 'ymin', 'xmax', 'ymax', 'label', 'confidence', 'x_size', 'y_size'])

    # labels to draw on images
    class_file = open(args.classes_path, 'r')
    input_labels = [line.rstrip('\n') for line in class_file.readlines()]
    print('Found {} input labels: {} ...'.format(len(input_labels), input_labels))

    if input_image_paths:
        print('Found {} input images: {} ...'.format(len(input_image_paths),
                                                     [os.path.basename(f) for f in input_image_paths[:5]]))
        start = timer()
        text_out = ''

        # This is for images
        for i, img_path in enumerate(input_image_paths):
            print(img_path)
            prediction, image = detect_object(yolo, img_path, save_img=save_img, save_img_path=args.output,
                                              postfix=args.postfix)
            y_size, x_size, _ = np.array(image).shape
            for single_prediction in prediction:
                out_df = out_df.append(pd.DataFrame([[os.path.basename(img_path.rstrip('\n')),
                                                      img_path.rstrip('\n')] + single_prediction + [x_size, y_size]],
                                                    columns=['image', 'image_path', 'xmin', 'ymin', 'xmax', 'ymax',
                                                             'label', 'confidence', 'x_size', 'y_size']))
        end = timer()
        print('Processed {} images in {:.1f}sec - {:.1f}FPS'.format(
            len(input_image_paths), end - start, len(input_image_paths) / (end - start)
        ))
        out_df.to_csv(args.box, index=False)

    # This is for videos
    if input_video_paths:
        print('Found {} input videos: {} ...'.format(len(input_video_paths),
                                                     [os.path.basename(f) for f in input_video_paths[:5]]))
        start = timer()
        for i, vid_path in enumerate(input_video_paths):
            output_path = os.path.join(args.output, os.path.basename(vid_path).replace('.', args.postfix + '.'))
            detect_video(yolo, vid_path, output_path=output_path)

        end = timer()
        print('Processed {} videos in {:.1f}sec'.format(
            len(input_video_paths), end - start
        ))
    # Close the current yolo session
    yolo.close_session()
