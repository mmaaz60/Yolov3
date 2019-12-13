""" The script converts the darklabel video annotation to yolo format.
Darklable Format <frame_no>,<no. of objects>,<x>,<y>,<w>,<h>,<label>,<x>,<y>,<w>,<h>,<label>, ...:
Yolo Format: <abs path to image>  <xmin>,<ymin>,<xmax>,<ymax>,<label>,<xmin>,<ymin>,<xmax>,<ymax>,<label>, ... """

import argparse
import cv2
import os


class Object(object):
    def __init__(self, x, y, w, h, label):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.label = label


def parse_args():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)

    parser.add_argument(
        "--video_folder_path", type=str, default="/home/maaz/Desktop/Visitor_Tracking/TrainingPipeline_TF2.0"
                                                 "/TrainYourOwnYOLO/Data/training_videos",
        help="Absolute path to the video file"
    )
    parser.add_argument(
        "--annotation_txt_folder_path", type=str, default="/home/maaz/Desktop/Visitor_Tracking/TrainingPipeline_TF2.0"
                                                          "/TrainYourOwnYOLO/Data/training_annotations",
        help="Absolute path to annotation .txt file"
    )
    parser.add_argument(
        "--output_txt_path", type=str, default="/home/maaz/Desktop/Visitor_Tracking/TrainingPipeline_TF2.0"
                                               "/TrainYourOwnYOLO/Data/ch04_yolo.txt",
        help="Absolute path to the output .txt file")
    parser.add_argument(
        "--image_folder", type=str, default="/home/maaz/Desktop/Visitor_Tracking/TrainingPipeline_TF2.0"
                                            "/TrainYourOwnYOLO/Data/Images",
        help="Absolute path to output images folder"
    )

    return parser.parse_args()


def write_frames_from_vidoe(video_path, image_folder, start_frame_no):
    # Create directory if not exists
    if not os.path.exists(image_folder):
        os.mkdir(image_folder)
    # Read the video and write frames to the image folder
    video = cv2.VideoCapture(video_path)
    ret, frame = video.read()
    frame_no = start_frame_no
    while ret:
        cv2.imwrite(image_folder + "/" + str(frame_no) + ".jpg", frame)
        frame_no += 1
        ret, frame = video.read()

    return frame_no


def create_yolo_annotation_file(darklabel_txt_path, image_folder_path, yolo_txt_path, start, end):
    def ParseLine(line):
        records = line.split(",")
        frame_no = records[0]
        no_of_objects = records[1]
        records = records[2:]

        object_list = []
        for i in range(0, int(len(records) / 5)):
            object_list.append(Object(records[i * 5 + 0], records[i * 5 + 1], records[i * 5 + 2],
                                      records[i * 5 + 3], records[i * 5 + 4]))
        return object_list, frame_no

    # Open the output .txt file
    if os.path.exists(yolo_txt_path):
        append_write = 'a'  # append if already exists
    else:
        append_write = 'x'  # make a new file if not
    yolo_txt = open(yolo_txt_path, append_write)

    previous = start
    with open(darklabel_txt_path) as darklabel_txt:
        for line in darklabel_txt:
            object_list, frame_no = ParseLine(line)

            # Empty frame annotations (In the start and in between non-empty frames)
            if (int(frame_no) + start > previous + 1):
                for i in range(previous + 1, int(frame_no) + start):
                    image_path = image_folder_path + "/" + str(i) + ".jpg"
                    if os.path.exists(image_path):
                        yolo_txt.write(image_path + "\n")

            # Non empty frame annotation
            image_path = image_folder_path + "/" + str(int(frame_no) + start) + ".jpg"
            # write annotations only if image exists
            if os.path.exists(image_path):
                yolo_txt.write(image_path + " ")
                # Hard coding the label 0 for person. For general implementation you can have .names file with class
                # names and parse label value from it.
                yolo_txt.write(" ".join([f"{o.x},{o.y},{int(o.x) + int(o.w)},{int(o.y) + int(o.h)},{0}"
                                         for o in object_list]))
                yolo_txt.write("\n")

            previous = int(frame_no) + start

        # Empty frames annotations in the end of file
        if previous != end:
            for i in range(previous + 1, end + 1):
                image_path = image_folder_path + "/" + str(i) + ".jpg"
                if os.path.exists(image_path):
                    yolo_txt.write(image_path + "\n")

    yolo_txt.close()


if __name__ == '__main__':
    # Parse the command line arguments
    args = parse_args()

    video_paths = sorted(os.listdir(args.video_folder_path))
    annotation_txt_paths = sorted(os.listdir(args.annotation_txt_folder_path))

    start = 0
    end = 0
    for (video_path, annotation_txt_path) in zip(video_paths, annotation_txt_paths):
        end = write_frames_from_vidoe(args.video_folder_path + "/" + video_path, args.image_folder, start)
        create_yolo_annotation_file(args.annotation_txt_folder_path + "/" + annotation_txt_path, args.image_folder,
                                    args.output_txt_path, start, end)
        start = end
