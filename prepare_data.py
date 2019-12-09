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
        "--video_path", type=str, default="/home/maaz/Desktop/Visitor_Tracking/TrainingPipeline_TF2.0"
                                          "/TrainYourOwnYOLO/Data/ch04_Person_1.avi",
        help="Absolute path to the video file"
    )
    parser.add_argument(
        "--annotation_txt_path", type=str, default="/home/maaz/Desktop/Visitor_Tracking/TrainingPipeline_TF2.0"
                                                   "/TrainYourOwnYOLO/Data/ch04_Person_1.txt",
        help="Absolute path to annotation .txt file"
    )
    parser.add_argument(
        "--output_txt_path", type=str, default="/home/maaz/Desktop/Visitor_Tracking/TrainingPipeline_TF2.0"
                                               "/TrainYourOwnYOLO/Data/ch04_Person_1_yolo.txt",
        help="Absolute path to the output .txt file")
    parser.add_argument(
        "--image_folder", type=str, default="/home/maaz/Desktop/Visitor_Tracking/TrainingPipeline_TF2.0"
                                            "/TrainYourOwnYOLO/Data/Images",
        help="Absolute path to output images folder"
    )

    return parser.parse_args()


def write_frames_from_vidoe(video_path, image_folder):
    # Create directory if not exists
    if not os.path.exists(image_folder):
        os.mkdir(image_folder)
    # Read the video and write frames to the image folder
    video = cv2.VideoCapture(video_path)
    ret, frame = video.read()
    frame_no = 0
    while ret:
        cv2.imwrite(image_folder + "/" + str(frame_no) + ".jpg", frame)
        frame_no += 1
        ret, frame = video.read()


def create_yolo_annotation_file(darklabel_txt_path, image_folder_path, yolo_txt_path):
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
    yolo_txt = open(yolo_txt_path, "w")

    with open(darklabel_txt_path) as darklabel_txt:
        for line in darklabel_txt:
            object_list, frame_no = ParseLine(line)
            image_path = image_folder_path + "/" + frame_no + ".jpg"
            # write annotations only if image exists
            if os.path.exists(image_path):
                yolo_txt.write(image_path + " ")
                # Hard coding the label 0 for person
                yolo_txt.write(" ".join([f"{o.x},{o.y},{int(o.x) + int(o.w)},{int(o.y) + int(o.h)},{0}"
                                         for o in object_list]))
                yolo_txt.write("\n")
    yolo_txt.close()


if __name__ == '__main__':
    # Parse the command line arguments
    args = parse_args()

    # Read the video and write the frames to images folder
    write_frames_from_vidoe(args.video_path, args.image_folder)

    # Read the darklable annotation file and create required formatted txt
    create_yolo_annotation_file(args.annotation_txt_path, args.image_folder, args.output_txt_path)
