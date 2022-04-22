import os
import time
from queue import Queue

import cv2
import numpy

from detector import Detector
from extractor import Extractor
from yolov5.utils.plots import plot_one_box


class Tracker():
    def __init__(self, args):
        self.detector = Detector(weight=args.yolov5_weight, persist=True)
        self.extractor = Extractor(weight=args.feature_net_weight)

        self.input_path = os.path.join(args.input_path)
        filename = os.path.basename(self.input_path)
        self.output_path = os.path.join(args.output_path, filename)

        self.video_type_dict = {
            ".mp4": "mp4v",
            ".avi": "pimi",
            ".ogv": "theo",
            ".fly": "flv1"
        }

        self.target_img_path = args.target_img_path
        self.positive_feature_buffer_maximum = 100
        # self.positive_feature_buffer = numpy.zeros(
        #     (self.positive_feature_buffer_maximum, 2048 if args.__contains__("feature-net-arch") else 512))
        self.positive_feature_buffer = None

    def compute_distance(self, features, type=0):
        """
        计算检测目标和正样本缓冲区的所有样本的加权平均距离
        0.权重平均模式：
            平均计算权重
        1.权重递增模式：
            该模式侧重于平滑镜头目标的关联性，但可能会导致检测一旦失误，后续会逐渐将误差放大。
        2.权重递减模式：
            该模式侧重于镜头突变目标的关联性，但对于平滑镜头中目标的关联性不如1模式
        计算距离中原始图片特征规定，一定会用作距离计算。
        :param features: 目标检测特征
        :param type: 权重模式
        :return: 加权距离
        """

        if type == 0:
            weights_matrix = [[1] for weight in range(1, len(self.positive_feature_buffer) + 1)]
        elif type == 1:
            weights_matrix = [[weight] for weight in range(1, len(self.positive_feature_buffer) + 1)]
        else:
            weights_matrix = [[weight] for weight in range(len(self.positive_feature_buffer), 0, -1)]

        # 权值矩阵
        coefficient = 1
        weights_matrix = numpy.array(weights_matrix) ** coefficient

        distances = []
        for positive_feature in self.positive_feature_buffer:
            distance = features - positive_feature
            distance = distance ** 2
            distance = numpy.sum(distance, axis=1)
            distances.append(distance[None])
        distances = numpy.concatenate(distances, axis=0)
        distances = distances * weights_matrix

        if type == 0:
            average_distance_without_weights = numpy.sum(distances, axis=0) / len(self.positive_feature_buffer)
        else:
            count = 0
            for i in range(1, len(self.positive_feature_buffer) + 1):
                count += i ** coefficient
            average_distance_without_weights = numpy.sum(distances, axis=0) / count

        return average_distance_without_weights

    def comput_iou(self):
        pass

    def track(self):
        cap = cv2.VideoCapture(self.input_path)

        input_fps = cap.get(cv2.CAP_PROP_FPS)
        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        ret, frame = cap.read()

        ending_frame = 100
        ending_frame = video_length

        output_fps = input_fps / 1
        video_type = os.path.splitext(self.output_path)[-1]
        fourcc = cv2.VideoWriter_fourcc(*self.video_type_dict.get(video_type))
        out = cv2.VideoWriter(self.output_path, fourcc, output_fps, (frame.shape[1], frame.shape[0]))

        current_frame = 0

        # 原始目标的图片、特征
        target_img = cv2.imread(self.target_img_path)
        target_img_feature0 = self.extractor.extract([target_img, ]).cpu().detach().numpy()
        # target_img_feature0 = self.extractor.extract([target_img, ])
        current_target_img_feature = target_img_feature0

        # 正样本特征缓存
        # self.positive_feature_buffer[0] = current_target_img_feature
        self.positive_feature_buffer = current_target_img_feature

        while cap.isOpened() and ret == True and current_frame <= ending_frame:
            tic = time.time()

            canvas = frame
            canvas, boxes = self.detector.detect(canvas)

            sub_canvases = []

            if current_frame >= 92 * input_fps:
                print()

            for i, box in enumerate(boxes):
                sub_canvas = canvas[box[1]:box[3], box[0]:box[2], :]
                cv2.imwrite("images/" + str(i) + ".jpg", sub_canvas)
                sub_canvases.append(sub_canvas)

            # 检测到目标的时候才能进行计算特征距离
            if len(sub_canvases) > 0:
                features = self.extractor.extract(sub_canvases).cpu().detach().numpy()
                distances = self.compute_distance(features)

                nearest_distance = numpy.min(distances)
                off = numpy.argmin(distances, axis=0)

                if nearest_distance < 0.6:
                    current_target_img_feature = features[off][None]
                    self.positive_feature_buffer = numpy.concatenate(
                        (self.positive_feature_buffer, current_target_img_feature), axis=0)
                    if len(self.positive_feature_buffer) > self.positive_feature_buffer_maximum:
                        self.positive_feature_buffer = self.positive_feature_buffer[
                                                       1: self.positive_feature_buffer_maximum + 1]
                    plot_one_box(boxes[off], canvas, color=[0, 0, 255])
            else:
                pass

            # cv2.putText(canvas, "FPS:%f" %(1. / (toc-tic)), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            out.write(canvas)

            toc = time.time()

            ret, frame = cap.read()
            current_frame += 1

            print("完成{:.2%}  {:.2}s  toc - tic = {:.2}s".format(current_frame / ending_frame, current_frame / input_fps,
                                                                toc - tic))

        cap.release()
        out.release()
        print()


class Feature_queue():
    def __init__(self, feature):
        self.feature0 = feature
        self.feature_queue = Queue()

    def put(self, feature):
        self.feature_queue.put(feature)

    def get(self):
        return self.feature_queue.get()
