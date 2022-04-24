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

        self.extractor = Extractor(weight=args.feature_net_weight, arch=args.feature_net_arch)
        self.feature_net_arch = args.feature_net_arch

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
        # 正样本缓冲区
        self.positive_feature_buffer = None
        # 原始正样本特征， 优先级高于缓冲区正样本所有正样本
        self.positive_feature0 = None

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
        :return: 原始正样本距离，正样本缓冲区加权距离
        """

        ###############################原始正样本的距离###########################################
        distance0 = features - self.positive_feature0
        distance0 = distance0 ** 2
        distance0 = numpy.sum(distance0, axis=1)

        ###############################正样本缓冲区的加权距离###########################################
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
            average_distance_with_weights = numpy.sum(distances, axis=0) / len(self.positive_feature_buffer)
        else:
            count = 0
            for i in range(1, len(self.positive_feature_buffer) + 1):
                count += i ** coefficient
            average_distance_with_weights = numpy.sum(distances, axis=0) / count

        return distance0, average_distance_with_weights

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
        current_target_img_feature = target_img_feature0

        # 正样本缓冲区
        self.positive_feature_buffer = current_target_img_feature
        # 原始正样本
        self.positive_feature0 = target_img_feature0

        while cap.isOpened() and ret == True and current_frame <= ending_frame:
            tic = time.time()

            canvas = frame
            canvas, boxes = self.detector.detect(canvas)

            sub_canvases = []

            for i, box in enumerate(boxes):
                sub_canvas = canvas[box[1]:box[3], box[0]:box[2], :]
                cv2.imwrite("images/" + str(i) + ".jpg", sub_canvas)
                sub_canvases.append(sub_canvas)

            """
                这里有几个阈值：
                1.原始正样本追踪阈值阈值 = 2.原始正样本存储阈值
                3.正样本缓冲区追踪阈值 > 4.正样本缓冲区存储阈值
                
            """

            if self.feature_net_arch == "resnet18":
                threshold1 = 0.5
                threshold3 = 0.7
                threshold4 = 0.3
            else:
                threshold1 = 0.5
                threshold3 = 0.6
                threshold4 = 0.3

            # 检测到目标的时候才进行计算特征距离
            if len(sub_canvases) > 0:
                features = self.extractor.extract(sub_canvases).cpu().detach().numpy()
                # distances_list (原始正样本距离，正样本缓冲区加权距离)
                distances_list = self.compute_distance(features)

                positive_feature0_nearest_distance = numpy.min(distances_list[0])
                positive_feature0_offset = numpy.argmin(distances_list[0], axis=0)

                # 原始正样本距离小于0.4，视为直接匹配成功，追踪最小距离目标并把该目标的特征加入缓冲区
                if positive_feature0_nearest_distance < threshold1:
                    current_target_img_feature = features[positive_feature0_offset][None]
                    self.positive_feature_buffer = numpy.concatenate(
                        (self.positive_feature_buffer, current_target_img_feature), axis=0)
                    if len(self.positive_feature_buffer) > self.positive_feature_buffer_maximum:
                        self.positive_feature_buffer = self.positive_feature_buffer[
                                                       1: self.positive_feature_buffer_maximum + 1]
                    plot_one_box(boxes[positive_feature0_offset], canvas, color=[0, 0, 255])

                # 原始正样本距离大于0.4，需要计算正样本缓冲区的距离
                else:
                    positive_feature_buffer_nearest_distance = numpy.min(distances_list[1])
                    positive_feature_buffer_offset = numpy.argmin(distances_list[1], axis=0)

                    # 正样本缓冲区的距离小于0.5时，追踪最小距离目标
                    if positive_feature_buffer_nearest_distance < threshold3:
                        current_target_img_feature = features[positive_feature_buffer_offset][None]
                        # 正样本缓冲区的距离小于0.3时，把该目标的特征加入缓冲区
                        if positive_feature_buffer_nearest_distance < threshold4:
                            self.positive_feature_buffer = numpy.concatenate(
                                (self.positive_feature_buffer, current_target_img_feature), axis=0)
                            if len(self.positive_feature_buffer) > self.positive_feature_buffer_maximum:
                                self.positive_feature_buffer = self.positive_feature_buffer[
                                                               1: self.positive_feature_buffer_maximum + 1]
                        plot_one_box(boxes[positive_feature_buffer_offset], canvas, color=[0, 0, 255])

            out.write(canvas)

            toc = time.time()

            ret, frame = cap.read()
            current_frame += 1

            print("完成{:.2%}  {}s  toc - tic = {:.2}s".format(current_frame / ending_frame, int(current_frame / input_fps),
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
