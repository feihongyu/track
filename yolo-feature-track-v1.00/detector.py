import cv2
import numpy
import torch

from yolov5.models.experimental import attempt_load
from yolov5.utils.augmentations import letterbox
from yolov5.utils.general import non_max_suppression, scale_coords
from yolov5.utils.plots import plot_images, plot_one_box, save_one_box
from yolov5.utils.torch_utils import select_device

TARGET = ["person"]


class Detector():
    def __init__(self, weight="yolov5/weights/yolov5l.pt", threshold=0.3, persist=False):
        self._init_model__(weight)
        self.threshold = threshold
        self.img_size = 640
        self.persist = persist
        self.output_path = r"yolov5/output/"

    def _init_model__(self, weight):
        self.weights = weight
        self.device = '0' if torch.cuda.is_available() else 'cpu'
        self.device = select_device(self.device)
        model = attempt_load(weight, map_location=self.device)
        model.to(self.device).eval()
        model.half()
        self.model = model
        self.names = model.module.names if hasattr(model, 'module') else model.names

    def preprocess(self, img):
        """
        图片预处理，生成yolov5输入的格式。
        图片大小：（640×640）
        图片维度：（C,H,W）
        输出维度：（1,C,H,W）
        :param img:
        :return:
        """
        img0 = img.copy()
        img = letterbox(img, new_shape=self.img_size)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = numpy.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img0, img

    def detect(self, img):
        """
        目标检测，采用yolov5的检测模式。
        输出原图坐标系的预测框。
        :param img:
        :return:
        """
        img0, img1 = self.preprocess(img)
        pred = self.model(img1, augment=False)[0]
        pred = pred.float()
        pred = non_max_suppression(pred, self.threshold, 0.4)
        pred_boxes = []
        for det in pred:
            if det is not None and len(det):
                det[:, :4] = scale_coords(img1.shape[2:], det[:, :4], img0.shape).round()
                for *x, conf, cls_id in det:
                    lbl = self.names[int(cls_id)]
                    if not lbl in TARGET:
                        continue
                    x1, y1 = int(x[0]), int(x[1])
                    x2, y2 = int(x[2]), int(x[3])
                    pred_boxes.append((x1, y1, x2, y2, lbl, conf))

        return img, pred_boxes


if __name__ == '__main__':
    d = Detector(weight="weights/yolov5m.pt", persist=True)
    img = cv2.imread(r"data/images/zidane.jpg")
    d.detect(img)
