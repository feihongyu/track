import cv2
import numpy
import torch

from feature.models import Resnet18FaceModel, Resnet50FaceModel


class Extractor():
    def __init__(self, weight="feature/weights/feature_net.pt", arch="resnet18"):
        self.arch = arch
        self.model = self.__init_model__(weight)
        self.infer_shape = (96, 128)

    def __init_model__(self, weight):
        model_class = Resnet18FaceModel
        if self.arch == "resnet50":
            model_class = Resnet50FaceModel

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        model = model_class(False).to(self.device)
        checkpoint = torch.load(weight, map_location=self.device)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        model.eval()

        return model

    def preprocess(self, img, resize=True):
        """
        图片预处理，生成特征提取网络输入的格式。
        图片维度：（C,H,W）
        输出维度：（1,C,H,W）
        :param img:
        :param resize:
        :return:
        """
        img0 = img.copy()
        if resize:
            h, w = img.shape[:2]
            img1 = cv2.resize(img, (0, 0), fx=self.infer_shape[0] / w, fy=self.infer_shape[1] / h,
                              interpolation=cv2.INTER_CUBIC)
            img1 = img1.astype(numpy.float32) / 255.
        else:
            img1 = img.astype(numpy.float32) / 255.
        img1 = torch.from_numpy(img1.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        return img0, img1

    def extract(self, imgs):
        imgs_buffer = []
        for img in imgs:
            _, img1 = self.preprocess(img)
            imgs_buffer.append(img1)

        imgs_buffer = torch.cat(imgs_buffer)

        logitses, features = self.model(imgs_buffer)

        return features
