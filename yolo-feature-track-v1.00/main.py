import argparse

from tracker import Tracker

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolov5_weight', type=str, default='yolov5/weights/yolov5l.pt', help='initial yolov5 weights path')
    parser.add_argument('--feature_net_weight', type=str, default='feature/weights/feature_net.pt', help='feature net weights path')
    parser.add_argument('--feature_net_arch', type=str, default='resnet50', help='feature net arch')
    parser.add_argument('--input_path', type=str, default='videos/input/human2.mp4', help='input video')
    parser.add_argument('--output_path', type=str, default='videos/output', help='output path')
    parser.add_argument('--target_img_path', type=str, default='images/human2.png', help='target image path')

    args = parser.parse_args()
    # args.feature_net_weight = "feature/weights/feature_net_resnet50.pt"
    # args.feature_net_arch = "resnet50"

    tracker = Tracker(args)
    tracker.track()
