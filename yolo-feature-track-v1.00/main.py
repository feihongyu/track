import argparse

from tracker import Tracker

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolov5_weight', type=str, default='yolov5/weights/yolov5l.pt',
                        help='initial yolov5 weights path')
    parser.add_argument('--feature_net_weight', type=str, default='feature/weights/feature_net_resnet18.pt',
                        help='feature net weights path')
    parser.add_argument('--feature_net_arch', type=str, default='resnet18', help='feature net arch')
    parser.add_argument('--input_path', type=str, default='videos/input/human6.mp4', help='input video')
    parser.add_argument('--output_path', type=str, default='videos/output', help='output path')
    parser.add_argument('--target_img_path', type=str, default='images/human6.png', help='target image path')
    parser.add_argument('--threshold1', type=float, default=0.4)
    parser.add_argument('--threshold2', type=float, default=0.4)
    parser.add_argument('--threshold3', type=float, default=0.65)
    parser.add_argument('--threshold4', type=float, default=0.2)
    
    args = parser.parse_args()

    args.feature_net_weight = "feature/weights/feature_net_resnet50.pt"
    args.feature_net_arch = "resnet50"
    # args.feature_net_weight = "feature/weights/0.03,resnet18.pth.tar"
    args.feature_net_arch = "resnet50"
    tracker = Tracker(args)
    tracker.track()

    # for i in range(8):
    #     args.feature_net_weight = "feature/weights/feature_net_resnet50.pt"
    #     args.feature_net_arch = "resnet50"
    #     args.input_path = "videos/input/human" + str(i) + ".mp4"
    #     args.target_img_path = "images/human" + str(i) + ".png"
    #     tracker = Tracker(args)
    #     tracker.track()
