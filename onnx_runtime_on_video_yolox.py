import argparse
import os
import cv2
import numpy as np
import onnxruntime
from yolox.data.data_augment import preproc as preprocess
from yolox.data.datasets import COCO_CLASSES
from yolox.utils import mkdir, multiclass_nms, demo_postprocess, vis


def make_parser():
    parser = argparse.ArgumentParser("onnxruntime inference sample")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="yolox_head_depth.onnx",
        help="Input your onnx model.",
    )
    parser.add_argument(
        "-i",
        "--image_path",
        type=str,
        default='selfie.jpg',
        help="Path to your input image.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default='demo_output',
        help="Path to your output directory.",
    )
    parser.add_argument(
        "-s",
        "--score_thr",
        type=float,
        default=0.5,
        help="Score threshould to filter the result.",
    )
    parser.add_argument(
        "--input_shape",
        type=str,
        default="640,640",
        help="Specify an input shape for inference.",
    )
    parser.add_argument(
        "--with_p6",
        action="store_true",
        help="Whether your model uses p6 in FPN/PAN.",
    )
    return parser


if __name__ == '__main__':
    args = make_parser().parse_args()
    import  time
    input_shape = tuple(map(int, args.input_shape.split(',')))
    session = onnxruntime.InferenceSession(args.model)

    cap = cv2.VideoCapture('/home/gaurav/Projects/video/v.mp4')
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float

    FPS = 0
    while True:
        ret_val, origin_img = cap.read()
        if ret_val:
            t0 = time.time()
            img, ratio = preprocess(origin_img, input_shape)
            ort_inputs = {session.get_inputs()[0].name: img[None, :, :, :]}
            output = session.run(None, ort_inputs)
            predictions = demo_postprocess(output[0])[0]

            boxes = predictions[:, :4]
            scores = predictions[:, 4:5] * predictions[:, 5:]

            boxes_xyxy = np.ones_like(boxes)
            boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
            boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
            boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
            boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
            boxes_xyxy /= ratio
            dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)
            if dets is not None:
                final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
                origin_img = vis(origin_img, final_boxes, final_scores, final_cls_inds,conf=args.score_thr, class_names=COCO_CLASSES)
            FPS = (FPS + (1. / (time.time() - t0))) / 2
            result_frame = cv2.putText(origin_img, "FPS=%.2f" % (FPS), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                       (0, 255, 0), 2)
        cv2.namedWindow("yolox", cv2.WINDOW_NORMAL)
        cv2.imshow("yolox", result_frame)
        ch = cv2.waitKey(1)

