# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm
from enum import Enum

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import ColorMode
from predictor import VisualizationDemo

# include the PointRend project
import sys, inspect 
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
point_rend_path = os.path.join(parent_dir, 'projects/PointRend')
sys.path.insert(1, point_rend_path)
import point_rend


class DetectionMode(Enum):
    InstanceSegmentation = 'InstanceSegmentation' # instance segmentation
    Keypoints = 'Keypoints' # key points
    PanopticSegmentation = 'PanopticSegmentation' # panoptic segmentation
    PointRend = 'PointRend' # point-rend is a sharper version of instance-segmentation

# constants
# MODE = DetectionMode.PanopticSegmentation
# MODE = DetectionMode.InstanceSegmentation
# MODE = DetectionMode.PointRend
MODE = DetectionMode.Keypoints

# 'bike_and_parkour'
# 'break_dance_1'
# 'break_dance_2'
# 'cat_island'
# 'cats_play_piano'
# 'cheetah_chase'
# 'jackie_chan_fight'
# 'kpop_dance_1'
# 'kpop_dance_2'
# 'weili_fight'
# 'shanghai_drive'
# 'ballet'

vname = 'ballet'
tail = 'pan'
if MODE == DetectionMode.Keypoints:
    tail = 'key'
elif MODE == DetectionMode.InstanceSegmentation:
    tail = 'inst'
elif MODE == DetectionMode.PointRend:
    tail = 'inst_p'

WINDOW_NAME = "COCO detections"
VIDEO_INPUT_DEFAULT = f'data/vids/{vname}.mp4'
OUTPUT_DEFAULT = f'data/vids/{vname}_{tail}.mkv'
CONFIDENCE_THRESHOLD_DEFAULT = 0.6
COLOR_MODE = ColorMode.IMAGE_BW
SAVE_FRAMES = True



def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin models")
    parser.add_argument(
        "--config-file",
        default='',
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument(
        "--video-input", 
        default=VIDEO_INPUT_DEFAULT,
        help="Path to video file."
    )
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        default=OUTPUT_DEFAULT,
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=CONFIDENCE_THRESHOLD_DEFAULT,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()

    mode = MODE
    model = None
    model_file = None
    model_weights = None
    if mode == DetectionMode.InstanceSegmentation:
        model = 'COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml'
        model_file = model_zoo.get_config_file(model)
        model_weights = model_zoo.get_checkpoint_url(model)
    elif mode == DetectionMode.Keypoints:
        model = 'COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml'
        model_file = model_zoo.get_config_file(model)
        model_weights = model_zoo.get_checkpoint_url(model)
    elif mode == DetectionMode.PanopticSegmentation:
        model = 'COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml'
        model_file = model_zoo.get_config_file(model)
        model_weights = model_zoo.get_checkpoint_url(model)
    elif mode == DetectionMode.PointRend:
        point_rend.add_pointrend_config(cfg)
        model_file = os.path.join(point_rend_path, 'configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco.yaml') 
        model_weights = 'https://dl.fbaipublicfiles.com/detectron2/PointRend/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco/164955410/model_final_3c3198.pkl'
        
    cfg.merge_from_file(model_file)     
    cfg.MODEL.WEIGHTS = model_weights

    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg




if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    print(args)
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg, instance_mode=COLOR_MODE)

    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation
            img = read_image(path, format="BGR")
            start_time = time.time()
            predictions, visualized_output = demo.run_on_image(img)
            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    "detected {} instances".format(len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )

            if args.output:
                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    out_filename = os.path.join(args.output, os.path.basename(path))
                else:
                    assert len(args.input) == 1, "Please specify a directory with args.output"
                    out_filename = args.output
                visualized_output.save(out_filename)
            else:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                if cv2.waitKey(0) == 27:
                    break  # esc to quit
    elif args.webcam:
        assert args.input is None, "Cannot have both --input and --webcam!"
        cam = cv2.VideoCapture(0)
        for vis in tqdm.tqdm(demo.run_on_video(cam)):
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, vis)
            if cv2.waitKey(1) == 27:
                break  # esc to quit
        cv2.destroyAllWindows()
    elif args.video_input:
        video = cv2.VideoCapture(args.video_input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(args.video_input)

        if args.output:
            if os.path.isdir(args.output):
                output_fname = os.path.join(args.output, basename)
                output_fname = os.path.splitext(output_fname)[0] + ".mkv"
            else:
                output_fname = args.output
            assert not os.path.isfile(output_fname), output_fname
            output_file = cv2.VideoWriter(
                filename=output_fname,
                # some installation of opencv may not support x264 (due to its license),
                # you can try other format (e.g. MPEG)
                fourcc=cv2.VideoWriter_fourcc(*"DIVX"),
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )
        assert os.path.isfile(args.video_input)
        ind = 0
        for vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
            if args.output:
                if SAVE_FRAMES:
                    cv2.imwrite('data/out_frames/vis_frame_{:08d}.jpg'.format(ind), vis_frame)
                ind += 1
                output_file.write(vis_frame)
            else:
                cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                cv2.imshow(basename, vis_frame)
                if cv2.waitKey(1) == 27:
                    break  # esc to quit
        video.release()
        if args.output:
            output_file.release()
        else:
            cv2.destroyAllWindows()
