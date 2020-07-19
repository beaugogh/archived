import torch, torchvision
print(torch.__version__)

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os
import cv2
import random
from enum import Enum
from google.colab.patches import cv2_imshow

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.engine.defaults import DefaultPredictor
from detectron2.structures.boxes import BoxMode
from detectron2.structures.instances import Instances
from detectron2.utils.logger import setup_logger

# import DensePose project
import sys, inspect 
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
dense_pose_path = os.path.join(parent_dir, 'projects/DensePose')
sys.path.insert(1, dense_pose_path)
import densepose

from densepose import add_densepose_config
from densepose.utils.logger import verbosity_to_level
from densepose.vis.base import CompoundVisualizer
from densepose.vis.bounding_box import ScoredBoundingBoxVisualizer
from densepose.vis.densepose import (
	DensePoseResultsContourVisualizer,
	DensePoseResultsFineSegmentationVisualizer,
	DensePoseResultsUVisualizer,
	DensePoseResultsVVisualizer,
)
from densepose.vis.extractor import CompoundExtractor, create_extractor

VISUALIZERS = {
	"dp_contour": DensePoseResultsContourVisualizer,
	"dp_segm": DensePoseResultsFineSegmentationVisualizer,
	"dp_u": DensePoseResultsUVisualizer,
	"dp_v": DensePoseResultsVVisualizer,
	"bbox": ScoredBoundingBoxVisualizer,
}

def prepare_predictor(threshold=0.5):
	print('prepare predictor...')
	cfg = get_cfg()
	add_densepose_config(cfg)
	model_file = './projects/DensePose/configs/densepose_rcnn_R_101_FPN_DL_s1x.yaml'
	model_weights = 'https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_101_FPN_DL_s1x/165712116/model_final_844d15.pkl'
	cfg.merge_from_file(model_file)	 
	cfg.MODEL.WEIGHTS = model_weights
	cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold 
	cfg.freeze()
	predictor = DefaultPredictor(cfg)
	# print(cfg.dump())
		
	return predictor
	
	
def prepare_context(vis_specs): 
	print('prepare context with specs: ', vis_specs)
	visualizers = []
	extractors = []
	for vis_spec in vis_specs:
		vis = VISUALIZERS[vis_spec]()
		visualizers.append(vis)
		extractor = create_extractor(vis)
		extractors.append(extractor)
	visualizer = CompoundVisualizer(visualizers)
	extractor = CompoundExtractor(extractors)
	context = {
		"extractor": extractor,
		"visualizer": visualizer
	}
	return context


def draw_image(input_img_path, output_img_path, predictor, context):
	input_img = read_image(input_img_path, format="BGR")
	outputs = predictor(input_img)["instances"]
	
	visualizer = context["visualizer"]
	extractor = context["extractor"]
	image = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
	image = np.tile(image[:, :, np.newaxis], [1, 1, 3])
	data = extractor(outputs)
	image_vis = visualizer.visualize(image, data)
	cv2.imwrite(output_img_path, image_vis)


# input: an original frame from video
# output: a frame image visualized by dense pose
def visualize_frame(frame, predictor, context):
	outputs = predictor(frame)["instances"]
	visualizer = context["visualizer"]
	extractor = context["extractor"]
	image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	image = np.tile(image[:, :, np.newaxis], [1, 1, 3])
	data = extractor(outputs)
	image_vis = visualizer.visualize(image, data)
	return image_vis

# generate dense-pose visualized frames based on a video input
def frame_generator(video, vis_specs):
	predictor = prepare_predictor()
	context = prepare_context(vis_specs)

	while video.isOpened():
		success, frame = video.read()
		if success:
			yield visualize_frame(frame, predictor, context)
		else:
			break

	video.release()
	cv2.destroyAllWindows()


def dp_video(video_name, vis_specs, parent_dir='data/vids/'):
	video_path = os.path.join(parent_dir, f'{video_name}.mp4')
	video = cv2.VideoCapture(video_path)
	width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
	size = (width, height)
	fps = video.get(cv2.CAP_PROP_FPS)
	num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
	basename = os.path.basename(video_path)
	print('size: ', size)
	print('fps: ', fps)
	print('#frames: ', num_frames)
	print('basename: ', basename)
	out_name = basename.split('.')[0] + '_' + vis_specs[0] + '.mp4'
	out_path = os.path.join(parent_dir, out_name)
	out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

	for i, frame in enumerate(frame_generator(video, vis_specs)):
		print(out_name, '{:.2%}...'.format(float(i) / num_frames))
		cv2.imwrite('data/out_frames/{:08d}.jpg'.format(i), frame)
		out.write(frame)

	out.release()
	print(f'video {out_path} generated')



def main():
	vis_specs_1 = ['dp_contour', 'bbox']
	vis_specs_2 = ['dp_segm', 'bbox']
	vis_specs_3 = ['dp_u', 'bbox']
	vis_specs_4 = ['dp_v', 'bbox']
	vis_specs_all = [vis_specs_2, vis_specs_3, vis_specs_4]
	video_name_all = [
		# 'bike_and_parkour',
		# 'break_dance_1',
		'break_dance_2',
		# 'kpop_dance_1',
		# 'kpop_dance_2',
		# 'jackie_chan_fight',
		# 'weili_fight',
		# 'ballet'
	]

	for video_name in video_name_all:
		for vis_specs in vis_specs_all:
			dp_video(video_name, vis_specs)
	



if __name__ == "__main__":
	main()
