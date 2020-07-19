# create a video with masks
import cv2
import numpy as np
from visualize_cv2 import model, display_instances, class_names

vname = 'cat_v.mov'

capture = cv2.VideoCapture(vname)
size = (
    int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
    int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
)
codec = cv2.VideoWriter_fourcc(*'DIVX')
output = cv2.VideoWriter(vname.split('.')[0] + '_masked.avi', codec, 30, size)

i = 0
while(capture.isOpened()):
    ret, frame = capture.read()
    i += 1
    print(i)

    if ret:
        # add mask to frame
        results = model.detect([frame], verbose=0)
        r = results[0]
        frame = display_instances(
            frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores']
        )
        output.write(frame)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

capture.release()
output.release()
cv2.destroyAllWindows()
