from absl import flags
import sys
FLAGS = flags.FLAGS
FLAGS(sys.argv)
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

# ADD THE PATH TO THE FOLDERS ACCORDING TO YOURS
from absl import app, flags, logging
from absl.flags import FLAGS
import numpy as np

from DeepSortWithYolov3.yolov3_tf2.models import YoloV3, YoloV3Tiny
from DeepSortWithYolov3.yolov3_tf2.utils import load_darknet_weights

# ADD THE PATH TO THE YOLOV3.WEIGHTS FILE
 # ADD THE SAME PATH WITH YOLOV3.TF WHICH WILL BE DOWNLOADED AUTOMATICALLY IN THE FOLDER WHEN THE CODE WILL RUN
flags.DEFINE_string('weights', 'D:\PycharmProjects\opencv\DeepSortWithYolov3\weights\yolov3.weights', 'path to weights file')
flags.DEFINE_string('output', 'D:\PycharmProjects\opencv\DeepSortWithYolov3\weights\yolov3.tf', 'path to output')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')

# def main(_argv):
#     if FLAGS.tiny:
#         yolo = YoloV3Tiny(classes=FLAGS.num_classes)
#     else:
#         yolo = YoloV3(classes=FLAGS.num_classes)
#     yolo.summary()
#     logging.info('model created')
#     load_darknet_weights(yolo, FLAGS.weights, FLAGS.tiny)
#     logging.info('weights loaded')
#     img = np.random.random((1, 320, 320, 3)).astype(np.float32)
#     output = yolo(img)
#     logging.info('sanity check passed')
#     yolo.save_weights(FLAGS.output)
#     logging.info('weights saved')
#
# if __name__ == '__main__':
#     try:
#         app.run(main)
#     except SystemExit:
#         pass
from DeepSortWithYolov3.yolov3_tf2.models import YoloV3
from DeepSortWithYolov3.yolov3_tf2.dataset import transform_images
from DeepSortWithYolov3.yolov3_tf2.utils import convert_boxes
from DeepSortWithYolov3.deep_sort import preprocessing
from DeepSortWithYolov3.deep_sort import nn_matching
from DeepSortWithYolov3.deep_sort.detection import Detection
from DeepSortWithYolov3.deep_sort.tracker import Tracker
from DeepSortWithYolov3.tools import generate_detections as gdet


# ADD THE PATH TO THE COCO.NAMES FILE
class_names = [c.strip() for c in open('D:\PycharmProjects\opencv\DeepSortWithYolov3\data\labels\coco.names').readlines()]
yolo = YoloV3(classes=len(class_names))
# ADD THE PATH TO THE YOLOV3.TF.INDEX IN WEIGHTS FILE (omit the .index from the name)
yolo.load_weights('D:\PycharmProjects\opencv\DeepSortWithYolov3\weights\yolov3.tf')
max_cosine_distance = 0.5
nn_budget = None
nms_max_overlap = 0.8
# ADD PATH TO THE MARS-SMALL128.PB FILE FROM MODEL_DATA FOLDER
model_filename = 'D:\PycharmProjects\opencv\DeepSortWithYolov3\model_data\mars-small128.pb'
encoder = gdet.create_box_encoder(model_filename, batch_size=1)
metric = nn_matching.NearestNeighborDistanceMetric('cosine', max_cosine_distance, nn_budget)
tracker = Tracker(metric)

vid = cv2.VideoCapture('D:\PycharmProjects\opencv\production ID_3800731.mp4') # ADD THE VIDEO FILE
# vid = cv2.VideoCapture(0) # FOR CAMERA
codec = cv2.VideoWriter_fourcc(*'mp4v')
vid_fps = int(vid.get(cv2.CAP_PROP_FPS))
vid_width,vid_height = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
# ADD PATH WHERE YOU WANT TO SAVE THE OUTPUT VIDEO
out = cv2.VideoWriter('D:\PycharmProjects\opencv\DeepSortWithYolov3\data\video\results.mp4', codec, vid_fps, (vid_width, vid_height))

from _collections import deque
pts = [deque(maxlen=30) for _ in range(1000)]
line_cross_set = set()

'''CHANGE Y1 AND Y2 ACCORDING TO YOUR VIDEO OR CAMERA POSITION'''
'''-----------------------------------------------------------'''

y1 = 450  # LEFT CORNER HIEGHT OF THE LINE
y2 = 450      # RIGHT CORNER HIEGHT OF THE LINE

def getCountLineCrossed(width, pointList):
    a = (0, y1)
    b = (width, y2)
    position = ((b[0] - a[0]) * (pointList[0][1] - a[1]) - (b[1] - a[1]) * (pointList[0][0] - a[0]))
    prevposition = ((b[0] - a[0]) * (pointList[0 - 5][1] - a[1]) - (b[1] - a[1]) * (pointList[0 - 5][0] - a[0]))

    if prevposition != 0 and position != 0:
        if position > 0 and prevposition < 0:
            print("crossed to right")
            return "right"
        if position < 0 and prevposition > 0:
            print("crossed to left")
            return "left"
    return None

def getDirection(image, pointList):
    dx = 0
    dy = 0
    for x in range(len(pointList)-1):
        cv2.line(image, pointList[x], pointList[x+1], [0, 0, 255], 3)
        dx += pointList[x+1][0] - pointList[x][0]
        dy += pointList[x+1][1] - pointList[x][1]
    x = ""
    y = ""
    if(dx < 0):
        x = "right"
    if(dx > 0):
        x = "left"
    if(dy < 0):
        y = "down"
    if(dy > 0):
        y = "up"
    return (x,y)

left = 0
right = 0
up = 0
down = 0
prevPeopleCount = 0
totalPeopleCount = 0
totalLineCrossedLeft = 0
totalLineCrossedRight = 0
totalLineCrossed = 0
lineCrossingIDs = []
peoplecount = 0
frames = 0

while True:
    _, img = vid.read()
    if img is None:
        print('Completed')
        break
    if not _:
        break

    frames += 1
    height, width, _ = img.shape

    img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_in = tf.expand_dims(img_in, 0)
    img_in = transform_images(img_in, 416)
    t1 = time.time()
    boxes, scores, classes, nums = yolo.predict(img_in)
    classes = classes[0]
    names = []
    for i in range(len(classes)):
        names.append(class_names[int(classes[i])])
    names = np.array(names)

    converted_boxes = convert_boxes(img, boxes[0])
    features = encoder(img, converted_boxes)

    detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
                  zip(converted_boxes, scores[0], names, features)]

    boxs = np.array([d.tlwh for d in detections])
    scores = np.array([d.confidence for d in detections])
    classes = np.array([d.class_name for d in detections])
    indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)

    detections = [detections[i] for i in indices]

    tracker.predict()
    tracker.update(detections)

    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        bbox = track.to_tlbr()
        class_name = track.get_class()
        color = colors[int(track.track_id) % len(colors)]
        color = [i * 255 for i in color]

        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1] - 30)), (int(bbox[0]) + (len(class_name)+ len(str(track.track_id))) * 17,
                                                               int(bbox[1])), color, -1)

        center = (int(((bbox[0]) + (bbox[2])) / 2), int(((bbox[1]) + (bbox[3])) / 2))
        pts[track.track_id].append(center)
        xdir = ""
        ydir = ""
        Id=int(track.track_id)
        if len(pts[track.track_id]) > 6:
            xdir, ydir = getDirection(img, pts[track.track_id])
            if (xdir == "left"):
                left += 1
            if (xdir == "right"):
                right += 1
            if (ydir == "up"):
                up += 1
            if (ydir == "down"):
                down += 1

            if (frames % 10 == 0):
                lineCrossed = getCountLineCrossed(width,pts[track.track_id])
                if lineCrossed != None:
                    if Id not in lineCrossingIDs:
                        if lineCrossed == "left":
                            totalLineCrossedLeft += 1
                        elif lineCrossed == "right":
                            totalLineCrossedRight += 1
                        totalLineCrossed += 1
                        lineCrossingIDs.append(Id)
                        line_cross_set.add(int(Id))
                else:
                    if Id in lineCrossingIDs:
                        lineCrossingIDs.remove(Id)
        line_cross_count=len(line_cross_set)

        '''TO MODIFY FONT, FONT SCALE, COLOR AND POSITION: EDIT THE BELOW PARAMETERS'''
        '''-------------------------------------------------------------------------'''
        font_style = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, class_name + "-" + str(track.track_id) , (int(bbox[0]), int(bbox[1] - 10)), 0, 0.75, (255, 255, 255), 2)
        cv2.line(img, (0, y1), (650, y2), [255, 255, 255], 3)
        cv2.line(img, (950, y1), (2000, y2), [255, 255, 255], 3)
        cv2.circle(img, center, 3, (0, 255, 255), 2)
        cv2.putText(img, "Car enter count  " + ":" + str(totalLineCrossedLeft), (970, 60),
                    font_style, 1.5, (0, 255, 0), 3)
        cv2.putText(img, "Car exit count  " + ":" + str(totalLineCrossedRight), (0, 60),
                    font_style, 1.5, (0, 255, 0), 3)
        # cv2.putText(img, "Total People line crossed " + ":" + str(totalLineCrossed), (0, 130),
        #             font_style, 0.8, (255, 255, 255), 2)

        cv2.putText(img, "Total Car crossed" + ":" + str(line_cross_count), (500, 160), font_style, 1.3,(0, 255, 255), 3)

    fps = 1./(time.time()-t1)
    cv2.putText(img, "FPS: {:.2f}".format(fps), (0,20), 0, 1, (0,0,255), 1)
    out.write(img)
    resize = cv2.resize(img, (1024, 790)) # THIS IS THE SIZE OF THE OUTPUT VIDEO THAT YOU'LL SEE ON THE SCREEN (change it accordingly)
    cv2.imshow('output', img)
    out.write(resize)
    if cv2.waitKey(1) == ord('q'):
        break
current_count = int(0)
vid.release()
# out.release()
