import sys
import os
import cv2
import time
CENTERTRACK_PATH = "/home/zjx/CenterTrack/src/lib/"
sys.path.insert(0, CENTERTRACK_PATH)

from detector import Detector
from opts import opts

MODEL_PATH = "/home/zjx/CenterTrack/models/coco_tracking.pth"
TASK = "tracking"  # or 'tracking,multi_pose' for pose tracking and 'tracking,ddd' for monocular 3d tracking
requires = '{} --load_model {}'.format(TASK, MODEL_PATH).split(' ')
opt = opts().init(requires)
#print(opt.load_model)
detector = Detector(opt)

base_addr = "/home/zjx/backup/UA-DETRAC/DETRAC-train-data/Insight-MVT_Annotation_Train/MVI_20011"
dirs = os.listdir(base_addr)
dirs.sort()
for dir_ in dirs:
  print(dir_)
  img = cv2.imread(base_addr+'/'+dir_)
  ret = detector.run(img)['results']
  for idx,tracklets in enumerate(ret):
      
      cv2.rectangle(img, (tracklets['bbox'][0],tracklets['bbox'][1]), (tracklets['bbox'][2],tracklets['bbox'][3]), (216,201,0), 2)
  cv2.imshow('hello',img)
  cv2.waitKey(100)
  # print(ret)
