# from .pycocotools.coco import COCO
# from .pycocotools.cocoeval import COCOeval
import logging
from tool.cocoapi_master.PythonAPI.pycocotools.coco import COCO
from tool.cocoapi_master.PythonAPI.pycocotools.cocoeval import COCOeval
logger = logging.getLogger(__name__)
import numpy as np
import skimage.io as io
import json_tricks as json
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
pylab.rcParams['figure.figsize'] = (8.0, 10.0)


annType = ['segm', 'bbox', 'keypoints']
annType = annType[2]      # specify type here
prefix = 'person_keypoints' if annType == 'keypoints' else 'instances'
print('Running demo for *{:s}* results.'.format(annType))

annFile = '../../../data/person_keypoints_val2017.json'
cocoGt = COCO(annFile)
# img = cocoGt.loadImgs([285])[0]
# I = io.imread(img['coco_url'])
# plt.axis('off')
# plt.imshow(I)
# plt.show()

resFile = annFile
outputs = json.load(open(resFile))['annotations']
cocoDt = cocoGt.loadRes(outputs)

cocoEval = COCOeval(cocoGt, cocoDt, annType)
# cocoEval.params.imgIds = imgIds
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()

a = 1
