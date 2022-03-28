#!/usr/bin/env python
import sys
import threading
import time

import numpy as np
import rospkg
import rospy
from cv_bridge import CvBridge, CvBridgeError

from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer

pkg = rospkg.RosPack().get_path('detic_ros')
sys.path.insert(0, pkg + '/third_party/CenterNet2/projects/CenterNet2/')
from centernet.config import add_centernet_config
from sensor_msgs.msg import Image
from pcl_msgs.msg import PointIndices
from jsk_recognition_msgs.msg import ClassificationResult
from jsk_recognition_msgs.msg import ClusterPointIndices

from detic.config import add_detic_config
from detic.modeling.utils import reset_cls_test


class Detectron2node(object):
    def __init__(self):
        rospy.logwarn("Initializing")
        setup_logger()

        self._bridge = CvBridge()
        self._last_msg = None
        self._msg_lock = threading.Lock()
        self._image_counter = 0

        self.cfg = get_cfg()
        add_centernet_config(self.cfg)
        add_detic_config(self.cfg)
        rospy.loginfo(self.cfg)
        self.cfg.merge_from_file(self.load_param('~config', pkg+ '/configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml'))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.load_param('~detection_threshold', 0.5) # set threshold for this model
        self.cfg.MODEL.WEIGHTS = self.load_param('~model', pkg
                                                 + '/models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth')
        self.cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand'
        self.cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH = pkg + '/datasets/metadata/lvis_v1_train_cat_info.json'
        # For better visualization purpose. Set to False for all classes.
        self.cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
        self.predictor = DefaultPredictor(self.cfg)

        BUILDIN_CLASSIFIER = {
            'lvis': 'datasets/metadata/lvis_v1_clip_a+cname.npy',
            'objects365': 'datasets/metadata/o365_clip_a+cnamefix.npy',
            'openimages': 'datasets/metadata/oid_clip_a+cname.npy',
            'coco': 'datasets/metadata/coco_clip_a+cname.npy',
        }


        BUILDIN_METADATA_PATH = {
            'lvis': 'lvis_v1_val',
            'objects365': 'objects365_v2_val',
            'openimages': 'oid_val_expanded',
            'coco': 'coco_2017_val',
        }

        vocabulary = 'lvis'  # change to 'lvis', 'objects365', 'openimages', or 'coco'
        metadata = MetadataCatalog.get(BUILDIN_METADATA_PATH[vocabulary])
        classifier = BUILDIN_CLASSIFIER[vocabulary]
        num_classes = len(metadata.thing_classes)
        reset_cls_test(self.predictor.model, classifier, num_classes)
        self._class_names = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]).get("thing_classes", None)

        self._visualization = self.load_param('~visualization',True)
        # self._result_pub = rospy.Publisher('~result', Result, queue_size=1)
        self._result_pub = rospy.Publisher('~result', ClassificationResult, queue_size=1)
        self._cluster_pub = rospy.Publisher('~cluster', ClusterPointIndices, queue_size=1)
        self._vis_pub = rospy.Publisher('~visualization', Image, queue_size=1)
        self._sub = rospy.Subscriber(
            self.load_param(
                '~input',
                '/hand_camera/color/image_rect_color'),
            Image,
            self.callback_image,
            queue_size=1)
        self.start_time = time.time()
        rospy.logwarn("Initialized")

    def run(self):

        rate = rospy.Rate(100)
        while not rospy.is_shutdown():
            if self._msg_lock.acquire(False):
                img_msg = self._last_msg
                self._last_msg = None
                self._msg_lock.release()
            else:
                rate.sleep()
                continue

            if img_msg is not None:
                self._image_counter = self._image_counter + 1
                if (self._image_counter % 11) == 10:
                    rospy.loginfo("Images detected per second=%.2f",
                                  float(self._image_counter) / (time.time() - self.start_time))

                np_image = self.convert_to_cv_image(img_msg)

                outputs = self.predictor(np_image)
                result = outputs["instances"].to("cpu")
                result_msg, cluster_msg = self.getResult(result)

                self._result_pub.publish(result_msg)
                self._cluster_pub.publish(cluster_msg)

                # Visualize results
                if self._visualization:
                    v = Visualizer(np_image[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)
                    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
                    img = v.get_image()[:, :, ::-1]

                    image_msg = self._bridge.cv2_to_imgmsg(img, 'bgr8')
                    self._vis_pub.publish(image_msg)

            rate.sleep()

    def getResult(self, predictions):
        if predictions.has("pred_masks"):
            masks = np.asarray(predictions.pred_masks)
        else:
            return

        result = ClassificationResult()
        result.header = self._header
        result.classifier = 'Detic'
        result.labels = predictions.pred_classes if predictions.has("pred_classes") else None
        result.label_names = np.array(self._class_names)[result.labels.numpy()]
        result.label_proba = predictions.scores if predictions.has("scores") else None

        cluster = ClusterPointIndices()
        cluster.header = self._header
        for mask in masks:
            indices = PointIndices()
            indices.header = self._header
            mask_idices = np.where(mask)
            indices.indices = mask_idices[0] * mask.shape[1] + mask_idices[1]
            cluster.cluster_indices.append(indices)

        return result, cluster

    def convert_to_cv_image(self, image_msg):

        if image_msg is None:
            return None

        cv_img = self._bridge.imgmsg_to_cv2(image_msg, 'bgr8')

        return cv_img

    def callback_image(self, msg):
        rospy.logdebug("Get an image")
        if self._msg_lock.acquire(False):
            self._last_msg = msg
            self._header = msg.header
            self._msg_lock.release()

    @staticmethod
    def load_param(param, default=None):
        new_param = rospy.get_param(param, default)
        rospy.loginfo("[Detic_ros] %s: %s", param, new_param)
        return new_param

def main(argv):
    rospy.init_node('detic_ros')
    node = Detectron2node()
    node.run()

if __name__ == '__main__':
    main(sys.argv)
