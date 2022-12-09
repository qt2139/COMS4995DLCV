import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from nets.facenet import Facenet
from nets_retinaface.retinaface import RetinaFace
from utils.anchors import Anchors
from utils.config import cfg_mnet, cfg_re50
from utils.utils import (Alignment_1, compare_faces, letterbox_image,
                         preprocess_input)
from utils.utils_bbox import (decode, decode_landm, non_max_suppression,
                              retinaface_correct_boxes)


def cv2ImgAddText(img, label, left, top, textColor=(255, 255, 255)):
    img = Image.fromarray(np.uint8(img))
    font = ImageFont.truetype(font='model_data/simhei.ttf', size=20)

    draw = ImageDraw.Draw(img)
    label = label.encode('utf-8')
    draw.text((left, top), str(label, 'UTF-8'), fill=textColor, font=font)
    return np.asarray(img)


# --------------------------------------#
# Be sure to pay attention to the correspondence between backbone and model_path.
# After changing facenet_model.
# Be sure to pay attention to re-encoding the faces.
# --------------------------------------#
class Retinaface(object):
    _defaults = {
        # ----------------------------------------------------------------------#
        #   retinaface's trained weight path
        # ----------------------------------------------------------------------#
        "retinaface_model_path": 'model_data/Retinaface_mobilenet0.25.pth',
        # ----------------------------------------------------------------------#
        #   The backbone network used by retinaface
        # ----------------------------------------------------------------------#
        "retinaface_backbone": "mobilenet",
        # ----------------------------------------------------------------------#
        #   Only prediction frames with scores greater than the confidence level in retinaface will be retained
        # ----------------------------------------------------------------------#
        "confidence": 0.5,
        # ----------------------------------------------------------------------#
        #   The size of nms_iou used for non-extreme suppression in retinaface
        # ----------------------------------------------------------------------#
        "nms_iou": 0.3,
        # ----------------------------------------------------------------------#
        # Whether image size limit is needed.
        # The input image size will affect the FPS significantly, you can reduce the input_shape if you want to speed up the detection speed.
        # When enabled, it will limit the input image size to input_shape. otherwise, use the original image for prediction.
        # It will lead to biased detection results, the trunk is resnet50 without this problem.
        # The input_shape can be adjusted according to the size of the input image, note that it is a multiple of 32, e.g. [640, 640, 3]
        # ----------------------------------------------------------------------#
        "retinaface_input_shape": [640, 640, 3],
        # ----------------------------------------------------------------------#
        #   Whether the image size limit is required.
        # ----------------------------------------------------------------------#
        "letterbox_image": True,

        # ----------------------------------------------------------------------#
        #   facenet's trained weight path
        # ----------------------------------------------------------------------#
        "facenet_model_path": 'model_data/facenet_mobilenet.pth',
        # ----------------------------------------------------------------------#
        #   The backbone network used by facenet, mobilenet
        # ----------------------------------------------------------------------#
        "facenet_backbone": "mobilenet",
        # ----------------------------------------------------------------------#
        #   The size of the input image used by facenet
        # ----------------------------------------------------------------------#
        "facenet_input_shape": [160, 160, 3],
        # ----------------------------------------------------------------------#
        #   Face distance thresholds used by facenet
        # ----------------------------------------------------------------------#
        "facenet_threhold": 0.9,

        # --------------------------------#
        # Whether to use Cuda
        # No GPU can be set to False
        # --------------------------------#
        "cuda": True
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # ---------------------------------------------------#
    #   Initialize Retinaface
    # ---------------------------------------------------#
    def __init__(self, encoding=0, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)

        # ---------------------------------------------------#
        #   Config information for different backbone networks
        # ---------------------------------------------------#
        if self.retinaface_backbone == "mobilenet":
            self.cfg = cfg_mnet
        else:
            self.cfg = cfg_re50

        # ---------------------------------------------------#
        #   Generation of a priori boxes
        # ---------------------------------------------------#
        self.anchors = Anchors(self.cfg, image_size=(
        self.retinaface_input_shape[0], self.retinaface_input_shape[1])).get_anchors()
        self.generate()

        try:
            self.known_face_encodings = np.load(
                "model_data/{backbone}_face_encoding.npy".format(backbone=self.facenet_backbone))
            self.known_face_names = np.load("model_data/{backbone}_names.npy".format(backbone=self.facenet_backbone))
        except:
            if not encoding:
                print("If the loading of existing face features fails, please check whether the relevant face feature file is generated under model_data.")
            pass

    # ---------------------------------------------------#
    #   获得所有的分类
    # ---------------------------------------------------#
    def generate(self):
        # -------------------------------#
        #   Loading models and weights
        # -------------------------------#
        self.net = RetinaFace(cfg=self.cfg, phase='eval', pre_train=False).eval()
        self.facenet = Facenet(backbone=self.facenet_backbone, mode="predict").eval()

        print('Loading weights into state dict...')
        state_dict = torch.load(self.retinaface_model_path)
        self.net.load_state_dict(state_dict)

        state_dict = torch.load(self.facenet_model_path)
        self.facenet.load_state_dict(state_dict, strict=False)

        if self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

            self.facenet = nn.DataParallel(self.facenet)
            self.facenet = self.facenet.cuda()
        print('Finished!')

    def encode_face_dataset(self, image_paths, names):
        face_encodings = []
        for index, path in enumerate(tqdm(image_paths)):
            # ---------------------------------------------------#
            #   Open Face Image
            # ---------------------------------------------------#
            image = np.array(Image.open(path), np.float32)
            # ---------------------------------------------------#
            #   Make a backup of the input image
            # ---------------------------------------------------#
            old_image = image.copy()
            # ---------------------------------------------------#
            #   Calculate the height and width of the input image
            # ---------------------------------------------------#
            im_height, im_width, _ = np.shape(image)
            # ---------------------------------------------------#
            #   Calculate the scale, which is used to convert the obtained prediction frame into the height and width of the original image
            # ---------------------------------------------------#
            scale = [
                np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0]
            ]
            scale_for_landmarks = [
                np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
                np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
                np.shape(image)[1], np.shape(image)[0]
            ]
            if self.letterbox_image:
                image = letterbox_image(image, [self.retinaface_input_shape[1], self.retinaface_input_shape[0]])
                anchors = self.anchors
            else:
                anchors = Anchors(self.cfg, image_size=(im_height, im_width)).get_anchors()

            # ---------------------------------------------------#
            #   Pass the processed images into the Retinaface network for prediction
            # ---------------------------------------------------#
            with torch.no_grad():
                # -----------------------------------------------------------#
                #   Image pre-processing, normalization.
                # -----------------------------------------------------------#
                image = torch.from_numpy(preprocess_input(image).transpose(2, 0, 1)).unsqueeze(0).type(
                    torch.FloatTensor)

                if self.cuda:
                    image = image.cuda()
                    anchors = anchors.cuda()

                loc, conf, landms = self.net(image)
                # -----------------------------------------------------------#
                #   Decoding the prediction frame
                # -----------------------------------------------------------#
                boxes = decode(loc.data.squeeze(0), anchors, self.cfg['variance'])
                # -----------------------------------------------------------#
                #   Obtain confidence in the prediction results
                # -----------------------------------------------------------#
                conf = conf.data.squeeze(0)[:, 1:2]
                # -----------------------------------------------------------#
                #   Decoding of key points of the face
                # -----------------------------------------------------------#
                landms = decode_landm(landms.data.squeeze(0), anchors, self.cfg['variance'])

                # -----------------------------------------------------------#
                #   Stacking of face detection results
                # -----------------------------------------------------------#
                boxes_conf_landms = torch.cat([boxes, conf, landms], -1)
                boxes_conf_landms = non_max_suppression(boxes_conf_landms, self.confidence)

                if len(boxes_conf_landms) <= 0:
                    print(names[index], "：未检测到人脸")
                    continue
                # ---------------------------------------------------------#
                #   If letterbox_image is used, remove the gray bar.
                # ---------------------------------------------------------#
                if self.letterbox_image:
                    boxes_conf_landms = retinaface_correct_boxes(boxes_conf_landms, \
                                                                 np.array([self.retinaface_input_shape[0],
                                                                           self.retinaface_input_shape[1]]),
                                                                 np.array([im_height, im_width]))

            boxes_conf_landms[:, :4] = boxes_conf_landms[:, :4] * scale
            boxes_conf_landms[:, 5:] = boxes_conf_landms[:, 5:] * scale_for_landmarks

            # ---------------------------------------------------#
            #   Select the largest face frame.
            # ---------------------------------------------------#
            best_face_location = None
            biggest_area = 0
            for result in boxes_conf_landms:
                left, top, right, bottom = result[0:4]

                w = right - left
                h = bottom - top
                if w * h > biggest_area:
                    biggest_area = w * h
                    best_face_location = result

            # ---------------------------------------------------#
            #   Intercepting images
            # ---------------------------------------------------#
            crop_img = old_image[int(best_face_location[1]):int(best_face_location[3]),
                       int(best_face_location[0]):int(best_face_location[2])]
            landmark = np.reshape(best_face_location[5:], (5, 2)) - np.array(
                [int(best_face_location[0]), int(best_face_location[1])])
            crop_img, _ = Alignment_1(crop_img, landmark)

            crop_img = np.array(
                letterbox_image(np.uint8(crop_img), (self.facenet_input_shape[1], self.facenet_input_shape[0]))) / 255
            crop_img = crop_img.transpose(2, 0, 1)
            crop_img = np.expand_dims(crop_img, 0)
            # ---------------------------------------------------#
            #   Using the image to calculate the feature vector of length 128
            # ---------------------------------------------------#
            with torch.no_grad():
                crop_img = torch.from_numpy(crop_img).type(torch.FloatTensor)
                if self.cuda:
                    crop_img = crop_img.cuda()

                face_encoding = self.facenet(crop_img)[0].cpu().numpy()
                face_encodings.append(face_encoding)

        np.save("model_data/{backbone}_face_encoding.npy".format(backbone=self.facenet_backbone), face_encodings)
        np.save("model_data/{backbone}_names.npy".format(backbone=self.facenet_backbone), names)

    # ---------------------------------------------------#
    #   Test pictures
    # ---------------------------------------------------#
    def detect_image(self, image):
        # ---------------------------------------------------#
        #   Make a backup of the input image, which is used later for drawing
        # ---------------------------------------------------#
        old_image = image.copy()
        # ---------------------------------------------------#
        #   Convert images to numpy form
        # ---------------------------------------------------#
        image = np.array(image, np.float32)

        # ---------------------------------------------------#
        #   Retinaface detection section - Start
        # ---------------------------------------------------#
        # ---------------------------------------------------#
        #   Calculate the height and width of the input image
        # ---------------------------------------------------#
        im_height, im_width, _ = np.shape(image)
        # ---------------------------------------------------#
        #   Calculate the scale, which is used to convert the obtained prediction frame into the height and width of the original image
        # ---------------------------------------------------#
        scale = [
            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0]
        ]
        scale_for_landmarks = [
            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
            np.shape(image)[1], np.shape(image)[0]
        ]

        # ---------------------------------------------------------#
        #   letterbox_image can add gray bars to the image to achieve undistorted resize
        # ---------------------------------------------------------#
        if self.letterbox_image:
            image = letterbox_image(image, [self.retinaface_input_shape[1], self.retinaface_input_shape[0]])
            anchors = self.anchors
        else:
            anchors = Anchors(self.cfg, image_size=(im_height, im_width)).get_anchors()

        # ---------------------------------------------------#
        #   Pass the processed images into the Retinaface network for prediction
        # ---------------------------------------------------#
        with torch.no_grad():
            # -----------------------------------------------------------#
            #   Image pre-processing, normalization.
            # -----------------------------------------------------------#
            image = torch.from_numpy(preprocess_input(image).transpose(2, 0, 1)).unsqueeze(0).type(torch.FloatTensor)

            if self.cuda:
                anchors = anchors.cuda()
                image = image.cuda()

            # ---------------------------------------------------------#
            #   Incoming network for prediction
            # ---------------------------------------------------------#
            loc, conf, landms = self.net(image)
            # ---------------------------------------------------#
            #   Decoding of the Retinaface network, we end up with prediction frames
            #   Decoding and non-extreme suppression of the prediction results
            # ---------------------------------------------------#
            boxes = decode(loc.data.squeeze(0), anchors, self.cfg['variance'])

            conf = conf.data.squeeze(0)[:, 1:2]

            landms = decode_landm(landms.data.squeeze(0), anchors, self.cfg['variance'])

            # -----------------------------------------------------------#
            #   Stacking of face detection results
            # -----------------------------------------------------------#
            boxes_conf_landms = torch.cat([boxes, conf, landms], -1)
            boxes_conf_landms = non_max_suppression(boxes_conf_landms, self.confidence)

            # ---------------------------------------------------#
            #   If there is no prediction box, return the original image and 'Unknow' and end the frame
            # ---------------------------------------------------#
            if len(boxes_conf_landms) <= 0:
                return old_image, 'Unknow'

            # ---------------------------------------------------------#
            #   If letterbox_image is used, remove the gray bar.
            # ---------------------------------------------------------#
            if self.letterbox_image:
                boxes_conf_landms = retinaface_correct_boxes(boxes_conf_landms, \
                                                             np.array([self.retinaface_input_shape[0],
                                                                       self.retinaface_input_shape[1]]),
                                                             np.array([im_height, im_width]))

            boxes_conf_landms[:, :4] = boxes_conf_landms[:, :4] * scale
            boxes_conf_landms[:, 5:] = boxes_conf_landms[:, 5:] * scale_for_landmarks

        # ---------------------------------------------------#
        #   Retinaface detection section - end
        # ---------------------------------------------------#

        # -----------------------------------------------#
        #   Facenet coding section - start
        # -----------------------------------------------#
        face_encodings = []
        for boxes_conf_landm in boxes_conf_landms:
            # ----------------------#
            #   Image capture, face correction
            # ----------------------#
            boxes_conf_landm = np.maximum(boxes_conf_landm, 0)
            crop_img = np.array(old_image)[int(boxes_conf_landm[1]):int(boxes_conf_landm[3]),
                       int(boxes_conf_landm[0]):int(boxes_conf_landm[2])]
            new_img_1 = cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite('img/new_img_1.jpg', new_img_1)
            landmark = np.reshape(boxes_conf_landm[5:], (5, 2)) - np.array(
                [int(boxes_conf_landm[0]), int(boxes_conf_landm[1])])
            crop_img, _ = Alignment_1(crop_img, landmark)

            # ----------------------#
            #   Face encoding
            # ----------------------#
            crop_img = np.array(
                letterbox_image(np.uint8(crop_img), (self.facenet_input_shape[1], self.facenet_input_shape[0]))) / 255
            crop_img = np.expand_dims(crop_img.transpose(2, 0, 1), 0)
            with torch.no_grad():
                crop_img = torch.from_numpy(crop_img).type(torch.FloatTensor)
                if self.cuda:
                    crop_img = crop_img.cuda()

                # -----------------------------------------------#
                #   Compute a feature vector of length 128 using facenet_model
                # -----------------------------------------------#
                face_encoding = self.facenet(crop_img)[0].cpu().numpy()
                face_encodings.append(face_encoding)
        # -----------------------------------------------#
        #   Facenet coding section - end
        # -----------------------------------------------#

        # -----------------------------------------------#
        #   Face Feature Matching - Start
        # -----------------------------------------------#
        face_names = []
        # Calculate all the faces in this frame and compare them with known faces
        for face_encoding in face_encodings:
            # -----------------------------------------------------#
            #   Take a face and compare it with all the faces in the database and calculate the score
            # -----------------------------------------------------#
            matches, face_distances = compare_faces(self.known_face_encodings, face_encoding,
                                                    tolerance=self.facenet_threhold)
            name = "Unknown"
            # -----------------------------------------------------#
            # Fetch the rating of this recent face
            # Fetch the serial number of the closest known face for the current incoming face
            # -----------------------------------------------------#
            # argim gets the index of the minimum distance and returns matches to find the label
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
            face_names.append(name)
            # print(face_names)
        # -----------------------------------------------#
        #   Face Feature Matching - End
        # -----------------------------------------------#

        for i, b in enumerate(boxes_conf_landms):
            text = "{:.4f}".format(b[4])
            b = list(map(int, b))
            # Calculate the ratio of the candidate box to the original area, b[0], b[1] are the coordinates of the upper left corner, b[1]b[2] are the coordinates of the lower right corner
            temp = (b[2] - b[0]) * (b[3] - b[1]) / (old_image.shape[1] * old_image.shape[0])
            # print(text)
            # print(temp)

            # ---------------------------------------------------#
            #   b[0]-b[3] are the coordinates of the face frame and b[4] is the score
            # ---------------------------------------------------#
            cv2.rectangle(old_image, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            # cv2.rectangle(old_image, (0, 0), (old_image.shape[1], old_image.shape[0]), (0, 0, 255), 2)
            cx = b[0]
            cy = b[1] + 12
            cv2.putText(old_image, text, (cx, cy),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

            # ---------------------------------------------------#
            #   b[5]-b[14] are the coordinates of the key points of the face
            # ---------------------------------------------------#
            cv2.circle(old_image, (b[5], b[6]), 1, (0, 0, 255), 4)
            cv2.circle(old_image, (b[7], b[8]), 1, (0, 255, 255), 4)
            cv2.circle(old_image, (b[9], b[10]), 1, (255, 0, 255), 4)
            cv2.circle(old_image, (b[11], b[12]), 1, (0, 255, 0), 4)
            cv2.circle(old_image, (b[13], b[14]), 1, (255, 0, 0), 4)

            name = face_names[i]
            # font = cv2.FONT_HERSHEY_SIMPLEX
            # cv2.putText(old_image, name, (b[0] , b[3] - 15), font, 0.75, (255, 255, 255), 2)
            # --------------------------------------------------------------#
            #   cv2 can not write Chinese, adding this paragraph can, but the detection speed will be somewhat reduced.
            #   If not necessary, you can switch to cv2 to display only English.
            # --------------------------------------------------------------#
            old_image = cv2ImgAddText(old_image, name, b[0] + 5, b[3] - 25)
        return old_image, face_names

    def get_FPS(self, image, test_interval):
        # ---------------------------------------------------#
        #   Make a backup of the input image, which is used later for drawing
        # ---------------------------------------------------#
        old_image = image.copy()
        # ---------------------------------------------------#
        #   Convert images to numpy form
        # ---------------------------------------------------#
        image = np.array(image, np.float32)

        # ---------------------------------------------------#
        #   Retinaface detection section - Start
        # ---------------------------------------------------#
        # ---------------------------------------------------#
        #   Calculate the height and width of the input image
        # ---------------------------------------------------#
        im_height, im_width, _ = np.shape(image)
        # ---------------------------------------------------#
        #   Calculate the scale, which is used to convert the obtained prediction frame into the height and width of the original image
        # ---------------------------------------------------#
        scale = [
            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0]
        ]
        scale_for_landmarks = [
            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
            np.shape(image)[1], np.shape(image)[0]
        ]

        # ---------------------------------------------------------#
        #   letterbox_image can add gray bars to the image to achieve undistorted resize
        # ---------------------------------------------------------#
        if self.letterbox_image:
            image = letterbox_image(image, [self.retinaface_input_shape[1], self.retinaface_input_shape[0]])
            anchors = self.anchors
        else:
            anchors = Anchors(self.cfg, image_size=(im_height, im_width)).get_anchors()

        # ---------------------------------------------------#
        #   Pass the processed images into the Retinaface network for prediction
        # ---------------------------------------------------#
        with torch.no_grad():
            # -----------------------------------------------------------#
            #   Image pre-processing, normalization.
            # -----------------------------------------------------------#
            image = torch.from_numpy(preprocess_input(image).transpose(2, 0, 1)).unsqueeze(0).type(torch.FloatTensor)

            if self.cuda:
                anchors = anchors.cuda()
                image = image.cuda()

            # ---------------------------------------------------------#
            #   Incoming network for prediction
            # ---------------------------------------------------------#
            loc, conf, landms = self.net(image)
            # ---------------------------------------------------#
            #   Decoding of the Retinaface network, we end up with prediction frames
            #   Decoding and non-extreme suppression of the prediction results
            # ---------------------------------------------------#
            boxes = decode(loc.data.squeeze(0), anchors, self.cfg['variance'])

            conf = conf.data.squeeze(0)[:, 1:2]

            landms = decode_landm(landms.data.squeeze(0), anchors, self.cfg['variance'])

            # -----------------------------------------------------------#
            #   Stacking of face detection results
            # -----------------------------------------------------------#
            boxes_conf_landms = torch.cat([boxes, conf, landms], -1)
            boxes_conf_landms = non_max_suppression(boxes_conf_landms, self.confidence)

        if len(boxes_conf_landms) > 0:
            # ---------------------------------------------------------#
            #   If letterbox_image is used, remove the gray bar.
            # ---------------------------------------------------------#
            if self.letterbox_image:
                boxes_conf_landms = retinaface_correct_boxes(boxes_conf_landms, \
                                                             np.array([self.retinaface_input_shape[0],
                                                                       self.retinaface_input_shape[1]]),
                                                             np.array([im_height, im_width]))

            boxes_conf_landms[:, :4] = boxes_conf_landms[:, :4] * scale
            boxes_conf_landms[:, 5:] = boxes_conf_landms[:, 5:] * scale_for_landmarks

            # ---------------------------------------------------#
            #   Retinaface detection section - end
            # ---------------------------------------------------#

            # -----------------------------------------------#
            #   Facenet coding section - start
            # -----------------------------------------------#
            face_encodings = []
            for boxes_conf_landm in boxes_conf_landms:
                # ----------------------#
                #   Image capture, face correction
                # ----------------------#
                boxes_conf_landm = np.maximum(boxes_conf_landm, 0)
                crop_img = np.array(old_image)[int(boxes_conf_landm[1]):int(boxes_conf_landm[3]),
                           int(boxes_conf_landm[0]):int(boxes_conf_landm[2])]
                landmark = np.reshape(boxes_conf_landm[5:], (5, 2)) - np.array(
                    [int(boxes_conf_landm[0]), int(boxes_conf_landm[1])])
                crop_img, _ = Alignment_1(crop_img, landmark)

                # ----------------------#
                #   Face encoding
                # ----------------------#
                crop_img = np.array(letterbox_image(np.uint8(crop_img),
                                                    (self.facenet_input_shape[1], self.facenet_input_shape[0]))) / 255
                crop_img = np.expand_dims(crop_img.transpose(2, 0, 1), 0)
                with torch.no_grad():
                    crop_img = torch.from_numpy(crop_img).type(torch.FloatTensor)
                    if self.cuda:
                        crop_img = crop_img.cuda()

                    # -----------------------------------------------#
                    #   Compute a feature vector of length 128 using facenet_model
                    # -----------------------------------------------#
                    face_encoding = self.facenet(crop_img)[0].cpu().numpy()
                    face_encodings.append(face_encoding)
            # -----------------------------------------------#
            #   Facenet coding section - end
            # -----------------------------------------------#

            # -----------------------------------------------#
            #   Face Feature Matching - Start
            # -----------------------------------------------#
            face_names = []
            for face_encoding in face_encodings:
                # -----------------------------------------------------#
                #   Take a face and compare it with all the faces in the database and calculate the score
                # -----------------------------------------------------#
                matches, face_distances = compare_faces(self.known_face_encodings, face_encoding,
                                                        tolerance=self.facenet_threhold)
                name = "Unknown"
                # -----------------------------------------------------#
                # Fetch the rating of this recent face
                # Fetch the serial number of the closest known face for the current incoming face
                # -----------------------------------------------------#
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]
                face_names.append(name)
            # -----------------------------------------------#
            #   Face Feature Matching - End
            # -----------------------------------------------#

        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                # ---------------------------------------------------------#
                #   Incoming network for prediction
                # ---------------------------------------------------------#
                loc, conf, landms = self.net(image)
                # ---------------------------------------------------#
                # Decoding of the Retinaface network, we end up with prediction frames
                # Decoding and non-extreme suppression of the prediction results
                # ---------------------------------------------------#
                boxes = decode(loc.data.squeeze(0), anchors, self.cfg['variance'])

                conf = conf.data.squeeze(0)[:, 1:2]

                landms = decode_landm(landms.data.squeeze(0), anchors, self.cfg['variance'])

                # -----------------------------------------------------------#
                # Stacking of face detection results
                # -----------------------------------------------------------#
                boxes_conf_landms = torch.cat([boxes, conf, landms], -1)
                boxes_conf_landms = non_max_suppression(boxes_conf_landms, self.confidence)

            if len(boxes_conf_landms) > 0:
                # ---------------------------------------------------------#
                # If letterbox_image is used, remove the gray bar.
                # ---------------------------------------------------------#
                if self.letterbox_image:
                    boxes_conf_landms = retinaface_correct_boxes(boxes_conf_landms, \
                                                                 np.array([self.retinaface_input_shape[0],
                                                                           self.retinaface_input_shape[1]]),
                                                                 np.array([im_height, im_width]))

                boxes_conf_landms[:, :4] = boxes_conf_landms[:, :4] * scale
                boxes_conf_landms[:, 5:] = boxes_conf_landms[:, 5:] * scale_for_landmarks

                # ---------------------------------------------------#
                # Retinaface detection section - end
                # ---------------------------------------------------#

                # -----------------------------------------------#
                # Facenet coding section - start
                # -----------------------------------------------#
                face_encodings = []
                for boxes_conf_landm in boxes_conf_landms:
                    # ----------------------#
                    # Image capture, face correction
                    # ----------------------#
                    boxes_conf_landm = np.maximum(boxes_conf_landm, 0)
                    crop_img = np.array(old_image)[int(boxes_conf_landm[1]):int(boxes_conf_landm[3]),
                               int(boxes_conf_landm[0]):int(boxes_conf_landm[2])]
                    landmark = np.reshape(boxes_conf_landm[5:], (5, 2)) - np.array(
                        [int(boxes_conf_landm[0]), int(boxes_conf_landm[1])])
                    crop_img, _ = Alignment_1(crop_img, landmark)

                    # ----------------------#
                    #   Face encoding
                    # ----------------------#
                    crop_img = np.array(letterbox_image(np.uint8(crop_img), (
                    self.facenet_input_shape[1], self.facenet_input_shape[0]))) / 255
                    crop_img = np.expand_dims(crop_img.transpose(2, 0, 1), 0)
                    with torch.no_grad():
                        crop_img = torch.from_numpy(crop_img).type(torch.FloatTensor)
                        if self.cuda:
                            crop_img = crop_img.cuda()

                        # -----------------------------------------------#
                        #   Compute a feature vector of length 128 using facenet_model
                        # -----------------------------------------------#
                        face_encoding = self.facenet(crop_img)[0].cpu().numpy()
                        face_encodings.append(face_encoding)
                # -----------------------------------------------#
                #   Facenet coding section - end
                # -----------------------------------------------#

                # -----------------------------------------------#
                #   Face Feature Matching - Start
                # -----------------------------------------------#
                face_names = []
                for face_encoding in face_encodings:
                    # -----------------------------------------------------#
                    #   Take a face and compare it with all the faces in the database and calculate the score
                    # -----------------------------------------------------#
                    matches, face_distances = compare_faces(self.known_face_encodings, face_encoding,
                                                            tolerance=self.facenet_threhold)
                    name = "Unknown"
                    # -----------------------------------------------------#
                    # Fetch the rating of this recent face
                    # Fetch the serial number of the closest known face for the current incoming face
                    # -----------------------------------------------------#
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                    face_names.append(name)
                # -----------------------------------------------#
                # Face Feature Matching - End
                # -----------------------------------------------#
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time
