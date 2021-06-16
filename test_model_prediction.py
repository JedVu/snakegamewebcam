import tensorflow.compat.v1 as tf
import tensorflow as tf2

import cv2
import time
import argparse
import numpy as np
import os

import posenet
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--cam_width', type=int, default=640)
parser.add_argument('--cam_height', type=int, default=480)
parser.add_argument('--scale_factor', type=float, default=0.7125)
parser.add_argument('--file', type=str, default=None, help="Optionally use a video file instead of a live camera")
args = parser.parse_args()




def main():
    with tf.Session() as sess:
        model_restored = load_model('D:/ChinhResources/snake/model/mobilenetv2_96x96_97percentage.h5')
        label_class = {0: 'down', 1: 'left', 2: 'right', 3: 'up'}

        model_cfg, model_outputs = posenet.load_model(args.model, sess)
        output_stride = model_cfg['output_stride']

        if args.file is not None:
            cap = cv2.VideoCapture(args.file)
        else:
            cap = cv2.VideoCapture(args.cam_id)
        cap.set(3, args.cam_width)
        cap.set(4, args.cam_height)

        # start = time.time()
        # frame_count = 0
        while True:
            input_image, display_image, output_scale = posenet.read_cap(
                cap, scale_factor=args.scale_factor, output_stride=output_stride)

            display_image = cv2.flip(display_image, 1)


            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
                model_outputs,
                feed_dict={'image:0': input_image}
            )

            pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multi.decode_multiple_poses(
                heatmaps_result.squeeze(axis=0),
                offsets_result.squeeze(axis=0),
                displacement_fwd_result.squeeze(axis=0),
                displacement_bwd_result.squeeze(axis=0),
                output_stride=output_stride,
                max_pose_detections=10,
                min_pose_score=0.15)

            keypoint_coords *= output_scale

            img_white = np.zeros([480, 640],dtype=np.uint8)
            img_white.fill(255)
            # TODO this isn't particularly fast, use GL for drawing and display someday...
            overlay_image = posenet.draw_skel_and_kp(
                display_image, pose_scores, keypoint_scores, keypoint_coords,
                min_pose_score=0.15, min_part_score=0.1)

            img_white = posenet.draw_skel_and_kp(
                img_white, pose_scores, keypoint_scores, keypoint_coords,
                min_pose_score=0.15, min_part_score=0.1)

            # time.sleep(0.3)
            


            img_white = cv2.resize(img_white, (96,96), interpolation = cv2.INTER_AREA)
            # path = 'D:\ChinhResources\snake\pictures'
            # cv2.imwrite(os.path.join(path , f'image{frame_count}.jpg'), img_white)
            # frame_count += 1
            img_white = img_to_array(img_white)
            img_white = img_white / 255.0
            img_white = np.expand_dims(img_white, axis=0)

            # # Predict -------------------------------------------------
            prediction = model_restored.predict(img_white, batch_size=10)

            label = np.argmax(prediction, axis = 1)
            name = label_class[label[0]]
            proba = round(max(prediction[0])*100)
            
            # # Display text about confidence rate above each box
            text = f'{name} #, {proba}%'
            # # ---------------------------------------------------------

            cv2.putText(overlay_image,text,(0,35),cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 255, 0),2)


            cv2.imshow('posenet', overlay_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        # print('Average FPS: ', frame_count / (time.time() - start))


if __name__ == "__main__":
    main()