import pygame
import time
import random
import cv2
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model
import posenet
import argparse
import tensorflow.compat.v1 as tf
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array

pygame.init()

 
white = (255, 255, 255)
yellow = (255, 255, 102)
black = (0, 0, 0)
red = (213, 50, 80)
green = (0, 255, 0)
blue = (50, 153, 213)
 
dis_width = 640
dis_height = 480
 
dis = pygame.display.set_mode([1280,480])
pygame.display.set_caption('Snake Game INTERACTION')

clock = pygame.time.Clock()
 
snake_block = 10
snake_speed = 5
 
font_style = pygame.font.SysFont("bahnschrift", 25)
score_font = pygame.font.SysFont("comicsansms", 35)
 
def your_score(score):
    value = score_font.render("Your Score: " + str(score), True, yellow)
    dis.blit(value, [0, 0])
 
def predicted(text):
    value = score_font.render(text, True, green)
    dis.blit(value, [350, 0])
 
def our_snake(snake_block, snake_list):
    for x in snake_list:
        pygame.draw.rect(dis, black, [x[0], x[1], snake_block, snake_block])
 
def message(msg, color):
    mesg = font_style.render(msg, True, color)
    dis.blit(mesg, [dis_width / 6, dis_height / 3])

def gameLoop(game_over = False, game_close = True):

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=int, default=101)
    parser.add_argument('--cam_id', type=int, default=0)
    parser.add_argument('--cam_width', type=int, default=640)
    parser.add_argument('--cam_height', type=int, default=480)
    parser.add_argument('--scale_factor', type=float, default=0.7125)
    parser.add_argument('--file', type=str, default=None, help="Optionally use a video file instead of a live camera")
    args = parser.parse_args()
    
    # Declare first position of snake and food -------------------------------
    x1 = dis_width / 2
    y1 = dis_height / 2
 
    x1_change = 0
    y1_change = 0
 
    snake_List = []
    Length_of_snake = 1
 
    foodx = round(random.randrange(0, dis_width - snake_block) / 10.0) * 10.0
    foody = round(random.randrange(0, dis_height - snake_block) / 10.0) * 10.0
    # -------------------------------------------------------------------------

    with tf.Session() as sess:

        # Restore my pretrain model on Colab --------------------------------------
        model_restored = load_model('D:/ChinhResources/snake/model/mobilenetv2_96x96_99percentage.h5')
        label_class = {0: 'down', 1: 'left', 2: 'right', 3: 'up'}
        # -------------------------------------------------------------------------

        model_cfg, model_outputs = posenet.load_model(args.model, sess)
        output_stride = model_cfg['output_stride']

        if args.file is not None:
            cap = cv2.VideoCapture(args.file)
        else:
            cap = cv2.VideoCapture(args.cam_id)
        cap.set(3, args.cam_width)
        cap.set(4, args.cam_height)

        frame_count = 0
        while not game_over:
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

            # path = 'D:\ChinhResources\snake\pictures'
            # cv2.imwrite(os.path.join(path , f'image{frame_count}.jpg'), img_white)
            # frame_count += 1

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
            
            # Display text about confidence rate above each box
            text = f'{name} #, {proba}%'
            # ---------------------------------------------------------
            
            # Put prediction result to screen
            predicted(text)
            
            # rotate & flip frame
            # overlay_image = cv2.flip(overlay_image, 1)
            overlay_image = cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB)
            surf = pygame.surfarray.make_surface(overlay_image)
            surf = pygame.transform.rotate(surf,270)
            surf = pygame.transform.flip(surf, True, False)

            # display webcam to pygame window
            dis.blit(surf, (681,0))
            pygame.display.update()

            # When Lose game -----------------------------------------
            while game_close == True:
                dis.fill(blue)
                message("Press C to Play (again) or Q to Quit", red)
                your_score(Length_of_snake - 1)
                pygame.display.update()
    
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_q:
                            game_over = True
                            game_close = False
                        if event.key == pygame.K_c:
                            gameLoop(False, False)
            # ---------------------------------------------------------

            # Play with keyboard -------------------
            # for event in pygame.event.get():
            #     if event.type == pygame.QUIT:
            #         game_over = True
            #     if event.type == pygame.KEYDOWN:
            #         if event.key == pygame.K_LEFT:
            #             x1_change = -snake_block
            #             y1_change = 0
            #         elif event.key == pygame.K_RIGHT:
            #             x1_change = snake_block
            #             y1_change = 0
            #         elif event.key == pygame.K_UP:
            #             y1_change = -snake_block
            #             x1_change = 0
            #         elif event.key == pygame.K_DOWN:
            #             y1_change = snake_block
            #             x1_change = 0
            # ----------------------------------------

            
            # Play with webcam integrated
            if name == 'left':
                x1_change = -snake_block
                y1_change = 0
            elif name == 'right':
                x1_change = snake_block
                y1_change = 0
            elif name == 'up':
                y1_change = -snake_block
                x1_change = 0
            elif name == 'down':
                y1_change = snake_block
                x1_change = 0
            # ----------------------------

            # When snake_Head touch to the boundary => Lose game 
            if x1 >= dis_width or x1 < 0 or y1 >= dis_height or y1 < 0:
                game_close = True
            # --------------------------------------------------

            # update position of snake_Head and draw it
            x1 += x1_change
            y1 += y1_change
            dis.fill(blue)
            pygame.draw.rect(dis, green, [foodx, foody, snake_block, snake_block]) #[coor x, coor y, lenght x, lenght y]

            snake_Head = []
            snake_Head.append(x1)
            snake_Head.append(y1)
            snake_List.append(snake_Head)

            # Cut the tail of snake_List when moving
            if len(snake_List) > Length_of_snake:
                del snake_List[0]
            # --------------------------------------

            # If snake_Head hit the body => Lose game
            # Should turn off to make it easier when interact with hands
            # for x in snake_List[:-1]:
            #     if x == snake_Head:
            #         game_close = True
            # ----------------------------------------

            our_snake(snake_block, snake_List)
            your_score(Length_of_snake - 1)
    
            # pygame.display.update()

            # Make length of snake longer when eat food
            if x1 == foodx and y1 == foody:
                foodx = round(random.randrange(0, dis_width - snake_block) / 10.0) * 10.0
                foody = round(random.randrange(0, dis_height - snake_block) / 10.0) * 10.0
                Length_of_snake += 1
            # -----------------------------------------

            clock.tick(snake_speed)
    
        pygame.quit()
        quit()
 
 
gameLoop()