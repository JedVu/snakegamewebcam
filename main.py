import pygame
import time
import random
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import posenet
import argparse
import tensorflow as tf
# import os
from tensorflow.keras.preprocessing.image import img_to_array

pygame.init()

white = (255, 255, 255)
yellow = (255, 255, 102)
black = (0, 0, 0)
red = (213, 50, 80)
green = (0, 255, 0)
blue = (50, 153, 213)
 
# dis_width = 640
# dis_height = 480

dis_width = 960
dis_height = 720

snake_block = 40
snake_speed = 5

dis = pygame.display.set_mode([dis_width+2*snake_block+256,dis_height])
pygame.display.set_caption('Playing snake game with body gesture')

head_img = pygame.image.load("D:\ChinhResources\snake\pictures\snake\snake_head_40x40.png")
body_img = pygame.image.load("D:\ChinhResources\snake\pictures\snake\snake_body_40x40.png")
tail_img = pygame.image.load("D:\ChinhResources\snake\pictures\snake\snake_tail_40x40.png")
mouse_img = pygame.image.load("D:\ChinhResources\snake\pictures\snake\mouse_40x40.png")

clock = pygame.time.Clock()
 

 
font_style = pygame.font.SysFont("bahnschrift", 25)
score_font = pygame.font.SysFont("comicsansms", 35)
navigate_font = pygame.font.SysFont("comicsansms", 60)
timeout_font = pygame.font.SysFont("comicsansms", 100)
 
def your_score(score):
    value = score_font.render("Your Score: " + str(score), True, yellow)
    dis.blit(value, [0, 0])
 
def predicted(text):
    value = navigate_font.render(text, True, red)
    dis.blit(value, [dis_width / 2 - 100, dis_height / 2 - 100])
 
def our_snake(snake_block, snake_list, direction):
    # pygame.draw.rect(dis, black, [snake_list[0][0], snake_list[0][1], snake_block, snake_block])

    # change the direction of snake_head image
    if direction == 'left':
        head_direction = pygame.transform.rotate(head_img, -90)
        dis.blit(head_direction, [snake_list[-1][0], snake_list[-1][1]])
    elif direction == 'right':
        head_direction = pygame.transform.rotate(head_img, 90)
        dis.blit(head_direction, [snake_list[-1][0], snake_list[-1][1]])
    elif direction == 'up':
        head_direction = pygame.transform.rotate(head_img, -180)
        dis.blit(head_direction, [snake_list[-1][0], snake_list[-1][1]])
    elif direction == 'down':
        head_direction = head_img
        dis.blit(head_direction, [snake_list[-1][0], snake_list[-1][1]])

    for x in snake_list[1:-1]:
        # pygame.draw.rect(dis, black, [x[0], x[1], snake_block, snake_block])
        dis.blit(body_img, [x[0], x[1]])
    
    dis.blit(tail_img, [snake_list[0][0], snake_list[0][1]])
 
def message(msg, color):
    for i, l in enumerate(msg.splitlines()):
        mesg = font_style.render(l, True, color)
        dis.blit(mesg, [dis_width / 6, dis_height / 3 + 40*i])

def countdown(text):
    if text == 'CHICKEN OHYAY!':
        value = timeout_font.render(text, True, yellow)
        dis.blit(value, [20, 350])
    else:
        value = navigate_font.render(text, True, yellow)
        dis.blit(value, [20, 350])

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--cam_width', type=int, default=640)
parser.add_argument('--cam_height', type=int, default=480)
parser.add_argument('--scale_factor', type=float, default=0.7125)
parser.add_argument('--file', type=str, default=None, help="Optionally use a video file instead of a live camera")
args = parser.parse_args()

def gameLoop(game_over = False, game_close = True):

    # Declare first position of snake and food -------------------------------
    x1 = dis_width / 2
    y1 = dis_height / 2
 
    x1_change = 0
    y1_change = 0
 
    snake_List = []
    Length_of_snake = 1
    
    foodx = round(random.randrange(0, dis_width - snake_block) / 40.0) * 40.0
    foody = round(random.randrange(0, dis_height - snake_block) / 40.0) * 40.0
    # -------------------------------------------------------------------------

    counter, counter_text = 64, '64'.ljust(0)
    with tf.compat.v1.Session() as sess:

        # Restore my pretrain model on Colab --------------------------------------
        model_restored = load_model('D:\ChinhResources\snake\model\jun 21\mobilenetv2_192x192_9920_37.h5')
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

        pygame.time.set_timer(pygame.USEREVENT, 1000)
        # frame_count = 0
        direction = 'down'

        while not game_over:

            # When Lose game -----------------------------------------
            while game_close == True:
                dis.fill(black)
                message("HOW MANY MOUSE YOU CAN CATCH?\nPress C to Play or Q to Quit", red)
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

            # this isn't particularly fast, use GL for drawing and display someday...
            overlay_image = posenet.draw_skel_and_kp(
                display_image, pose_scores, keypoint_scores, keypoint_coords,
                min_pose_score=0.15, min_part_score=0.1)

            img_white = posenet.draw_skel_and_kp(
                img_white, pose_scores, keypoint_scores, keypoint_coords,
                min_pose_score=0.15, min_part_score=0.1)

            img_white = cv2.resize(img_white, (192,192), interpolation = cv2.INTER_AREA)
            img_white = img_to_array(img_white)
            img_white = img_white / 255.0
            img_white = np.expand_dims(img_white, axis=0)

            # # Predict -------------------------------------------------
            prediction = model_restored.predict(img_white, batch_size=10)

            label = np.argmax(prediction, axis = 1)
            direction = label_class[label[0]]
            # proba = round(max(prediction[0])*100)
            
            # Display text about confidence rate above each box
            text = f'{direction}' #, {proba}%'
            # ---------------------------------------------------------
            
            # Put prediction result to screen
            predicted(text)

            
            # # rotate & flip frame
            # overlay_image = cv2.flip(overlay_image, 1)
            overlay_image = cv2.resize(overlay_image, (256,192), interpolation = cv2.INTER_AREA)
            overlay_image = cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB)
            surf = pygame.surfarray.make_surface(overlay_image)
            surf = pygame.transform.rotate(surf,270)
            surf = pygame.transform.flip(surf, True, False)

            # display webcam to pygame window
            dis.blit(surf, (dis_width+2*snake_block+1,0))
            

            # Play with keyboard -------------------
            # for event in pygame.event.get():
            #     if event.type == pygame.USEREVENT:
            #         if counter == 0:
            #             game_close = True
            #         counter -= 1
            #         counter_text = str(counter) if counter > 2 else 'CHICKEN OHYAY!!!'
            #     if event.type == pygame.QUIT:
            #         game_over = True
            #     if event.type == pygame.KEYDOWN:
            #         if event.key == pygame.K_LEFT:
            #             x1_change = -snake_block
            #             y1_change = 0
            #             direction = 'left'
            #         elif event.key == pygame.K_RIGHT:
            #             x1_change = snake_block
            #             y1_change = 0
            #             direction = 'right'
            #         elif event.key == pygame.K_UP:
            #             y1_change = -snake_block
            #             x1_change = 0
            #             direction = 'up'
            #         elif event.key == pygame.K_DOWN:
            #             y1_change = snake_block
            #             x1_change = 0
            #             direction = 'down'
            # ----------------------------------------

            # # Play with webcam integrated
            for event in pygame.event.get():
                # print('event')
                if event.type == pygame.USEREVENT:
                    if counter == 0:
                        game_close = True
                    counter -= 1
                    counter_text = str(counter).rjust(3) if counter > 2 else 'CHICKEN OHYAY!!!'
                if event.type == pygame.QUIT:
                    game_over = True
                if direction == 'left':
                    x1_change = -snake_block
                    y1_change = 0
                elif direction == 'right':
                    x1_change = snake_block
                    y1_change = 0
                elif direction == 'up':
                    y1_change = -snake_block
                    x1_change = 0
                elif direction == 'down':
                    y1_change = snake_block
                    x1_change = 0
            # print('loop end')
            # ----------------------------

            # When snake_Head touch to the boundary => Lose game 
            # if x1 >= dis_width or x1 < 0 or y1 >= dis_height or y1 < 0:
            #     game_close = True
            # --------------------------------------------------

            # When snake_Head touch to the boundary => restart position of snake to the opposite site
            
            if x1 > dis_width:
                x1 = -40
            elif x1 < 0:
                x1 = dis_width
            elif y1 > dis_height:
                y1 = -40
            elif y1 < 0:
                y1 = dis_height
            
            # update position of snake_Head and draw it
            x1 += x1_change
            y1 += y1_change
            # ----------------------------------------------------------------------------------------

            
            pygame.display.update()
            dis.fill(black)
            

            # pygame.draw.rect(dis, green, [foodx, foody, snake_block, snake_block]) #[coor x, coor y, lenght x, lenght y]
            dis.blit(mouse_img, [foodx, foody])

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

            our_snake(snake_block, snake_List, direction)
            your_score(Length_of_snake - 1)

            # pygame.display.update()

            # Make length of snake longer when eat food
            if (x1 == foodx and y1 == foody or 
                x1 == foodx+snake_block and y1 == foody+snake_block or 
                x1 == foodx-snake_block and y1 == foody-snake_block or
                x1 == foodx+snake_block and y1 == foody-snake_block or
                x1 == foodx-snake_block and y1 == foody+snake_block):
                foodx = round(random.randrange(0, dis_width - snake_block) / 40.0) * 40.0
                foody = round(random.randrange(0, dis_height - snake_block) / 40.0) * 40.0
                Length_of_snake += 1
            # -----------------------------------------
            countdown(counter_text)
            
            clock.tick(snake_speed)

        pygame.quit()
        quit()
 
 
gameLoop()