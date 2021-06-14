import pygame
import time
import random
import cv2
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model
 
pygame.init()
 
white = (255, 255, 255)
yellow = (255, 255, 102)
black = (0, 0, 0)
red = (213, 50, 80)
green = (0, 255, 0)
blue = (50, 153, 213)
 
dis_width = 600
dis_height = 400
 
dis = pygame.display.set_mode([1200,400])
pygame.display.set_caption('Snake Game INTERACTION')

clock = pygame.time.Clock()
 
snake_block = 10
snake_speed = 10
 
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

    # Restore my pretrain model on Colab --------------------------------------
    model_restored = load_model('D:/ChinhResources/snake/model/keras_model.h5')
    label_class = {0: 'Left', 1: 'Right', 2: 'Up', 3: 'Down', 4: 'Normal'}
    # -------------------------------------------------------------------------

    cap = cv2.VideoCapture(0)
    while not game_over:
        ret, frame = cap.read()
        
        image = frame.copy()
        image = cv2.flip(image,1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Preprocess image to predict ----------------------------------------------
        np.set_printoptions(suppress=True)
        navigation = Image.fromarray(image)
        # Create the array of the right shape to feed into the keras model
        # The 'length' or number of images you can put into the array is
        # determined by the first position in the shape tuple, in this case 1.
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

        #resize the image to a 224x224 with the same strategy as in TM2:
        #resizing the image to be at least 224x224 and then cropping from the center
        size = (224, 224)
        navigation = ImageOps.fit(navigation, size, Image.ANTIALIAS)

        #turn the image into a numpy array
        image_array = np.asarray(navigation)
        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        # Load the image into the array
        data[0] = normalized_image_array
        # ---------------------------------------------------------------------------

        # Predict -------------------------------------------------
        prediction = model_restored.predict(data, batch_size=10)
        label = np.argmax(prediction, axis = 1)
        name = label_class[label[0]]
        proba = round(max(prediction[0])*100)
        
        # Display text about confidence rate above each box
        text = f'{name}' #, {proba}%'
        # ---------------------------------------------------------
        
        # Put prediction result to screen
        # cv2.putText(frame,text,(0,35),cv2.FONT_HERSHEY_SIMPLEX, 1,green,2)
        predicted(text)

        # rotate & flip frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        surf = pygame.surfarray.make_surface(frame)
        surf = pygame.transform.rotate(surf,270)
        # surf = pygame.transform.flip(surf, True, False)

        # display webcam to pygame window
        dis.blit(surf, (601,0))
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
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_over = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    x1_change = -snake_block
                    y1_change = 0
                elif event.key == pygame.K_RIGHT:
                    x1_change = snake_block
                    y1_change = 0
                elif event.key == pygame.K_UP:
                    y1_change = -snake_block
                    x1_change = 0
                elif event.key == pygame.K_DOWN:
                    y1_change = snake_block
                    x1_change = 0
        # ----------------------------------------

        
        # Play with webcam integrated
        # if name == 'Left':
        #     x1_change = -snake_block
        #     y1_change = 0
        # elif name == 'Right':
        #     x1_change = snake_block
        #     y1_change = 0
        # elif name == 'Up':
        #     y1_change = -snake_block
        #     x1_change = 0
        # elif name == 'Down':
        #     y1_change = snake_block
        #     x1_change = 0
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