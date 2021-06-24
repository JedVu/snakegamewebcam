import streamlit as st

st.set_page_config(
    layout="wide"
)

menu = ['Introduction','Behind the scenes','Live demo','Contact me']
choice = st.sidebar.selectbox('What I can do?', menu)


if choice == 'Introduction':
    # title = st.title("Navigate baby snake with body gesture")
    st.markdown("<h1 style='text-align: center; color: black;'>Navigate baby snake with body gesture</h1>", unsafe_allow_html=True)
    col1, col2, col3 = st.beta_columns((1, 3, 1))

    with col1:
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.image("media/mouse.png", use_column_width='always')
        # st.image("media/_left.png", use_column_width='always')

    with col2:
        st.markdown("<h3 style='text-align: center; color: black;'>I will tell you a story about ...me</h1>", unsafe_allow_html=True)
        # st.write("I will tell you a story about me...")
        video_file = open(r'media/baby_snake_2.mp4', 'rb')
        video_bytes = video_file.read()
        st.video(video_bytes)
        # st.markdown("<h3 style='text-align: right; color: black;'>I will tell you a story about ...me</h1>", unsafe_allow_html=True)

        
    with col3:
        # food = st.multiselect('what dog want to feed?', ['fish', 'cat', 'dog food'])
        # if food == ['fish']:
        #     st.write("do the lab to catch fish first")
        st.image("media/snake_and_mouse.png", use_column_width='always')
    
elif choice == 'Behind the scenes':
    st.markdown("<h1 style='text-align: center; color: black;'>How the snake game with body gesture made...</h1>", unsafe_allow_html=True)

    col1, col2, col3 = st.beta_columns((1,2,1))

    with col1:
        st.image(r"media/CDS_logo.jpg", use_column_width='always')
        
    with col2:
        st.markdown("<h2 style='text-align: center; color: black;'>Progress flow</h2>", unsafe_allow_html=True)
        st.image(r"media/flow.png", use_column_width='always')

        st.markdown("<h2 style='text-align: center; color: black;'>Posenet</h2>", unsafe_allow_html=True)
        st.image(r"media/posenet_1_1.png", use_column_width='always')
        st.image(r"media/posenet_2.png", use_column_width='always')
        st.image(r"media/posenet_3.png", use_column_width='always')
        st.image(r"media/posenet_4.png", use_column_width='always')


        st.markdown("<h2 style='text-align: center; color: black;'>Collect images from webcam and convert</h2>", unsafe_allow_html=True)
        st.image(r"media/convert_image_to_model.gif", use_column_width='always')
        st.image(r"media/right_left_down_pose.gif", use_column_width='always')

        st.markdown("<h2 style='text-align: center; color: black;'>Find some bad images which lost the skeleton pose</h2>", unsafe_allow_html=True)
        st.markdown("<div align = 'center'>Should check carefully because they affect on the accuracy of model directly</div>",unsafe_allow_html=True)
        st.image(r"media/find_bug_manually.png", use_column_width='always')

        st.markdown("<h2 style='text-align: center; color: black;'>Upload to Google Drive and use Colab Pro to train model</h2>", unsafe_allow_html=True)
        st.markdown("<div align = 'center'>Randomly choice images as guideline 8.5a_Real_ML_Project_with_Tensorflow_Keras.ipynb <br> and split them to Train_set, Validation_set and Test_set</div>",unsafe_allow_html=True)
        st.image(r"media/data_for_colab_train_model.png", use_column_width='always')

        st.markdown("<h3 style='text-align: center; color: black;'>Use MobileNetv2 to train model for 4 classes</h3>", unsafe_allow_html=True)
        st.markdown("<div align = 'center'>(Actually I used Xception first and the accuracy is better than MobileNetv2 but Xception is too heavy and maybe slow)</div>",unsafe_allow_html=True)
        st.image(r"media/mobilenetv2.png", use_column_width='always')
        st.write("[Example about MobileNetv2 work](https://www.youtube.com/watch?v=SibMvEVpqsk)")
        
        st.markdown("<h3 style='text-align: center; color: black;'>Evaluate model after training model</h3>", unsafe_allow_html=True)
        st.image(r"media/evaluate_model_after_train.png", use_column_width='always')

        st.markdown("<h3 style='text-align: center; color: black;'>Visualization accuracy and loss</h3>", unsafe_allow_html=True)
        st.image(r"media/visualization_accuracy.png", use_column_width='always')

        st.markdown("<h3 style='text-align: center; color: black;'>Visualization test with Test_set</h3>", unsafe_allow_html=True)
        st.image(r"media/test_test_set.png", use_column_width='always')

        st.markdown("<h3 style='text-align: center; color: black;'>SciKit Learn classification_report with full Test_set</h3>", unsafe_allow_html=True)
        st.image(r"media/classification_report.png", use_column_width='always')
        st.write("[Click to open my Colab file](https://colab.research.google.com/drive/1Xa4iggL5B9TAnaXB-oyhG2-kxx_SF18R?usp=sharing)")
        

        st.markdown("<h2 style='text-align: center; color: black;'>Put PoseNet, Trained Model with MobileNetv2 and Snake Game<br>together</h2>", unsafe_allow_html=True)
        
        st.markdown("<div align = 'left'>Snake game, I learned how to create it from a course of Udemy</div>",unsafe_allow_html=True)
        st.markdown("<div align = 'left'>But... It's really hard to play if combine with body gesture</div>",unsafe_allow_html=True)
        st.markdown("<div align = 'left'>So... I had to re-design all play rules and layout also (slower speed, bigger snake and mouse...)</div>",unsafe_allow_html=True)

        st.markdown("<h3 style='text-align: left; color: black;'>New rules:</h3>", unsafe_allow_html=True)
        st.markdown("<div align = 'left'>&nbsp;&nbsp;&nbsp;1. Snake can touch his body</div>",unsafe_allow_html=True)
        st.markdown("<div align = 'left'>&nbsp;&nbsp;&nbsp;2. Snake go through from this side to another side (no die if snake hit the border)</div>",unsafe_allow_html=True)
        st.markdown("<div align = 'left'>&nbsp;&nbsp;&nbsp;3. Snake can catch the mouse if snake head reach near to the mouse, no need to catch mouse with correct position</div>",unsafe_allow_html=True)
        st.write("[Click to open my snake on Github](https://github.com/JedVu/snakegamewebcam)")

        st.markdown("<h2 style='text-align: center; color: black;'>Streamlit part to present my snake</h2>", unsafe_allow_html=True)
        st.markdown("<div align = 'center'>(Introduce & Live demo)</div>",unsafe_allow_html=True)

        st.markdown("<h2 style='text-align: center; color: black;'>Weak points</h2>", unsafe_allow_html=True)
        st.markdown("""<div align = 'left'>
                        <p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Although the scores are very good, but my model detect pose depends on PoseNet skeleton and the light position of room.
                        If the brightness of the body right side is better than the left, the skeleton pose will be easy to loss the body left side and make the wrong prediction.
                        The other reason is: I used Posenet fast detection mode because the game needed.
                        </p>
                    </div>""",unsafe_allow_html=True)
        st.image(r"media/right_lost_pose.png", use_column_width='always')

        st.markdown("<h2 style='text-align: center; color: black;'>Future work</h2>", unsafe_allow_html=True)
        st.markdown("<div align = 'left'>I love to play games although I'm not too young. So I have not to be satisfied with my snake<br>I need to think how to improve my snake</div>",unsafe_allow_html=True)
        st.markdown("<div align = 'left'>&nbsp;&nbsp;&nbsp;1. Find another model to decrease 'lost pose' or try to use 'slower' posenet</div>",unsafe_allow_html=True)
        st.markdown("<div align = 'left'>&nbsp;&nbsp;&nbsp;2. Improve my code to keep the prediction more stable</div>",unsafe_allow_html=True)
        st.markdown("<div align = 'left'>&nbsp;&nbsp;&nbsp;3. Find a way to host my app to web</div>",unsafe_allow_html=True)
        

    with col3:
        st.image(r"media/CDS_logo_flip.jpg", use_column_width='always')
        

elif choice == 'Live demo':
    st.markdown("<h1 style='text-align: center; color: black;'>Under contruction...</h1>", unsafe_allow_html=True)
    st.markdown("<div align = 'center'>Maybe pygame can not host on streamlit, I will find a way to host my game</div>",unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: black;'>Source code in my github</h2>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: black;'>https://github.com/JedVu/snakegamewebcam</h4>", unsafe_allow_html=True)

else:
    st.markdown("<h1 style='text-align: center; color: black;'>You can catch me via:</h1>", unsafe_allow_html=True)

    col1, col2 = st.beta_columns((1,1))
    with col1:
        
        st.markdown("<div align = 'left'> </div>",unsafe_allow_html=True)
        st.markdown("<div align = 'right'>My mobile: </div>",unsafe_allow_html=True)
        st.markdown("<div align = 'left'> </div>",unsafe_allow_html=True)
        st.markdown("<div align = 'right'>My email: </div>",unsafe_allow_html=True)
        st.markdown("<div align = 'left'> </div>",unsafe_allow_html=True)
        st.markdown("<div align = 'right'>Or send me a message on Github: </div>",unsafe_allow_html=True)

    with col2:
        st.markdown("<div align = 'left'> </div>",unsafe_allow_html=True)
        st.markdown("<div align = 'left'>+84 909 365 518</div>",unsafe_allow_html=True)
        st.markdown("<div align = 'left'> </div>",unsafe_allow_html=True)
        
        st.write("[jed.cksoft@gmail.com](jed.cksoft@gmail.com)")
        st.write("[Jed Github](https://github.com/JedVu/snakegamewebcam)")
    
