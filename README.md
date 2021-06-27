# snakegamewebcam
*** This project, I don't want to use the corrdinates of PoseNet to get corr of Left/Right Wrist. I trained the skeleton pose to make the project a bit harder.

## Snake game with Body gesture
### Detail product:
    https://share.streamlit.io/jedvu/snakegamewebcam/main/main_streamlit.py
### Youtube:
    https://www.youtube.com/watch?v=PYaNoSkl5GE
    
## Guildline to install this repo
1. pip install library needed

2. pip install posenet

3. Copy replace posenet folder here to ..\miniconda3\envs\<name of your environment>\Lib\site-packages\

    (because I modified something in posenet original)

4. Extracted posenet model (in case you can not download posenet to your PC)

    Copy _model folder to your root app folder

    https://drive.google.com/file/d/1nXaHY2I0ydlN7NMMGt1x90grqnVcEBQi/view?usp=sharing

5. Edit link to predict model (if needed)

6. run main.py



