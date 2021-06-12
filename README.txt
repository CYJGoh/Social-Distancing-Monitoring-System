Before running the GUI, please ensure all the dependencies have been installed.

Dependency 1: Python libraries
Requirement:
PyTorch (Ensure CUDA is running to process the input faster but, CPU will still work)
Streamlit
NumPy
PyYAML
Pandas
OpenCV
tqdm
Pillow
ffmpeg

Dependency 2: FFMPEG
1) Download the FFMPEG full build from this link https://www.gyan.dev/ffmpeg/builds/ffmpeg-git-full.7z
2) Extract the folder
3) Navigate to bin/ffmpeg.exe
4) Paste that file into this source code directory

After ensuring all dependencies are installed and updated, you may proceed to run the GUI.
To run: 
1) Open up Ananconda Prompt
2) Change your directory to this folder
3) Enter this command to run: streamlit run gui.py
4) The GUI shall be loaded in your default browser

To inference your images or videos, please ensure that your images and video files are placed in the images and videos folder respectively so that the system is able to find your file.
