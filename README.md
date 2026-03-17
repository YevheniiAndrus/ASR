Automatic Speech Recognition

A simple RNN network to perfom automatic speech recognition commands from google commands dataset. 
Can be used for robotic applications that are supposed to accept voice commands like start, stop, go, left, right etc

How to run
1. Edit config.cfg file and provide dataset_path value with the location of google_speech_commands dataset. Dataset can be found here - https://www.kaggle.com/datasets/neehakurelli/google-speech-commands
2. Provide command list that system will be recognizing. Available commands - Yes", "No", "Up", "Down", "Left", "Right", "On", "Off", "Stop", "Go", "Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", and "Nine".
3. modify model's parameters if needed
4. Run in terminal python3 asr.py

After training has finished you will be able to see trainig loss and accuracy of the model as well as testing accuracy. The model itself will be saved as asr.keras and will be ready to use for inference
