Notebooks and functions used to try various classification tasks on processed multimodal vocal data.

These notebooks are structures so that they can be run in Google Colab given that the necessary folder structure is in Google Drive.

1. Running the code:
   Follow the step by step process outlines in the python notebooks. If interested in the internal workings of the
   functions used, look at the other python scripts containing them.

2. Data
   It is recommended that you work with the data within the data folder. Do not commit your data csvs. Add them to your gitignore file.

The preprocessed CSV files of the extracted features using the my Masters Thesis are provided in the Google Drive link provided to access the code in Colab (see below). This allows for the running of the notebooks if desired. The preprocess notebook is provided should there be need to preprocess other data. The original CSV files that are not preprocessed are provided in Google Drive as well.

These contain the full feature sets extracted from the raw data. This includes:
- metadata 
- phonation labels 
- Vocal Technique Error labels 
- Audio features extracted from audio recorded with a Behringer C-3 condenser microphone (mic)
- Audio features extracted from audio recorded with a 2017 MacBook Air (computer)
- Audio features extracted from audio recorded with a iPhone 14 Pro (phone)
- Video features extracted from video recorded with a 2017 MacBook Air (computer)
- Video features extracted from video recorded with a iPhone 14 Pro (phone)
- Raw biosignal data (PZT, EMG, EEG)

For further details, look at the associated Masters Thesis. 
https://drive.google.com/file/d/1HUONhfO8qYaY_pvhSrwu6N99NA8Z_QRQ/view?usp=sharing 

All the code is set up to run within Google Colab. 
The all code (scripts, notebooks) and data can also be viewed within Google Colab through this link: 
https://drive.google.com/drive/folders/1gX4fMXhi32mpUk2z7-yEH6NWgBUIGKls?usp=sharing

