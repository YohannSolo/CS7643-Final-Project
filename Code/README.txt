The code files are in the Code folder. There are files for QLoRA, LoRA, and LoftQ. The main.py file combines them so that all configurations of models tested can be run from one file using command line arguments.

To run the main.py file, these are the arguments to be specified:
python main.py [model_type] [rank] [model_dir (optional)] [train_dir (optional)] [save_dir (optional)]

For example, to run a QLoRA model with a rank of 8 using the default model, training, and save directories:
python main.py qlora 8


The required packages for this project can be found in the requirements.txt file.