#!/bin/bash

python3 plot_reconstruction.py --directory ../reconstruction/logs/dorsal_stream/ --model_names diffusion cnn linear --image_ids 14 21 30 19 0 22 34 29
python3 plot_reconstruction_gif.py --directory ../reconstruction/logs/dorsal_stream/ --model_names diffusion cnn linear --image_ids 14 21 30 19 0 22 34 29

python3 plot_reconstruction.py --directory ../reconstruction/logs/ventral_stream/ --model_names diffusion cnn linear --image_ids 33 25 66 2 42  63 71 99
python3 plot_reconstruction_gif.py --directory ../reconstruction/logs/ventral_stream/ --model_names diffusion cnn linear --image_ids 33 25 66 2 42  63 71 99

python3 plot_reconstruction.py --directory ../reconstruction/logs/ventral_stream/ --model_names diffusion cnn linear --image_ids 19 15 8 20  37 61 46 84