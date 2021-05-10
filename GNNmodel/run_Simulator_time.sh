#!/bin/bash

cd /pylon5/eg5fp3p/rachels/angranl_stuff/GN_NeuronTransport/
source activate /pylon5/eg5fp3p/rachels/angranl_stuff/neurongeo

# ? Pipe simulator: debug setting
# * Training
python main_Simulator_time.py --train --shuffle-dataset --data-fname ./data/Pipe/PipeDataAll_17_test/PipeDataSmall_17_time.h5 --checkpt-path ./logs/Pipe/checkpoints_GNtime_debug/  --log-interval 1 --log-file ./logs/Pipe/checkpoints_GNtime_debug/train.log --batch-size 4 --epochs 20 --num-worker 8 --num-layers 10 --num-hidden 128 --lr 0.005 --lr-gama 0.8 --lr-stepsize 3 

python main_Simulator_time.py --train --shuffle-dataset --data-fname ./data/Pipe/PipeDataAll_17_test/PipeDataSmall_17_time.h5 --checkpt-path ./logs/Pipe/checkpoints_GNtime_debug/  --log-interval 1 --log-file ./logs/Pipe/checkpoints_GNtime_debug_teacher0.5/train.log --batch-size 4 --epochs 20 --num-worker 8 --num-layers 10 --num-hidden 128 --lr 0.001 --lr-gama 0.8 --lr-stepsize 3

# * Predict
python main_Simulator_time.py --predict --shuffle-dataset --folds 5 --id-fold 0 --seed 1 --data-fname ./data/Pipe/PipeDataAll_17_test/PipeDataSmall_17_time.h5 --checkpt-fname ./logs/Pipe/checkpoints_GNtime_debug/pipe_epoch_19.pth --predict-path ./logs/Pipe/prediction_GNtime_debug/ --log-interval 1 --log-file ./logs/Pipe/checkpoints_GNtime_debug/predict.log --batch-size 128 --epochs 20 --num-worker 4 --num-layers 10 --num-hidden 128

python main_Simulator_time.py --predict --shuffle-dataset --folds 5 --id-fold 0 --seed 1 --data-fname ./data/Pipe/PipeDataAll_17_test/PipeDataSmall_17_time.h5 --checkpt-fname ./logs/Pipe/checkpoints_GNtime_debug/pipe_epoch_19.pth --predict-path ./logs/Pipe/prediction_GNtime_debug_teacher0.5/ --log-interval 1 --log-file ./logs/Pipe/checkpoints_GNtime_debug_teacher0.5/predict.log --batch-size 128 --epochs 20 --num-worker 4 --num-layers 10 --num-hidden 128

# ? Pipe simulator: test real data
# * Training

# !DEBUG
python main_Simulator_time.py --train --shuffle-dataset --data-fname ./data/Pipe/PipeDataAll_17_test/PipeDataAll_17_time_0.h5 --checkpt-path ./logs/Pipe/checkpoints_GNtime_debug_small/  --log-interval 1 --log-file ./logs/Pipe/checkpoints_GNtime_debug_small/train.log --batch-size 4 --epochs 20 --num-worker 8 --num-layers 10 --num-hidden 128 --lr 0.001 --lr-gama 0.8 --lr-stepsize 3 


python main_Simulator_time.py --train --shuffle-dataset --data-fname ./data/Pipe/PipeDataAll_17_test/PipeDataAll_17_time_1.h5 --checkpt-path ./logs/Pipe/checkpoints_GNtime_debug/  --log-interval 1 --log-file ./logs/Pipe/checkpoints_GNtime_debug/train.log --batch-size 16 --epochs 20 --num-worker 8 --num-layers 10 --num-hidden 128 --lr 0.001 --lr-gama 0.8 --lr-stepsize 3 

python main_Simulator_time.py --train --shuffle-dataset --data-fname ./data/Pipe/PipeDataAll_17_test/PipeDataAll_17_time_0.h5 --checkpt-path ./logs/Pipe/checkpoints_GNtime_debug_sameBC/  --log-interval 1 --log-file ./logs/Pipe/checkpoints_GNtime_debug_sameBC/train.log --batch-size 4 --epochs 20 --num-worker 8 --num-layers 10 --num-hidden 128 --lr 0.001 --lr-gama 0.8 --lr-stepsize 3 


# * Prediction

# !DEBUG
python main_Simulator_time.py --predict --shuffle-dataset --folds 5 --id-fold 0 --seed 1 --data-fname ./data/Pipe/PipeDataAll_17_test/PipeDataAll_17_time_0.h5 --checkpt-fname ./logs/Pipe/checkpoints_GNtime_debug_small/pipe_epoch_19.pth --predict-path ./logs/Pipe/prediction_GNtime_debug_small/ --log-interval 1 --log-file ./logs/Pipe/checkpoints_GNtime_debug_small/predict.log --batch-size 128 --epochs 20 --num-worker 4 --num-layers 10 --num-hidden 128

python main_Simulator_time.py --predict --shuffle-dataset --folds 5 --id-fold 0 --seed 1 --data-fname ./data/Pipe/PipeDataAll_17_test/PipeDataSmall_17_time.h5 --checkpt-fname ./logs/Pipe/checkpoints_GNtime_debug/pipe_epoch_19.pth --predict-path ./logs/Pipe/prediction_GNtime_debug_teacher0.5/ --log-interval 1 --log-file ./logs/Pipe/checkpoints_GNtime_debug_teacher0.5/predict.log --batch-size 128 --epochs 20 --num-worker 4 --num-layers 10 --num-hidden 128

python main_Simulator_time.py --predict --shuffle-dataset --data-fname ./data/Pipe/PipeDataAll_17_test/PipeDataAll_17_time_0.h5 --checkpt-fname ./logs/Pipe/checkpoints_GNtime_debug_sameBC/pipe_epoch_19.pth  --log-interval 1 --predict-path ./logs/Pipe/prediction_GNtime_debug_sameBC/ --log-file ./logs/Pipe/checkpoints_GNtime_debug_sameBC/predict.log --batch-size 4 --epochs 20 --num-worker 8 --num-layers 10 --num-hidden 128 --lr 0.001 --lr-gama 0.8 --lr-stepsize 3 
