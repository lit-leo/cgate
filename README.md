# ChannelGatedCL
This is an unofficial PyTorch implementation of the paper 
"Conditional Channel Gated Networks for Task-Aware Continual Learning"
by Abati et al. (https://arxiv.org/abs/2004.00070).

At the moment, the code implements the task-incremental setting only. I have put some work into 
implementing class-incremental setup, but it is not finished yet. 

I have tried to follow the paper as close as possible, 
but maybe there are some details I missed/got wrong. So please use this implementation with care and 
feel free to ask questions and open PRs.

## Requirements
- Python 3.8
- PyTorch 1.7.1 (was also tested on 1.5.0, may work on some older versions)

The rest required modules are listed in requirements.txt

## How to run
1. Set up a proper configuration by editing config/cfg.py 
   
2. Run the following command to train the model

>    make train 

3. All tensorboard logs, checkpoints, and task-incremental accuracies
   will be stored in results/*experiment_tag* 
   
To visualize the sparseness of the model and see how many kernels were frozen by each task, 
check Sparse_visualization.ipynb

## Checkpoints

Due to some differences in implementation, the hyperparameters for ResNet18 mentioned in the paper didn't work well. I changed the optimizer and performed upscaling + augmentations to get better quality. 

You can download checkpoint with the best results for ResNet18 so far [here](https://yadi.sk/d/RF2KW59DVh-kHQ).
For consistency with other experiments, you may unzip it to the *results* folder. 

## Notes
There are a couple of differences between my and the original authors' implementation:

- To speed up the computation, it is assumed that the batch contains elements from one particular task only.

- I also observed a strange sparsity pattern for all values of lambda_sparse I used (from 0.1 to 0.5). There is free kernels attrition in the first couple of layers when the first 2-3 tasks use all available capacity, but later layers show underuse. This may be a result of the bug, so let me know if you find one :)
