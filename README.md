# Attention-modular

### This repo is for nlp final project. And here is a brief summary.In my report, I use code of tf-1.0. However, The implementation of tf-2.0 is similar with 1.0 and get comparable performance compared with tf-1.0, given same amount of data.  

# tf-2.0 implementation
#### tf-2.0 implementation for attention modular is my primary focus. I successfully implemented global attention mechanism with dot scoring for this project. 
#### A download script is provided to you so you only need to run script to download data.
#### The data pipeline is implemented in data generator. By doing this, it is possible to save memory to allow larger batch_size to use in deep neural network. This generator is not only work in tf.data pipeline but also in training data read-in stage. 
#### The class is re-written in keras class to keep gradient flow in close connection. 
#### piece-wise learning rate is introduced in training stage. For dropout setup, first eight epochs needs learning rate of 1. And reduced by half in following turns. We end up 12 epochs in total.
#### I also build a test pipeline to support evaluation in rolling manner. Perplex and BLEU are two supporting metrics in current implementations. For each batch, by_batch evaluation will perform. And after an epoch ends, the overall performance will be reported.
#### ________________________________________________________________________________________________________________________

# How to run tf-2.0 attention modular
#### 1. Download data by calling get_data.py, you may need to create a folder called "data". Use a different name will result in error
#### 2. To train, call train_mod.py. If you need to change parameter, please follow the format in python file.
#### 3. To test, call test.py. Make sure you get valid checkpoint and provide the path of checkpoint to test.py.
#### 4. Due to the fact that multiple students run gpu on mamba. In my testing environment, I'm not able to load and run model for all training dataset. How much can load in depends on how many tasks runing on single GPU card. So there is no guarantee that you can load whole dataset without OOM. To my best luck, I'm able to run with 50K records. To my worst luck, I can run only 500 records.

#### ________________________________________________________________________________________________________________________

# tf-1.0 implementation
#### tf-1.0 implementation of attention modular works in session run manner. Dot scoring is implemented for this project. I referred the code from https://github.com/dillonalaird/Attention and modified the code for this project. I used modified code to run experiment with the whole training dataset and get evaluation result accordingly. 
#### A download script is provided to you so you only need to run script to download data.
#### The code is being modified into python 3 language, as the running environment in my machine is python 3.
#### I modified optimizer and add piece-wise learning rate. This is not recommended by Tensorflow as the learning rate of SGD by default should be fixed。 But this is not what has been done in the paper so I modified it accordingly.
#### I re-write checkpoint setup to enable re-training after pause. So does the case for evaluation. A valid filepath will be verified firstly and then launch validation.
#### I modified sub-process to accommodate file processing and enable BLEU evaluation in linux environment.
#### I adapted encoding format to support error-free encoding in python 3 between byte and string.
#### ________________________________________________________________________________________________________________________

# How to run tf-1.0 attention modular
#### Make sure your tensorflow environment is 1.10.
#### 1. Download data by calling get_data.py, you may need to create a folder called "data". Use a different name will result in error
#### 2. To train, call main.py. If you need to change parameter, please follow the format in python file.
#### 3. To test, call main.py. Make sure you get valid checkpoint and provide the path of checkpoint to test.py.
#### example to test: python main.py --is_test=True --checkpoint ./checkpoint
#### Please check and change hyper-parameters accordingly as your need.
#### 4. This is the version I use to conduct experiment and report stats in my final report.



