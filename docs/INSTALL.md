# Installation

Slighty modified from the CenterNet installation guide. Install the [DCNv2_latest](https://github.com/jinfagang/DCNv2_latest) externally instead of within the repository. Tested on Ubuntu 20.04 with Python3.8 and [PyTorch]((http://pytorch.org/)) 1.13.1. NVIDIA GPU(s) is(are) needed for both training and testing.


After install Anaconda:

0. [Optional but recommended] create a new environment. 

    ~~~
    conda create -n kgn-pro python=3.8
    ~~~
    And activate the environment.
    
    ~~~
    conda activate kgn-pro
    ~~~

1. Install pytorch. For pytorch 1.13.1:

    ~~~
    conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
    ~~~
    
    
3. Clone this repo:

    ~~~
    git clone git@github.com:WaitDerek/KGN-Pro.git 
    ~~~


4. Install the requirements

    ~~~
    pip install -r requirements.txt
    ~~~
    
5. Install and Compile Deformable Convolutional Networks V2 from [DCNv2_latest](https://github.com/jinfagang/DCNv2_latest). 

    ~~~
    # DCNv2=/path/to/clone/DCNv2
    git clone https://github.com/jinfagang/DCNv2_latest $DCNv2
    cd $DCNv2
    python setup.py install --user
    ~~~

6. The following actions are required to resolve the conflict because of the version shown in [realease](https://github.com/pyro-ppl/pyro/releases/tag/1.9.0)
    ~~~
    cd $PATH_TO_KGN-PRO_ENV/lib/python3.8/site-packages/pyro/optim
    sed -i '38s/_torch_scheduler_base = torch.optim.lr_scheduler.LRScheduler/_torch_scheduler_base = torch.optim.lr_scheduler._LRScheduler/' pytorch_optimizers.py
    ~~~

7. [Optional, only required if you are using extremenet or multi-scale testing] Compile NMS.

    ~~~
    cd ./src/lib/external
    make
    ~~~
