# CTCSpeechRecognition

Work in progress. Implementation of the <a href="https://github.com/baidu-research/warp-ctc/">Baidu Warp-CTC</a> using torch7. Feeds spectrogram data into a neural network using the Torch7 library, training itself with the CTC activation function.

Current implementation runs on CUDA 7.0.

To install torch7 follow the guide <a href="http://torch.ch/docs/getting-started.html">here</a>.

To install the Baidu warp-ctc library follow the guide at the end of the readme <a href="https://github.com/baidu-research/warp-ctc/README.md">here</a>.


To install CUDA (CUDA 7.0 is required):

Download the .run file of your platform <a href="https://developer.nvidia.com/cuda-toolkit-70">here</a>:

Example for Ubuntu 14.04:
```
wget http://developer.download.nvidia.com/compute/cuda/7_0/Prod/local_installers/cuda_7.0.28_linux.run
```
Then install using below commands.
```
sudo chmod +x ./cuda_7.0.28_linux.run
sudo ./cuda_7.0.28_linux.run
```
When prompted with this message type no. Highly recommended to install the latest driver from the nvidia website of your GPU.
```
Install NVIDIA Accelerated Graphics Driver for Linux-x86_64 346.46? ((y)es/(n)o/(q)uit):
```
Finally modify the .bashrc located at ~/.bashrc, including these lines at the end:
```
export PATH=/usr/local/cuda-7.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-7.0/lib64:$LD_LIBRARY_PATH
```
Restart the terminal for changes to take effect.

Other Dependencies can be installed via luarocks:
<a href="https://github.com/soumith/lua---audio">Audio Library for Torch</a>:
```
luarocks install https://raw.githubusercontent.com/soumith/lua---audio/master/audio-0.1-0.rockspec
```

<a href="https://github.com/torch/optim">Optim: numeric optimization package for Torch</a>:
```
luarocks install optim
```

<a href="https://github.com/Element-Research/rnn">rnn: Recurrent Neural Network library for Torch7's nn</a>:
```
luarocks install rnn
```

It is also suggested to update the following libraries:
```
luarocks install torch
luarocks install nn
luarocks install dpnn
```
For CUDA implementation:
```
luarocks install cutorch
luarocks install cunn
luarocks install cunnx
```
Main method located at AN4Test.lua.

Training data is currently the <a href="http://www.speech.cs.cmu.edu/databases/an4/">AN4 Audio database</a>. 

Within the AN4Test we specify the file path to the an4 dataset which can be downloaded <a href="http://www.speech.cs.cmu.edu/databases/an4/an4_raw.bigendian.tar.gz">here</a>. 

Extract the folder and give the filepath in the AN4Test.lua script. 

**Note that the data has to be converted into wav format, a bash script ConvertAN4ToWav.sh has been included to help with this (place this into the an4 directory then run).**
