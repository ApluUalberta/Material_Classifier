# NVIDIA Jetson Xavier NX Material Classifier Repository

## System Requirements on Windows 10 (Tensorflow onnx, and Keras)

### Versions of tools
* Anaconda Version: 4.8.3
* Pip Version: 20.0.2
* Python Version: 3.7.3


#### Tensorflow GPU Installation
* Step 1: Install a 2.x version of tensorflow<br />

```bash
  pip install tensorflow
```
            *        In our case, at this time, the tensorflow version used was version 2.2.0
<br />

* Step 2: Have a CUDA enabled NVIDIA GPU card with CUDA compute capability of 3.5 or higher. Check here:<br />

    * [NVIDIA GPU Compute Capability list](https://developer.nvidia.com/cuda-gpus)

* Step 3: Check to make sure that your NVIDIA GPU Driver is 418.x or higher with the following command:
```bash
nvidia-smi
```
        * you will see the driver version in the top middle

* Step 4: Download CUDA Toolkit 10.1 (sub-points before doing so) <br />
    * MAKE SURE THAT YOU USE CUDA v10.1
    * CUPTI is a needed software that is shipped with CUDA Toolkit
* Step 5: Download cuDNN SDK (Version 7.6.4 or version 7.6.5)
    * Once downloaded, you will need to merge the cuDNN include, library, and lib directories with your CUDA Toolkit 10.1 include, lib, and bin directories.
        * We did this by just manually pasting the dll, include headers, and .lib files in the corresponding directories (no auto merging needed)

* Step 6: Set your PATH Variables to point to the proper CUDA Toolkit version (done in anaconda prompt terminal)
```bash
SET PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\bin;%PATH%
SET PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\extras\CUPTI\lib64;%PATH%
SET PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\include;%PATH%
SET PATH=C:\tools\cuda\bin;%PATH%
```
    * For us, our toolkit path was in the given location, so this may vary depending on where your version 10.1 toolkit is placed.
* Step 7: Download keras, onnxtokeras, cv2, numpy, onnx, tqdm, matplotlib etc <br />

```bash
pip install keras
pip install onnx
pip install onnxtokeras
pip install cv2
pip install numpy
pip install tqdm
pip install matplotlib
```
* It is important to note that these are examples of external modules used in the program that could be different depending on what your system already possesses. If running the program and getting unresolved module imports, try a pip install
* The keras version that we used is 2.3.0-tf

# Jetson Xavier NX Progress

## [PyTorch Installation process here with notes right under](https://forums.developer.nvidia.com/t/pytorch-for-jetson-nano-version-1-5-0-now-available/72048)
* Make sure that you install the correct jetpack version or this will result in a libcudart error
* When following the torchvision setup, make sure that you use the desired torch package that you installed, and use
```bash
sudo python3 setup.py.install
```
    * Avoid using pip3 install 'pillow<7' as this is not necessary for python 3 users and follow the compatibility chart.
## To run a python script on boot:
* Step 1: Find the absolute path to your script and COPY THE COMMAND OUTPUT
```bash
readlink -f file.txt
```
* Step 2: Open crontab
```bash
crontab -e
```
* Step 3: scroll all the way down to the bottom and put the following
```bash
@reboot paste/path/here
```
* Step 4: Save and exit
* Step 5: Reboot the machine
```bash
sudo reboot
```

# Troubles with the setup (fixed but possible with other systems)
## We ran into a large problem with allowing the GPU to allow memory growth and not being able to find the Convolutional algorithm. Click the link below to learn more about the multiple solutions
## [Click this link to read on the error](https://github.com/tensorflow/tensorflow/issues/24496)
We tried the following options at the beginning of the program to remedy the given error:
```bash
config.gpu_options.allow_growth = True
```
* This Solution had no effect on the error for our Tensorflow version

```bash
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0],True)
```
* This solution resulted in an initialization error where we were not allowed to edit physical devices upon initialization

## The Solution was to use the following OS line in the beginning:
```bash
os.environ['TF_KERAS'] = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
```
source: [Scroll down to Bensuperpc's answer](https://stackoverflow.com/questions/53698035/failed-to-get-convolution-algorithm-this-is-probably-because-cudnn-failed-to-in)

# Coming tasks
## More difficult and important tasks

* [ ] Send the predictions from the NX back to the computer via ethernet
* [ ] Uploading Files to the Jetson Xavier NX via Ethernet and process predictions accordingly
* [ ] Shaping input data accordingly (More difficult and requires some theory)

## UI Tasks - Terminal first - Needs some collaboration
- [ ] Allow the input of different onnx models into the code system 
- [ ] Allow the image sizes to be altered
- [ ] Enable training toggle
    - [ ] Enable Data augmentation for training toggle
    - [ ] Enable L2 regression toggle
- [ ] Configure Category directories
    - [ ] changing counts accordingly


All rights Reserved to Metrized Consulting Inc.