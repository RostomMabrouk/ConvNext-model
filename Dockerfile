FROM mcr.microsoft.com/azureml/pytorch-1.10-ubuntu18.04-py37-cpu-inference:20220516.v3


RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 libgl1-mesa-glx -y
RUN apt-get update && apt-get install libgl1