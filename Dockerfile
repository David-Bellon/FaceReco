FROM python:3.10.6
RUN pip3 install opencv-python
RUN pip3 install pymongo
RUN pip3 install pandas
RUN pip3 install matplotlib
RUN pip3 install numpy --pre torch torchvision torchaudio --force-reinstall --index-url https://download.pytorch.org/whl/nightly/cu118
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN export DISPLAY=10.0
COPY live_demo.py /temp/
COPY model_arq.py /temp/
COPY Face_RecognitionV2.plt /
COPY person.webp /
CMD ["python3", "temp/live_demo.py"]
