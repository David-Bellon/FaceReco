# FaceReco
Ok you, listen, fast and easy so pay atention. You have two ways of running this stupid thing.

## FIRST
I gave you a dockerfile, you run  
'''
sudo docker build -t "whatever-name" .
'''
This creates the Image, then you simply run the image creating a container but copy this or won't.  
'''
sudo docker run --name "your-name" --env="DISPLAY" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" --device /dev/video0 test1
'''

