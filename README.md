# FaceReco
Ok you, listen, fast and easy so pay atention. You have two ways of running this stupid thing.

## FIRST
I gave you a dockerfile, you run  
```
sudo docker build -t "whatever-name" .
```
This creates the Image, then you simply run the image creating a container but copy this or won't work.  
```
sudo docker run --name "your-name" --env="DISPLAY" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" --device /dev/video0 "name-of-your-image"  
```
And you have it, enjoy. To stop running the webcam just press the Key ESC and it will close. If you don't have webcam doesn't work but I think you are not that stupid.


## SECOND

You pull the Image from my profile in Docker Hub, simple, run  
```
sudo docker pull mrajoy/test1
```
Now you have stolen my Image so you run the same code of above but I will put it here again because I asume you are to lazy to read.  
```
sudo docker run --name "your-name" --env="DISPLAY" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" --device /dev/video0 "name-of-your-image" 
```

## WARNINGS
I only test this in Ubuntu so I don't know if in windows works. I really don't care so if it doesn't do it yourself I guess.  
And another thing, I don't know but maybe the container doesn't work if you poweroff or reboot the pc and you have to do the docker run again, the Image is still fine. Just so you know.

Get out of here.
