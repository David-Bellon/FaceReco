import cv2
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
from model_arq import Model

to_tensor = ToTensor()

device = torch.device("cuda")


model = torch.load("Face_RecognitionV2.plt", map_location=torch.device("cuda"))

def cam_test():
    cam = cv2.VideoCapture(0)
    width = cam.get(3)
    height = cam.get(4)
    model.eval()
    while True:
        rate, frame = cam.read()
        source = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image_s = cv2.resize(source, (32, 32))

        input_s = to_tensor(Image.fromarray(image_s)).to(device)

        input_s = input_s[None, :, :, :]

        output_s = model(input_s)

        tlc = int(output_s[0][0].item() * (width - 1))
        brc = int(output_s[0][1].item() * (width - 1))
        tlr = int(output_s[0][2].item() * (height - 1))
        brr = int(output_s[0][3].item() * (height - 1))
        frame = cv2.rectangle(frame, (tlc, tlr), (brc, brr), (0, 0, 255), 2)
        cv2.imshow("video", frame)
        key = cv2.waitKey(1)
        if key == 27:
            break

def photo_test():
    frame = cv2.imread("person.webp")
    width = frame.shape[1]
    height = frame.shape[0]
    source = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    image_s = cv2.resize(source, (32, 32))

    input_s = to_tensor(Image.fromarray(image_s)).to(device)

    input_s = input_s[None, :, :, :]

    output_s = model(input_s)

    tlc = int(output_s[0][0].item() * (width - 1))
    brc = int(output_s[0][1].item() * (width - 1))
    tlr = int(output_s[0][2].item() * (height - 1))
    brr = int(output_s[0][3].item() * (height - 1))

    frame = cv2.rectangle(frame, (tlc, tlr), (brc, brr), (0, 0, 255), 2)
    cv2.imshow("video", frame)
    cv2.waitKey(0)

photo_test()