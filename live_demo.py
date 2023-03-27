import cv2
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
from model_arq import Model
import pymongo
import io

client = pymongo.MongoClient("0.0.0.0:27017")
db = client["database"]
images = db["images"]

to_tensor = ToTensor()
image_path = "person.webp"

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


model = torch.load("Face_RecognitionV2.plt", map_location=device)

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
    frame = cv2.imread(image_path)
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
    cv2.imwrite("face_det.jpg", frame)
    im = Image.open("face_det.jpg")
    img_bytes = io.BytesIO()
    im.save(img_bytes, format='JPEG')
    image = {
        "data": img_bytes.getvalue()
    }
    img_id = images.insert_one(image).inserted_id

cam_test()
