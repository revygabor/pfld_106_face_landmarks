import time
import os
import argparse

import numpy as np
import cv2
import dlib
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
face_detector = dlib.get_frontal_face_detector()

def main(video_path, res_path, backbone='v3'):
  if backbone == "v2":
    from models.pfld import PFLDInference, AuxiliaryNet
    model_path=r'checkpoint/v2/v2.pth'
  elif backbone == "v3":
    from models.mobilev3_pfld import PFLDInference, AuxiliaryNet
    model_path=r'checkpoint/v3/v3.pth'
  # elif backbone == "ghost":
  #   from models.ghost_pfld import PFLDInference, AuxiliaryNet
  elif backbone == "lite":
    from models.lite import PFLDInference, AuxiliaryNet
    model_path=r'checkpoint/lite/lite.pth'
  else:
    raise ValueError("backbone is not implemented")
    exit()

  checkpoint = torch.load(model_path, map_location=device)
  plfd_backbone = PFLDInference().to(device)
  plfd_backbone.load_state_dict(checkpoint['plfd_backbone'], strict=False)

  
  video_capture = cv2.VideoCapture(video_path)
  if not video_capture.isOpened():
      print("Error: video can't be opened.")
  else:
      print("The video source has been opened correctly...")

  start = time.time()
  image_index = 0
  while True:
    ret, frame = video_capture.read()
    if not ret:
      break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray, 0)
    if len(faces) == 0:
      cv2.imwrite(f'{res_path}{os.sep}{image_index:05}.jpg', frame)
      image_index += 1
      continue
    
    face = faces[0]
    left = face.left()
    top = face.top()
    right = face.right()
    bottom = face.bottom()

    image_crop = frame[top:bottom, left:right]
    
    # 网络输入是BGR格式的图片
    img_crop_resized = cv2.resize(image_crop, (112, 112))
    image_data = img_crop_resized.transpose(2, 0, 1)[np.newaxis].astype(np.float32) / 255


    plfd_backbone.eval()
    plfd_backbone = plfd_backbone.to(device)
    image_data = torch.from_numpy(image_data).float().to(device)

    with torch.no_grad():
      output = plfd_backbone(image_data)

    landmarks = output[1].reshape(-1, 2)
    landmarks[:, 0] = landmarks[:, 0] * image_crop.shape[1]
    landmarks[:, 1] = landmarks[:, 1] * image_crop.shape[0]
    img_copy = frame.copy().astype(np.uint8)
    for (x, y) in landmarks:
      cv2.circle(img_copy, (int(x+left), int(y+top)), 2, (0, 0, 255), -1)
    cv2.rectangle(img_copy, (left, top), (right, bottom), (255, 0, 0), 2)
    img_copy = cv2.imwrite(f'{res_path}{os.sep}{image_index:05}.jpg', img_copy)
    image_index += 1

  t = (time.time() - start) / image_index
  print('average infer time: {:.4f}ms, FPS: {:.2f}'.format(t * 1000, 1 / t))

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Process video and draw PFLD landmarks + head detection.')
  parser.add_argument('video_path', help='path to the video')
  parser.add_argument('--res_dir', default=None,
                      help='directory path to store resulting images')
  parser.add_argument('--backbone', default='v3', choices=['v2', 'v3', 'lite'],
                      help='backbone of the network')
  args = parser.parse_args()

  video_path = args.video_path
  if not os.path.exists(video_path):
    print("File doesn't exist. Exiting.")
    exit()

  res_path = args.res_dir
  if res_path is None:
    res_path = os.path.splitext(os.path.basename(video_path))[0]
  os.makedirs(res_path, exist_ok=True)

  main(video_path, res_path, backbone=args.backbone)
