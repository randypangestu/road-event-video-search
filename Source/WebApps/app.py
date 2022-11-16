import os
os.system("pip freeze")
import cv2
from PIL import Image
import clip
import torch
import math
import numpy as np
import torch
import datetime
import gradio as gr


# Load the open CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)  
   

def inference_image(video, image):
# The frame images will be stored in video_frames
  video_frames = []
  # Open the video file
  
  capture = cv2.VideoCapture(video)
  fps = capture.get(cv2.CAP_PROP_FPS)
  
  current_frame = 0
  # Read the current frame
  ret, frame = capture.read()
  while capture.isOpened() and ret:
      ret,frame = capture.read()
      print('Read a new frame: ', ret)
      current_frame += 1
      if ret:
        video_frames.append(Image.fromarray(frame[:, :, ::-1]))

  
  # Print some statistics
  print(f"Frames extracted: {len(video_frames)}")

  # You can try tuning the batch size for very large videos, but it should usually be OK
  batch_size = 256
  batches = math.ceil(len(video_frames) / batch_size)
  
  # The encoded features will bs stored in video_features
  video_features = torch.empty([0, 512], dtype=torch.float16).to(device)
  
  # Process each batch
  for i in range(batches):
    print(f"Processing batch {i+1}/{batches}")
  
    # Get the relevant frames
    batch_frames = video_frames[i*batch_size : (i+1)*batch_size]
    
    # Preprocess the images for the batch
    batch_preprocessed = torch.stack([preprocess(frame) for frame in batch_frames]).to(device)
    
    # Encode with CLIP and normalize
    with torch.no_grad():
      batch_features = model.encode_image(batch_preprocessed)
      batch_features /= batch_features.norm(dim=-1, keepdim=True)
  
    # Append the batch to the list containing all features
    video_features = torch.cat((video_features, batch_features))
  
  # Print some stats
  print(f"Features: {video_features.shape}")

  loaded_image = preprocess(image).unsqueeze(0).to(device)
  display_heatmap=False
  display_results_count=1
  # Encode and normalize the search query using CLIP
  with torch.no_grad():
    image_features = model.encode_image(loaded_image)
    image_features /= image_features.norm(dim=-1, keepdim=True)

  # Compute the similarity between the search query and each frame using the Cosine similarity
  similarities = (100.0 * video_features @ image_features.T)
  values, best_photo_idx = similarities.topk(display_results_count, dim=0)

  for frame_id in best_photo_idx:
    frame = video_frames[frame_id]
    # Find the timestamp in the video and display it
    seconds = round(frame_id.cpu().numpy()[0]/fps)
  return frame,f"Found at {str(datetime.timedelta(seconds=seconds))}"
 

def inference(video, text):
  # The frame images will be stored in video_frames
  video_frames = []
  # Open the video file
  
  capture = cv2.VideoCapture(video)
  fps = capture.get(cv2.CAP_PROP_FPS)
  
  current_frame = 0
  # Read the current frame
  ret, frame = capture.read()
  while capture.isOpened() and ret:
      ret,frame = capture.read()
      print('Read a new frame: ', ret)
      current_frame += 1
      if ret:
        video_frames.append(Image.fromarray(frame[:, :, ::-1]))

  
  # Print some statistics
  print(f"Frames extracted: {len(video_frames)}")
  
  
  # You can try tuning the batch size for very large videos, but it should usually be OK
  batch_size = 256
  batches = math.ceil(len(video_frames) / batch_size)
  
  # The encoded features will bs stored in video_features
  video_features = torch.empty([0, 512], dtype=torch.float16).to(device)
  
  # Process each batch
  for i in range(batches):
    print(f"Processing batch {i+1}/{batches}")
  
    # Get the relevant frames
    batch_frames = video_frames[i*batch_size : (i+1)*batch_size]
    
    # Preprocess the images for the batch
    batch_preprocessed = torch.stack([preprocess(frame) for frame in batch_frames]).to(device)
    
    # Encode with CLIP and normalize
    with torch.no_grad():
      batch_features = model.encode_image(batch_preprocessed)
      batch_features /= batch_features.norm(dim=-1, keepdim=True)
  
    # Append the batch to the list containing all features
    video_features = torch.cat((video_features, batch_features))
  
  # Print some stats
  print(f"Features: {video_features.shape}")
 
 
  search_query=text
  display_heatmap=False
  display_results_count=1
  # Encode and normalize the search query using CLIP
  with torch.no_grad():
    text_features = model.encode_text(clip.tokenize(search_query).to(device))
    text_features /= text_features.norm(dim=-1, keepdim=True)

  # Compute the similarity between the search query and each frame using the Cosine similarity
  similarities = (100.0 * video_features @ text_features.T)
  values, best_photo_idx = similarities.topk(display_results_count, dim=0)


  for frame_id in best_photo_idx:
    frame = video_frames[frame_id]
    # Find the timestamp in the video and display it
    seconds = round(frame_id.cpu().numpy()[0]/fps)
  return frame,f"Found at {str(datetime.timedelta(seconds=seconds))}"
  
title = "Video Search"
description = "Gradio demo for using OpenAI's CLIP to search inside videos. To use it, simply upload your video and add your text. Read more at the links below."
article = "<p style='text-align: center'><a href='https://github.com/haltakov/natural-language-youtube-search' target='_blank'>Github Repo</a></p>"

examples=[['../Data/car_crash.mp4','car crash'], ['../Data/football.mp4', 'the ball goes into the net'], ['../Data/pedestrian.mp4', 'a man with gray winter jacket']]
gr.Interface(
    inference, 
    ["video","text"], 
    [gr.outputs.Image(type="pil", label="Output"),"text"],
    title=title,
    description=description,
    article=article,
    examples=examples
    ).launch(debug=True,enable_queue=True,share=True)

# gr.Interface(
#     inference, 
#     ["video","image"], 
#     [gr.outputs.Image(type="pil", label="Output"),"text"],
#     title=title,
#     description=description,
#     article=article,
#     examples=examples
#     ).launch(debug=True,enable_queue=True,share=True)