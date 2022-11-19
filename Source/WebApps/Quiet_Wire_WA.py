import os
# os.system("pip freeze")
import cv2
from PIL import Image
import clip
import torch
import math
import numpy as np
import torch
import datetime
import gradio as gr
import time

# Query choices
choices_to_query = {
    "Crash": "vehicle crash",
    "Traffic Jam": "traffic jam",
    "Flood": "flash flood",
    "Demonstration": "crowd demonstration"
}

# Load the open CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def inference(video, query, advance_query, slider):
  # The frame images will be stored in video_frames
  video_frames = []

  # Open the video file
  capture = cv2.VideoCapture(video)
  fps = capture.get(cv2.CAP_PROP_FPS)
  total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
  
  current_frame = 1
  num_skip_frames = slider
  start = time.time()
  while capture.isOpened():
    if current_frame <= num_skip_frames:
      ret = capture.grab()
      current_frame += 1
      continue
    ret, frame = capture.read()
    current_frame = 1
    # print('Read a new frame: ', ret)
    if ret:
      video_frames.append(Image.fromarray(frame[:, :, ::-1]))
    else:
      break

  # Print some statistics
  print(f"Finished extracting frames in {time.time()-start} s")
  print(f"Frames extracted: {len(video_frames)}")
  
  # You can try tuning the batch size for very large videos, but it should usually be OK
  batch_size = 256
  batches = math.ceil(len(video_frames) / batch_size)
  
  # The encoded features will bs stored in video_features
  video_features = []
  
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
    video_features.append(batch_features)

  video_features = torch.cat(video_features, dim=0)
  
  # Print some stats
  print(f"Features: {video_features.shape}")
 
  search_query=query
  display_heatmap=False
  display_results_count=1

  # Encode and normalize the search query using CLIP
  if not isinstance(search_query, str):
    query_input = preprocess(Image.fromarray(search_query)).unsqueeze(0).to(device)
    with torch.no_grad():
      query_features = model.encode_image(query_input)
      query_features /= query_features.norm(dim=-1, keepdim=True)

  else:
    if advance_query != "":
      search_query = advance_query
    else:
      search_query = choices_to_query[search_query]

    with torch.no_grad():
      query_features = model.encode_text(clip.tokenize(search_query).to(device))
      query_features /= query_features.norm(dim=-1, keepdim=True)

  # Compute the similarity between the search query and each frame using the Cosine similarity
  similarities = (100.0 * video_features @ query_features.t())
  print(similarities.cpu().tolist())
  values, best_photo_idx = similarities.topk(display_results_count, dim=0)

  for frame_id in best_photo_idx:
    frame_id = frame_id.item()
    frame = video_frames[frame_id]
    # Find the timestamp in the video and display it
    frame_id += (frame_id+1)*num_skip_frames
    seconds = round(frame_id/fps)

  return frame, f"Found at {str(datetime.timedelta(seconds=seconds))} with similarity {values.item()}"

# Gradio UI
with gr.Blocks() as demo:
  gr.Markdown("Query based on text or image to search event in a video.")
  choices = list(choices_to_query.keys())
  with gr.Tab("Text Query"):
    with gr.Row():
      with gr.Column():
        text_video_comp = gr.Video(label="Input Video")
        simple_text_query_comp = gr.Radio(choices=choices, value=choices[0], label="Text Query")
        with gr.Accordion("Advance Query", open=False):
          advance_query = gr.Textbox(label="Advance Text Query")
        text_num_skip_frames_comp = gr.Slider(0, 100, value=0, step=1, label="Num Skip Frames")
        text_button = gr.Button("Search")
      with gr.Column():
        text_output = [gr.outputs.Image(type="pil", label="Output"), gr.Textbox(label="Output")]
  with gr.Tab("Image Query"):
    with gr.Row():
      with gr.Column():
        img_video_comp = gr.Video(label="Input Video")
        image_query_comp = gr.Image(label="Image Query")
        img_num_skip_frames_comp = gr.Slider(0, 100, value=0, step=1, label="Num Skip Frames")
        image_button = gr.Button("Search")
      with gr.Column():
        image_output = [gr.outputs.Image(type="pil", label="Output"), gr.Textbox(label="Output")]

  text_button.click(inference, inputs=[text_video_comp, simple_text_query_comp, advance_query, text_num_skip_frames_comp], outputs=text_output)
  image_button.click(inference, inputs=[img_video_comp, image_query_comp, advance_query, img_num_skip_frames_comp], outputs=image_output)

demo.launch(share=True, debug=True)