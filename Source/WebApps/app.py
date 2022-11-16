import gradio as gr
from inference import *

title = "Video Search"
description = "Gradio demo for using OpenAI's CLIP to search inside videos. To use it, simply upload your video and add your text. Read more at the links below."
article = "<p style='text-align: center'><a href='https://github.com/haltakov/natural-language-youtube-search' target='_blank'>Github Repo</a></p>"

examples=[['../Data/car_crash.mp4','car crash'], ['../Data/football.mp4', 'the ball goes into the net'], ['../Data/pedestrian.mp4', 'a man with gray winter jacket']]
gr.Interface(
    inference_text, 
    ["video","text"], 
    [gr.outputs.Image(type="pil", label="Output"),"text"],
    title=title,
    description=description,
    article=article,
    examples=examples
    ).launch(debug=True,enable_queue=True,share=True)