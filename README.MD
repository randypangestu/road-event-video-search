# Road Event Video Search

This repository provides the data, scripts, and demo examples for the Road Event Video Search WebApp.

## Deskripsi

**Road Event Video Search** is a system that combines textual and visual data to detect anomalies in vehicle and pedestrian traffic.

This system uses a **zero-shot learning** model trained on both visual and textual data. Due to the extensive and diverse training data, the model is highly generalizable and can be applied to various use cases.

In this project, we apply the **CLIP zero-shot learning** model to a specific case: traffic-related events. We categorize these events into the following types:

- Crashes  
- Traffic jams  
- Floods  
- Demonstrations


## Installation
### Run Directly on Colab (Option 1)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1bwXqydl9oEfYEzFwwZWLL92Me-EZJLit?usp=sharing)

Untuk mencoba WebApp ini bisa langsung mengunjungi link di atas untuk melakukan deployment WebApp di Colab. Jalankan cell yang ada secara berurutan dan nantinya akan muncul link Gradio yang bisa diakses secara public.

### Install via pip (Option 2)

```bash
cd Source/WebApps/
pip3 install -r requirements.txt
```

### Run With Python

```bash
cd Source/WebApps/
python3 Quiet_Wire_WA.py
```
