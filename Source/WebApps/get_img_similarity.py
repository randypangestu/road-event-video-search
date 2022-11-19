def get_text_sim_img(text, frame, clip):
  search_query=text
  # Encode and normalize the search query using CLIP
  batch_preprocessed = torch.stack([preprocess(frame)]).to(device)

  with torch.no_grad():
    text_features = model.encode_text(clip.tokenize(search_query).to(device))
    batch_features = model.encode_image(batch_preprocessed)
    print(text_features.shape, batch_features.shape)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    batch_features /= batch_features.norm(dim=-1, keepdim=True)


  return text_features @ batch_features.T

text1 = 'accident'
text2 = 'normal'
#frame = Image.open('truck_red.png')
frame = Image.open('truck.png')
feat1 = get_text_sim_img(text1, frame, clip)
print(text1, feat1)
feat2 = get_text_sim_img(text2, frame, clip)
print(text2, feat2)
