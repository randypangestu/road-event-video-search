def get_text_feature(text, clip):
  search_query=text
  # Encode and normalize the search query using CLIP
  with torch.no_grad():
    text_features = model.encode_text(clip.tokenize(search_query).to(device))
    text_features /= text_features.norm(dim=-1, keepdim=True)
  return text_features
  
text1 = 'water flood'
text2 = 'disaster'
frame = Image.open('truck.png')
feat1 = get_text_feature(text1, clip)
feat2 = get_text_feature(text2, clip)
feat1 @ feat2.T
