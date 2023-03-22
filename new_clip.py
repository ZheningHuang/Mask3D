import numpy as np
import torch
from pkg_resources import packaging
import clip
import cv2
import os
from PIL import Image
import glob
from PIL import Image, ImageDraw, ImageFont
import os


clip.available_models()
model, preprocess = clip.load("ViT-B/32")
model.cuda().eval()
input_resolution = model.visual.input_resolution
context_length = model.context_length
vocab_size = model.vocab_size

print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
print("Input resolution:", input_resolution)
print("Context length:", context_length)
print("Vocab size:", vocab_size)


folder_path = "/home/zh340/everydaycounts/hpc-work/NIPS2023/Mask3D_clean/project_2d_with_bg/['scene0011_00']/"

# Get a list of all files in the folder
file_list = os.listdir(folder_path)

file_dict = {}

# iterate through each file name
for file_name in file_list:
    # split the file name into segments using "_"
    segments = file_name.split("_")
    # extract the index from the second segment
    index = int(segments[1])
    # add the file name to the list for that index in the dictionary
    if index in file_dict:
        file_dict[index].append(file_name)
    else:
        file_dict[index] = [file_name]

# create a list of lists sorted by index
result = [[key, sorted(file_dict[key])] for key in sorted(file_dict.keys())]

classes = ["bathtub" ,"bed", "bookshelf", "cabinet" ,"chair", "counter", "curtain", "desk", "door", "floor", "picture", "refrigerator", "shower", "curtain", "sink", "sofa", "table", "toilet", "wall", "window"]

# classes.append("nothing")

text_descriptions = [f"This is a point cloud projection of a {label}" for label in classes]
text_tokens = clip.tokenize(text_descriptions).cuda()
with torch.no_grad():
    text_features = model.encode_text(text_tokens).float()

for i in range(70):
    image_list = []
    image_list_non_tensor = []
    file_list = result[i][1]
    # print(file_list)
    # Loop over all files in the folder
    for filename in file_list:
        image_path = os.path.join(folder_path, filename)
        # Check if the file is an image file (e.g. PNG, JPEG)
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
            # Read the image file into a numpy array
            image = Image.open(image_path).convert("RGB")
            # Convert the color format from BGR to RGB
            # Append the image to the list
            image_list_non_tensor.append(image)
            image_list.append(preprocess(image))
    image_input = torch.tensor(np.stack(image_list)).cuda()
    with torch.no_grad():
        image_features = model.encode_image(image_input).float()
    
    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    top_probs, top_labels = text_probs.cpu().topk(1, dim=-1)


    top_labels = top_labels.squeeze()

    # use the tensor to query the list
    predict = [classes[i] for i in top_labels]

    # Set the text color
    color = (255, 0, 0)

    # Define the output directory
    output_dir = "with_background"

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print("----------")
    print(image_list_non_tensor)
    print(predict)

    # Loop through the image and text lists
    for count, image in enumerate(image_list_non_tensor):

        # Create a drawing context
        draw = ImageDraw.Draw(image)

        # Define the text to add
        text = predict[count]

        # Define the position to place the text
        x, y = 10, 10

        # Add the text to the image
        draw.text((x, y), text, size= 30, fill=color)

        # Save the modified image to the output directory
        image_filename = os.path.basename(file_list[count])
        output_path = os.path.join(output_dir, image_filename)
        image.save(output_path)

    print("proposal", i)
    print(predict)


