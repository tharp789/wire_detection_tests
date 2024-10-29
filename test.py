import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from PIL import Image

def segment_image(image, model):
    # Preprocess the image
    input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_image = torch.from_numpy(input_image).permute(2, 0, 1).float() / 255.0
    input_image = input_image.unsqueeze(0).to('cuda')
    # Perform segmentation
    with torch.no_grad():
        # output = model(input_image)
        output = model(input_image)['out'][0]
    output_predictions = output.argmax(0)
    return output, output_predictions


image = cv2.imread('up_view.png')
model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True).to('cuda')
model.eval()

output, output_predictions = segment_image(image,model)

palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
colors = (colors % 255).numpy().astype("uint8")

# plot the semantic segmentation predictions of 21 classes in each color
r = Image.fromarray(output_predictions.byte().cpu().numpy())
r.putpalette(colors)

plt.imshow(r)
plt.show()