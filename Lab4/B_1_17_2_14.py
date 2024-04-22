# python B_1_17_2_14.py Input PytorchOutput 4 mask.txt
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import time
import sys


def load_kernel(kernel_path):
    # Read kernel from file
    kernel_file = "mask.txt"
    kernel3d = []
    with open(kernel_file, "r") as f:
        lines = f.readlines()
        kernel_size = int(lines[0].strip())  # Read kernel size
        kernel_values = []
        for line in lines[1:]:
            values = line.split()
            parsed_values = []
            for value in values:
                if '/' in value:
                    numerator, denominator = map(int, value.split('/'))
                    parsed_values.append(numerator / denominator)
                else:
                    parsed_values.append(float(value))
            kernel_values.append(parsed_values)

        for i in range(3):  # each channel
            kernel3d.append(kernel_values)
    # Define convolution kernel
    kernel = torch.tensor(kernel3d)
    return kernel, kernel_size


def load_input_images(input_folder):
  input_files = sorted(os.listdir(input_folder))
  input_images = []
  for input_file in input_files:
    input_path = os.path.join(input_folder, input_file)
    input_images.append(Image.open(input_path))
  return input_images


def pytorch_convolution(input_images, kernel, batch_size=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using", device)
    transform = transforms.Compose([transforms.ToTensor()])
    pytorch_outputs = []

    timings = {'total_execution_time_ms': 0, 'batch_times_ms': []}

    for i in range(0, len(input_images), batch_size):

        batch_images = input_images[i:i+batch_size]
        batch_tensors = []
        for image in batch_images:
            image_tensor = transform(image).unsqueeze(0).to(device)
            batch_tensors.append(image_tensor)

        batch_tensor = torch.cat(batch_tensors, dim=0)

        # Apply padding
        padding = kernel.size(2) // 2  # Assuming square kernel
        batch_tensor_padded = F.pad(batch_tensor, (padding, padding, padding, padding), mode='constant', value=0)

        # Apply convolution
        batch_start_time = time.time()
        output = F.conv3d(batch_tensor_padded.unsqueeze(1), kernel.unsqueeze(0).unsqueeze(0).to(device))
        batch_end_time = time.time()

        # Convert output tensor to numpy array and normalize
        output_images = output.squeeze(1).cpu().numpy().transpose(0, 2, 3, 1)

        for out in output_images:
            pytorch_outputs.append(out)

        batch_time_ms = (batch_end_time - batch_start_time) * 1000
        timings['batch_times_ms'].append(batch_time_ms)

    timings['total_execution_time_ms'] = sum(timings['batch_times_ms'])

    return pytorch_outputs, timings

def save_output(pytorch_outputs, output_path):
  for i in range(len(pytorch_outputs)):
    # Save PyTorch output image as JPEG
    output_image_tosave = (pytorch_outputs[i] - pytorch_outputs[i].min()) / (pytorch_outputs[i].max() - pytorch_outputs[i].min()) * 255
    output_image_tosave = output_image_tosave.astype('uint8')
    if output_image_tosave.ndim == 3 and output_image_tosave.shape[2] == 1:
        output_image_tosave = output_image_tosave.squeeze(2)  # Remove the third dimension if it's 1
    output_image_tosave = Image.fromarray(output_image_tosave)
    output_image_tosave.save(os.path.join(output_path, f"pytorch_output_{i}.jpg"))



def profile(timings):
  print("Total execution time of F.conv3d (ms) :", timings['total_execution_time_ms'])
  for i, batch_time in enumerate(timings['batch_times_ms']):
    print(f"\t Batch {i+1} execution time (ms) : {batch_time}")

def main():
  if len(sys.argv) != 5:
      print("Usage: python B_1_17_2_14.py <input_folder> <output_folder> <batch_size> <mask_file>")
      return
  
  input_folder = sys.argv[1]
  pytorch_folder =sys.argv[2]
  batch_size = int(sys.argv[3])
  mask_file_path =  sys.argv[4]

  # Define convolution kernel
  kernel, kernel_size = load_kernel(mask_file_path)

  # Define batch size
  print("Loading Input Images...")
  input_images = load_input_images(input_folder)


  print("Processing...")
  pytorch_outputs, timings = pytorch_convolution(input_images, kernel, batch_size)

  print("Writing Output...")
  save_output(pytorch_outputs, pytorch_folder)

  profile(timings)


if __name__ == "__main__":
    main()

# TO RUN:
# python B_1_17_2_14.py Input PytorchOutput 4 mask.txt
