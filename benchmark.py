import os
import time
import numpy as np
import tensorflow as tf
from pathlib import Path
import PerformanceMetrics as pm
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications.resnet50 import ResNet50, decode_predictions

def process(img_path, resnet_model):
	img = load_img(img_path, target_size = (224, 224))
	img = img_to_array(img)
	img = np.expand_dims(img, axis=0)
	img = preprocess_input(img)
	
	start = time.time()
	y = resnet_model.predict(img)
	y = decode_predictions(y, top=1)[0]
	return y, time.time() - start

#===============================================================

print('===============================================================================')
print('= Load ResNet50 Pre-trained on ImageNet =======================================')
print('===============================================================================')
model = ResNet50(weights='imagenet')

latencies = []
data_size = []

print('===============================================================================')
print('= Start Benchmark =============================================================')
print('===============================================================================')
i=0
directory = Path('./dataset')
for file in directory.glob("*.jpg"):
	if i > 10:
		break
	file_path = file.resolve()
	y, latency = process(file_path, model)
	print(f"Image: {file.name} Predicted: {y} Latency: {latency}")
	i = i + 1

	latencies.append(latency)
	data_size.append( os.path.getsize(file_path) )

#print('===============================================================================')
#print('= Print Results ===============================================================')
#print('===============================================================================')

metrics = pm.PerformanceMetrics(latencies, data_size)
metrics.print_summary()