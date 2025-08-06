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

print('===============================================================================')
print('= Starting Process ============================================================')
print('===============================================================================')

def predict(img, resnet_model):
	y = resnet_model.predict(img)
	y = decode_predictions(y, top=1)[0]
	return y[0][1]

def process(img_path, resnet_model):
	img = load_img(img_path, target_size = (224, 224))
	img = img_to_array(img)
	img = np.expand_dims(img, axis=0)
	img = preprocess_input(img)
	
	start = time.time()
	y = predict(img, resnet_model)
	return y, time.time() - start


print('===============================================================================')
print('= Load ResNet50 Pre-trained on ImageNet =======================================')
print('===============================================================================')

metrics = pm.PerformanceMetrics()
model   = ResNet50(weights='imagenet')

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
	metrics.add_prediction(latency, file.name, os.path.getsize(file_path), y)

print('===============================================================================')
print('= Save Results ================================================================')
print('===============================================================================')

metrics.save_predictions()
metrics.save_metrics()
metrics.save_cdf()
