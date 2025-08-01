import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print('===============================================================================')
print('= Load CIFAR-10 dataset =======================================================')
print('===============================================================================')
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Preprocess the data
x_train = x_train.astype('float32') / 255.0
x_test  = x_test.astype('float32') / 255.0

# One-hot encode the labels
y_train = to_categorical(y_train, 10)
y_test  = to_categorical(y_test, 10)

print('===============================================================================')
print('= Load ResNet50 Pre-trained on ImageNet =======================================')
print('===============================================================================')
base_model = ResNet50(weights='imagenet', 
                      include_top=False, 
                      input_shape=(32, 32, 3))

# Freeze the base model
base_model.trainable = False

print('===============================================================================')
print('= Build the classification model ==============================================')
print('===============================================================================')
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(1024, activation='relu'),
    Dense(10, activation='softmax')  
])

print('===============================================================================')
print('= Compile the model ===========================================================')
print('===============================================================================')
model.compile(optimizer=Adam(learning_rate=0.0001), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

print('===============================================================================')
print('= Train the model =============================================================')
print('===============================================================================')
model.fit(x_train, y_train, 
          batch_size=64, 
          epochs=10, 
          validation_data=(x_test, y_test))

print('===============================================================================')
print('= Evaluate the model ==========================================================')
print('===============================================================================')
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc}")
model.save('image_class.keras')
