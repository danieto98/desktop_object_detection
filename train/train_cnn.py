import tensorflow as tf
from tensorflow.keras import layers, models
import sys

# Get directory paths and nr. of epochs
data_dir = ""
model_dir = ""
EPOCHS = 1
if (len(sys.argv)) == 4:
	data_dir = sys.argv[1]
	model_dir = sys.argv[2]
	EPOCHS = int(sys.argv[3])
else:
	exit("Usage: train_cnn.py input_dataset_path model_output_path nr_epochs")

# Set parameters
IMG_WIDTH = 220
IMG_HEIGHT = 220
BATCH_SIZE = 30

# Initialize image data generators by splitting into training and testing data
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, validation_split=0.25)
train_generator = train_datagen.flow_from_directory(data_dir, target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE, class_mode='categorical', subset='training')
test_generator = train_datagen.flow_from_directory(data_dir, target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE, class_mode='categorical', subset='validation')

# Create CNN model
model = models.Sequential([
	layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
	layers.MaxPooling2D(),
	layers.Conv2D(32, 3, padding='same', activation='relu'),
	layers.MaxPooling2D(),
	layers.Conv2D(64, 3, padding='same', activation='relu'),
	layers.MaxPooling2D(),
	layers.Flatten(),
	layers.Dense(512, activation='relu'),
	layers.Dense(3, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_generator, steps_per_epoch = train_generator.samples // BATCH_SIZE, epochs = EPOCHS)

# Evaluate the model
results = model.evaluate(test_generator)
print('test loss, test acc:', results)

# Save trained model
model.save(model_dir)