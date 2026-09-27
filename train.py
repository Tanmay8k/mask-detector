import os
import tensorflow as tf
from tensorflow import ImageDataGenerator
Train = "dataset/Train"
Val = "dataset/Validation"
Test = "dataset/Test"
Img = (224, 224) 
Batch = 32      
print("Loading and preparing image")
train = ImageDataGenerator(
    rescale=1.0 / 255.0, 
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)
gen = ImageDataGenerator(rescale=1.0 / 255.0)
generator = train.flow_from_directory(
    Train,
    target_size=Img,
    batch_size=Batch,
    class_mode="categorical",
    shuffle=True
)
vgenerator = gen.flow_from_directory(
    Val,
    target_size=Img,
    batch_size=Batch,
    class_mode="categorical",
    shuffle=False
)
print(f"\nClass Mapping: {generator.class_indices}")
print("\n Building MobileNetV2 Model Architecture")
base_model = tf.keras.applications.MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)
for layer in base_model.layers:
    layer.trainable = False
x = base_model.output
x = tf.keras.layers.AveragePooling2D(pool_size=(7, 7))(x)
x = tf.keras.layers.Flatten(name="flatten")(x)
x = tf.keras.layers.Dense(128, activation="relu")(x)
x = tf.keras.layers.Dropout(0.5)(x)
outputs = tf.keras.layers.Dense(2, activation="softmax")(x)
model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
model.compile(
    loss="categorical_crossentropy",
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    metrics=["accuracy"]
)
EPOCHS = 5  
print("\n Starting model training")
history = model.fit(
    generator,
    ch=len(generator),
    ta=vgenerator,
    eps=len(vgenerator),
    epochs=EPOCHS
)
mfile = "mask_detector.h5"
model.save(mfile)
print(f"\n Training complete! Model saved successfully as '{mfile}'")