####################################################################################################
# This script builds and trains a Siamese network using MobileNetV2 as the base model
# for feature extraction with contrastive loss. It includes data loading, preprocessing,
# model definition, training, and saving the trained model in .pb format. It is optimized with Euclidean
# distance calculation and img res of 128,128 for better compatibility with MobileNetV2.
####################################################################################################

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
import os
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

# Data Preparation
def load_image_pairs(dataset_path):
    """Generate balanced positive and negative pairs with better sampling"""
    data = []
    labels = []
    object_dirs = [d for d in os.listdir(dataset_path) 
                   if os.path.isdir(os.path.join(dataset_path, d))]
    
    for obj_dir in object_dirs:
        obj_path = os.path.join(dataset_path, obj_dir)
        images = [os.path.join(obj_path, img) for img in os.listdir(obj_path)]
        num_images = len(images)
        
        # Generate positive pairs
        if num_images >= 2:
            for i in range(num_images):
                for j in range(i+1, num_images):
                    data.append((images[i], images[j]))
                    labels.append(1)
        
        # Generate negative pairs
        other_dirs = [d for d in object_dirs if d != obj_dir]
        if other_dirs and images:
            for img_path in images:
                neg_dir = np.random.choice(other_dirs)
                neg_path = os.path.join(dataset_path, neg_dir)
                neg_images = os.listdir(neg_path)
                if neg_images:
                    neg_img = np.random.choice(neg_images)
                    data.append((img_path, os.path.join(neg_path, neg_img)))
                    labels.append(0)
    
    return np.array(data), np.array(labels)

# Preprocessing with MobileNetV2 normalization
def preprocess_image(path):
    img = tf.keras.preprocessing.image.load_img(path, target_size=(64, 128))
    img = tf.keras.preprocessing.image.img_to_array(img)
    # Randomly rotate image between -30 and +30 degrees
    angle = np.random.uniform(-30, 30)
    img = tf.keras.preprocessing.image.random_rotation(img, angle, row_axis=0, col_axis=1, channel_axis=2)

    # Apply random width and height shift (up to 20% of image size)
    img = tf.keras.preprocessing.image.random_shift(img, wrg=0.2, hrg=0.2, row_axis=0, col_axis=1, channel_axis=2)

    # Apply random zoom (up to 20% zoom-in or zoom-out)
    #img = tf.keras.preprocessing.image.random_zoom(img, zoom_range=(0.8, 1.2), row_axis=0, col_axis=1, channel_axis=2)

    # Apply random horizontal flip (50% chance)
    #if np.random.rand() > 0.5:
    #    img = tf.keras.preprocessing.image.flip_axis(img, axis=1)

    # Apply random brightness adjustment (up to ±30% change)
    brightness_factor = np.random.uniform(0.7, 1.3)
    img = tf.image.adjust_brightness(img, brightness_factor)

    return (img / 127.5) - 1.0  # Proper MobileNetV2 normalization

# Siamese Network Architecture
def build_siamese_network(input_shape):
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        pooling="avg",
        weights='imagenet'
    )
    base_model.trainable = False  # Freeze base model
    
    inputs = layers.Input(shape=input_shape, name="images")
    x = base_model(inputs)
    x = layers.Dense(128, activation=None)(x)
    outputs = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name="features")(x)
    
    return Model(inputs, outputs, name="feature_extractor")

# Contrastive Loss - For similarity/dissimilarity calculation, we use Sigmoid with Binary Crossentropy loss which gives a probabilistic value
def contrastive_loss(y_true, y_pred, margin=1.0):
    y_true = tf.cast(y_true, tf.float32)
    return tf.reduce_mean(
        y_true * tf.square(y_pred) + 
        (1 - y_true) * tf.square(tf.maximum(margin - y_pred, 0))
    )

# Contrastive Accuracy
def contrastive_accuracy(y_true, y_pred, margin=1.0):
    pred_labels = tf.cast(y_pred < margin, tf.float32)
    return tf.reduce_mean(tf.cast(tf.equal(y_true, pred_labels), tf.float32))

# Model Saving
def save_frozen_pb(model, output_dir):
    full_model = tf.function(lambda x: model(x))
    concrete_func = full_model.get_concrete_function(
        tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype, name="images")
    )
    
    frozen_func = convert_variables_to_constants_v2(concrete_func)
    tf.io.write_graph(
        graph_or_graph_def=frozen_func.graph,
        logdir=output_dir,
        name="frozen_model.pb",
        as_text=False
    )

# Training Setup
input_shape = (128, 128, 3)  # Adjusted for MobileNetV2 input
feature_extractor = build_siamese_network(input_shape)

# Build Siamese Network
input_a = layers.Input(shape=input_shape)
input_b = layers.Input(shape=input_shape)

emb_a = feature_extractor(input_a)
emb_b = feature_extractor(input_b)

# Euclidean distance calculation
distance = layers.Lambda(
    lambda x: tf.sqrt(tf.reduce_sum(tf.square(x[0] - x[1]), axis=1, keepdims=True)),
    name='distance'
)([emb_a, emb_b])

siamese_model = Model([input_a, input_b], distance)
siamese_model.compile(
    optimizer=Adam(1e-4),
    loss=contrastive_loss,
    metrics=[contrastive_accuracy]
)

# Data Loading and Training
dataset_path = "dataset_siam"
pairs, labels = load_image_pairs(dataset_path)

# Process images in parallel
x1 = np.array([preprocess_image(p[0]) for p in pairs])
x2 = np.array([preprocess_image(p[1]) for p in pairs])

# Train the model
siamese_model.fit(
    [x1, x2], labels,
    batch_size=32,
    epochs=20,
    validation_split=0.2
)

# Save final feature extractor
save_frozen_pb(feature_extractor, "frozen_model")
print("Model successfully saved in .pb format")