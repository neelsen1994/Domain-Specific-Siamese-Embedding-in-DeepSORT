##############################################################################################
# This script builds and trains a Siamese network using MobileNetV2 as the base model
# for feature extraction with contrastive loss. It includes data loading, preprocessing,
# model definition, training, and saving the trained model in .pb format. It is optimized with 
# absolute difference distance calculation and img res of 128, 64
##############################################################################################

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_custom_objects
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

# Data Preparation
def load_image_pairs(dataset_path):
    """
    Load pairs of images for training Siamese network.
    Create positive and negative pairs for contrastive/triplet loss.
    """
    data = []
    labels = []

    object_dirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]

    for obj_dir in object_dirs:
        obj_path = os.path.join(dataset_path, obj_dir)
        images = [os.path.join(obj_path, img) for img in os.listdir(obj_path)]
        
        # Positive pairs
        for i in range(len(images)):
            for j in range(i+1, len(images)):
                data.append((images[i], images[j]))
                labels.append(1)  # Positive pair
        
        # Negative pairs
        for neg_dir in object_dirs:
            if neg_dir == obj_dir:
                continue
            neg_path = os.path.join(dataset_path, neg_dir)
            neg_image = np.random.choice(os.listdir(neg_path))
            neg_image_path = os.path.join(neg_path, neg_image)
            data.append((images[0], neg_image_path))
            labels.append(0)  # Negative pair
    
    return data, labels

# Step 2: Freeze the Model and Save as .pb
def save_frozen_pb(model, output_path):
    # Convert Keras model to a ConcreteFunction
    full_model = tf.function(lambda x: model(x))
    full_model = full_model.get_concrete_function(
        tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype, name="images"))

    # Freeze the ConcreteFunction
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_graph = frozen_func.graph

    # Save the frozen graph to a .pb file
    tf.io.write_graph(graph_or_graph_def=frozen_graph,
                      logdir=output_path,
                      name="frozen_model.pb",
                      as_text=False)
    print(f"Frozen model saved at: {output_path}/frozen_model.pb")

    # Print tensor names for debugging
    print("\nFrozen graph input tensor names:")
    for input_tensor in frozen_func.inputs:
        print(input_tensor)

    print("\nFrozen graph output tensor names:")
    for output_tensor in frozen_func.outputs:
        print(output_tensor)



# Siamese Network Architecture
def build_siamese_network(input_shape):
    base_model = MobileNetV2(input_shape=input_shape, include_top=False, pooling="avg")
    for layer in base_model.layers:
        layer.trainable = False
    
    input_tensor = layers.Input(shape=input_shape, name="images:0")
    features = base_model(input_tensor)
    dense = layers.Dense(128, activation="linear")(features)
    #return Model(input_tensor, dense, name="feature_extractor")
    normalized_output = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name="features")(dense)
    return Model(input_tensor, normalized_output, name="feature_extractor")

def contrastive_accuracy(y_true, y_pred, margin=1.0):
    """
    Custom accuracy metric for Siamese networks using contrastive loss.
    y_true: True labels (1 for similar, 0 for dissimilar).
    y_pred: Predicted distances (output of the model).
    margin: The margin used in contrastive loss.
    """
    # If the predicted distance is less than margin, consider it a "similar" pair
    y_pred_labels = tf.cast(y_pred < margin, tf.float32)
    
    # Compare predicted labels with true labels
    accuracy = tf.reduce_mean(tf.cast(tf.equal(y_true, y_pred_labels), tf.float32))
    return accuracy

# Contrastive Loss
def contrastive_loss(y_true, y_pred):
    margin = 1.0
    y_true = tf.cast(y_true, tf.float32)
    return tf.reduce_mean(y_true * tf.square(y_pred) + (1 - y_true) * tf.square(tf.maximum(margin - y_pred, 0)))

# Siamese Network Training
input_shape = (64, 128, 3)
feature_extractor = build_siamese_network(input_shape)

input_a = layers.Input(shape=input_shape)
input_b = layers.Input(shape=input_shape)

output_a = feature_extractor(input_a)
output_b = feature_extractor(input_b)

distance = layers.Lambda(lambda tensors: tf.abs(tensors[0] - tensors[1]))([output_a, output_b])
output = layers.Dense(1, activation="sigmoid")(distance)

# Register the custom metric
get_custom_objects()['contrastive_accuracy'] = contrastive_accuracy

siamese_network = Model([input_a, input_b], output)
siamese_network.compile(optimizer=Adam(0.0001), loss=contrastive_loss, metrics=["contrastive_accuracy"])

# Train the model
dataset_path = "dataset_siam"
pairs, labels = load_image_pairs(dataset_path)
pairs = np.array(pairs)
labels = np.array(labels)

def preprocess_image(path):
    img = tf.keras.preprocessing.image.load_img(path, target_size=(64, 128))
    return tf.keras.preprocessing.image.img_to_array(img) / 255.0

x1 = np.array([preprocess_image(p[0]) for p in pairs])
x2 = np.array([preprocess_image(p[1]) for p in pairs])

siamese_network.fit([x1, x2], labels, batch_size=32, epochs=20)

# Save the model
#feature_extractor.save("siamese_feature_extractor.h5")
#feature_extractor.save("./siamese_feature_extractor_pb", save_format='tf')

#print("Model saved as siamese_feature_extractor.h5")

#model = load_model("siamese_feature_extractor.h5", compile=False)
#tf.saved_model.save(model, "siamese_feature_extractor_pb")
#siamese_model.save(save_path, save_format='tf')



output_dir = "frozen_model"
save_frozen_pb(feature_extractor, output_dir)

print("Model saved in .pb format")
