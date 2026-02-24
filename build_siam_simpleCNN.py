###################################################################################################
# This script builds and trains a Siamese network with a simple CNN architecture
# for feature extraction using triplet loss. It includes data loading, preprocessing, 
# model definition, training, and saving the trained model in .pb format.
###################################################################################################

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, MaxPooling2D, GlobalAveragePooling2D, Dense, Lambda, Add
from tensorflow.keras.models import Model
import os
from glob import glob
import random
from collections import defaultdict
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

input_shape = (128, 64, 3)  # Height=128, Width=64 (original DeepSORT format)

def preprocess_image(path):
    img = cv2.imread(path)  # Keep as BGR (do not convert to RGB!)
    img = cv2.resize(img, (64, 128))  # Width, Height for OpenCV
    img = img.astype(np.float32) / 255.0  # Normalize to [0, 1]
    return img  # Return without batch dimension

def build_feature_extractor(input_shape=(128, 64, 3)):
    inputs = Input(shape=input_shape, name="images")
    
    # Original DeepSORT architecture
    x = Conv2D(32, (3,3), strides=1, padding='same', name='conv1')(inputs)
    x = BatchNormalization(name='bn1')(x)
    x = ReLU(name='relu1')(x)
    
    x = Conv2D(32, (3,3), strides=1, padding='same', name='conv2')(x)
    x = BatchNormalization(name='bn2')(x)
    x = ReLU(name='relu2')(x)
    x = MaxPooling2D((3,3), strides=2, name='pool1')(x)
    
    # Residual blocks
    x = _residual_block(x, 32, name='res1')
    x = _residual_block(x, 32, name='res2')
    
    x = Conv2D(64, (3,3), strides=1, padding='same', name='conv3')(x)
    x = BatchNormalization(name='bn3')(x)
    x = ReLU(name='relu3')(x)
    x = MaxPooling2D((3,3), strides=2, name='pool2')(x)
    
    x = _residual_block(x, 64, name='res3')
    x = _residual_block(x, 64, name='res4')
    
    x = GlobalAveragePooling2D(name='gap')(x)
    x = Dense(128, activation=None, name='fc')(x)
    outputs = Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name="features")(x)
    
    return Model(inputs, outputs, name="feature_extractor")

def _residual_block(x, filters, name=None):
    shortcut = x
    x = Conv2D(filters, (3,3), padding='same', name=f'{name}_conv1')(x)
    x = BatchNormalization(name=f'{name}_bn1')(x)
    x = ReLU(name=f'{name}_relu1')(x)
    x = Conv2D(filters, (3,3), padding='same', name=f'{name}_conv2')(x)
    x = BatchNormalization(name=f'{name}_bn2')(x)
    x = Add(name=f'{name}_add')([shortcut, x])
    x = ReLU(name=f'{name}_out')(x)
    return x

def triplet_loss(y_true, y_pred, margin=0.2):
    anchor = y_pred[:, 0]  # (batch_size, 128)
    positive = y_pred[:, 1]  # (batch_size, 128)
    negative = y_pred[:, 2]  # (batch_size, 128)
    
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
    
    basic_loss = pos_dist - neg_dist + margin
    loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0))
    
    return loss

def load_dataset(dataset_path):
    data = defaultdict(list)
    person_folders = glob(os.path.join(dataset_path, "*"))
    
    for folder in person_folders:
        person_id = os.path.basename(folder)
        image_paths = glob(os.path.join(folder, "*.jpg")) + glob(os.path.join(folder, "*.png"))
        data[person_id].extend(image_paths)
    
    return data

def generate_triplets(data, batch_size=32):
    person_ids = list(data.keys())
    
    while True:
        batch_persons = random.sample(person_ids, min(batch_size, len(person_ids)))
        
        anchors, positives, negatives = [], [], []
        
        for person_id in batch_persons:
            person_images = data[person_id]
            if len(person_images) < 2:
                continue
                
            anchor_path, positive_path = random.sample(person_images, 2)
            
            # Find a negative from different person
            negative_path = None
            for _ in range(10):
                neg_person_id = random.choice(person_ids)
                if neg_person_id != person_id and data[neg_person_id]:
                    negative_path = random.choice(data[neg_person_id])
                    break
            
            if negative_path is None:
                continue
                
            anchors.append(preprocess_image(anchor_path))
            positives.append(preprocess_image(positive_path))
            negatives.append(preprocess_image(negative_path))
        
        if not anchors:
            continue
            
        yield (np.array(anchors), np.array(positives), np.array(negatives)), np.zeros(len(anchors))

def build_siamese_network(input_shape=(128, 64, 3)):
    input_anchor = Input(shape=input_shape, name='anchor_input')
    input_positive = Input(shape=input_shape, name='positive_input')
    input_negative = Input(shape=input_shape, name='negative_input')
    
    feature_extractor = build_feature_extractor(input_shape)
    
    embedding_anchor = feature_extractor(input_anchor)
    embedding_positive = feature_extractor(input_positive)
    embedding_negative = feature_extractor(input_negative)
    
    stacked_embeddings = tf.stack([embedding_anchor, embedding_positive, embedding_negative], axis=1)
    
    siamese_model = Model(
        inputs=[input_anchor, input_positive, input_negative],
        outputs=stacked_embeddings,
        name='siamese_network'
    )
    
    return siamese_model, feature_extractor

def save_frozen_pb(model, output_dir):
    # Create a concrete function from the Keras model
    full_model = tf.function(lambda x: model(x))
    concrete_func = full_model.get_concrete_function(
        tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype, name="images")
    )
    
    # Convert variables to constants and save as .pb
    frozen_func = convert_variables_to_constants_v2(concrete_func)
    tf.io.write_graph(
        graph_or_graph_def=frozen_func.graph,
        logdir=output_dir,
        name="frozen_model_cnn_highstep.pb",
        as_text=False
    )
    print(f"Saved frozen model to {output_dir}/frozen_model_cnn_highstep.pb")

def transfer_weights(src_model, dst_model):
    """Proper weight transfer between models with layer name matching"""
    for dst_layer in dst_model.layers:
        if dst_layer.weights:
            try:
                # Find corresponding layer in source model
                src_layer = src_model.get_layer(dst_layer.name)
                dst_layer.set_weights(src_layer.get_weights())
            except ValueError as e:
                print(f"Couldn't transfer weights for {dst_layer.name}: {str(e)}")
                # Initialize with default weights if transfer fails
                dst_layer.build(dst_layer.input_shape)

def train_feature_extractor(dataset_path, epochs=50, batch_size=32, output_dir="output"):
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    data = load_dataset(dataset_path)
    print("Num classes:", len(data))
    print("Min imgs/class:", min(len(v) for v in data.values()))
    print("Max imgs/class:", max(len(v) for v in data.values()))
    print("Classes with <2 images:", sum(1 for v in data.values() if len(v) < 2))
    
    # Build model
    siamese_model, feature_extractor = build_siamese_network(input_shape)
    
    # Compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    siamese_model.compile(optimizer=optimizer, loss=triplet_loss)

    # Create dataset with proper batching
    def create_dataset():
        for (anchors, positives, negatives), _ in generate_triplets(data, batch_size):
            # Yield each triplet separately
            for i in range(len(anchors)):
                yield (anchors[i], positives[i], negatives[i]), 0.0
    
    # Define output signature
    output_signature = (
        (
            tf.TensorSpec(shape=(128, 64, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(128, 64, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(128, 64, 3), dtype=tf.float32)
        ),
        tf.TensorSpec(shape=(), dtype=tf.float32)
    )
    
    # Create and batch dataset
    dataset = tf.data.Dataset.from_generator(
        create_dataset,
        output_signature=output_signature
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    # Train model
    history = siamese_model.fit(
        dataset,
        epochs=epochs,
        steps_per_epoch=2000,
        verbose=1
    )
     
    #transfer_weights(siamese_model.layers[3], feature_extractor)  # Transfer from shared extractor
    
    # 4. Verify architecture and weights
    feature_extractor.summary()
    test_input = np.random.rand(1, 128, 64, 3).astype(np.float32)
    test_output = feature_extractor(test_input)
    print(f"Output shape: {test_output.shape}")  # Should be (1, 128)
    
    # 5. Save in .pb format
    save_frozen_pb(feature_extractor, output_dir)
    
    return feature_extractor, history

# Example usage:
model = train_feature_extractor("./dataset_siam_21", epochs=15, batch_size=32)