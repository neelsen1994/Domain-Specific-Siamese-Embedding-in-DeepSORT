###############################################################################################
# This script loads a trained TensorFlow frozen model (.pb file), preprocesses input images,
# runs inference to extract embeddings, and compares the embeddings using cosine similarity.
# This is useful for debugging and ensuring the model behaves as expected.
###############################################################################################

import cv2
import numpy as np
import tensorflow as tf

# Function to load a .pb model
def load_pb_model(path_to_pb):
    with tf.io.gfile.GFile(path_to_pb, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
        return graph

# Function to print graph details (nodes, inputs, outputs)
def print_graph_details(graph):
    print("\nModel Graph Details:")
    print("-" * 40)
    for op in graph.get_operations():
        print(f"Operation: {op.name} | Type: {op.type}")
        for output in op.outputs:
            print(f"  Output: {output.name} | Shape: {output.shape}")
            
def cosine_similarity(emb1, emb2):
    """Compute cosine similarity between two embeddings"""
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

# Function to calculate the number of parameters in the graph
def calculate_parameters(graph):
    total_params = 0
    print("\nCalculating Number of Parameters:")
    print("-" * 40)
    for op in graph.get_operations():
        # Only focus on variable operations (weights/biases)
        if op.type in ['VariableV2', 'Const']:
            for output in op.outputs:
                shape = output.shape.as_list()
                if shape:  # Ensure the tensor has a shape
                    param_count = np.prod(shape)
                    total_params += param_count
                    print(f"{op.name} -> Shape: {shape}, Parameters: {param_count}")
    print(f"\nTotal Parameters: {total_params}")

def test_mars_model(model_path, img1_path, img2_path):
    # Load model
    with tf.io.gfile.GFile(model_path, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")
    
    # Get tensors (update names if different)
    input_tensor = graph.get_tensor_by_name("images:0")
    output_tensor = graph.get_tensor_by_name("features:0") # "features:0" "Identity:0" for some models
    
    # Preprocess (CRITICAL: BGR, no normalization)
    def load_image(path):
        img = cv2.imread(path)  # Keep BGR
        img = cv2.resize(img, (64, 128))  # Width, Height # 128, 64 for mobnetv1
        return np.expand_dims(img.astype(np.float32), axis=0)  # Add batch
    
    img1 = load_image(img1_path)
    img2 = load_image(img2_path)
    
    # Run inference
    with tf.compat.v1.Session(graph=graph) as sess:
        feat1 = sess.run(output_tensor, {input_tensor: img1})[0]
        feat2 = sess.run(output_tensor, {input_tensor: img2})[0]
    
    # Verify embeddings
    print("Embedding norms:", np.linalg.norm(feat1), np.linalg.norm(feat2))  # Should be ≈1.0
    
    # Calculate TRUE cosine similarity
    similarity = np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))
    return similarity

# Test with obviously different images
img_same = "./dataset_siam_21/object_7/frame_100.jpg"  # Two crops of same person        data_mars
img_diff = "./dataset_siam_21/object_127/frame_100.jpg"       # Completely different object      data_mars "./dataset_siam/object_1/frame_600.jpg"


similarity = test_mars_model("./runs/turkey_reid/best_model.pb", img_same, img_diff)
#similarity = test_mars_model("./output/frozen_model.pb", img_same, img_diff)    #"./output/frozen_model_cnn.pb" "./output/frozen_model.pb" "./model_feature_extractor/frozen_model_mobnetv1.pb"
print(f"Cosine similarity: {similarity:.4f}")  # Should be >0.8 for same, <0.3 for different