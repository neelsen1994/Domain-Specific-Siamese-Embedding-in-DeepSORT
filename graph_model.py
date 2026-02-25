##############################################################################################
# This script loads two TensorFlow frozen models (.pb files), preprocesses input images,
# runs inference to extract embeddings, and compares the embeddings using cosine similarity.
# It also prints model graph details and calculates the number of parameters in each model. 
# This is useful for debugging and validating model performance.
##############################################################################################

import tensorflow as tf
import numpy as np
import cv2
import os

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

# Load and preprocess an image
def preprocess_image(image_path, target_size=(64, 128)):
    """
    Preprocess image to feed into the model.
    Args:
        image_path: Path to the input image.
        target_size: Model input size (default is 64x128).
    Returns:
        Preprocessed image as a numpy array.
    """
    img = cv2.imread(image_path)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img = cv2.resize(img, (target_size[0], target_size[1]))  # Resize image
    #img = img.astype(np.float32) / 255.0  # Normalize to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    print(f"\nPreprocessed image shape: {img.shape} (should be (1, 128, 64, 3))")
    return img

# Run an image through the model and get output
def test_model(graph, image, input_tensor_name, output_tensor_name):
    """
    Run inference on a single image using the model.
    Args:
        graph: The TensorFlow graph.
        image: Preprocessed image.
        input_tensor_name: Name of the input tensor.
        output_tensor_name: Name of the output tensor.
    Returns:
        Model output as numpy array.
    """
    with tf.compat.v1.Session(graph=graph) as sess:
        # Get input and output tensors
        input_tensor = graph.get_tensor_by_name(input_tensor_name)
        output_tensor = graph.get_tensor_by_name(output_tensor_name)
        
        # Run inference
        output = sess.run(output_tensor, feed_dict={input_tensor: image})
        return output
    
def get_embeddings(graph, image, input_tensor_name, output_tensor_name):
    """Run model inference to get embeddings"""
    with tf.compat.v1.Session(graph=graph) as sess:
        input_tensor = graph.get_tensor_by_name(input_tensor_name)
        output_tensor = graph.get_tensor_by_name(output_tensor_name)
        embeddings = sess.run(output_tensor, feed_dict={input_tensor: image})
    return embeddings.squeeze() 

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

if __name__ == "__main__":
    # Configuration
    model1_path = "./model_feature_extractor/mars-small128.pb"
    model2_path = "./runs/turkey_reid/best_model.pb" #"./output/frozen_model_cnn_highstep.pb"    #"./model_feature_extractor/frozen_model.pb"
    image1_path = "./dataset_siam/object_1/frame_100.jpg"
    image2_path = "./dataset_siam/object_2/frame_400.jpg"

    #image1_path = "./data_mars/object_1/f1.png"
    #image2_path = "./data_mars/object_2/f4.png"
    
    # Tensor names (update these based on your models)
    model1_input = "images:0"
    model1_output = "features:0" 
    model2_input = "images:0"
    model2_output = "Identity:0"

    # Load both models
    print("Loading models...")
    model1 = load_pb_model(model1_path)
    model2 = load_pb_model(model2_path)

    #print_graph_details(model2)
    #calculate_parameters(model1)
    calculate_parameters(model2)

#    # Preprocess images
#    print("\nPreprocessing images...")
#    image1_m1 = preprocess_image(image1_path)
#    image2_m1 = preprocess_image(image2_path)
#
#    image1_m2 = preprocess_image(image1_path)  # target_size=(64, 128) for mobnetv1 model
#    image2_m2 = preprocess_image(image2_path)  # target_size=(64, 128) for mobnetv1 model
#
#    # Get embeddings from both models
#    print("\nExtracting embeddings:")
#    
#    # Model 1
#    emb1_model1 = get_embeddings(model1, image1_m1, model1_input, model1_output)
#    emb2_model1 = get_embeddings(model1, image2_m1, model1_input, model1_output)
#    print(f"Model 1 embeddings shape: {emb1_model1.shape}")
#
#    # Model 2
#    emb1_model2 = get_embeddings(model2, image1_m2, model2_input, model2_output)
#    emb2_model2 = get_embeddings(model2, image2_m2, model2_input, model2_output)
#    print(f"Model 2 embeddings shape: {emb2_model2.shape}")
#
#    # Compute cosine similarities
#    sim_model1 = cosine_similarity(emb1_model1, emb2_model1)
#    sim_model2 = cosine_similarity(emb1_model2, emb2_model2)
#
#    print("\nResults:")
#    print(f"Model 1 Cosine Similarity: {sim_model1:.4f}")
#    print(f"Model 2 Cosine Similarity: {sim_model2:.4f}")
#    
#    # Compare models
#    print("\nComparison:")
#    if abs(sim_model1 - sim_model2) < 0.1:
#        print("Both models agree on similarity")
#    elif sim_model1 > sim_model2:
#        print("Model 1 predicts higher similarity")
#    else:
#        print("Model 2 predicts higher similarity")
#
#
#
#
## Main function
##if __name__ == "__main__":
##    # Path to your .pb model and test image
##    path_to_pb = "./model_feature_extractor/frozen_model.pb" #mars-small128_deep.pb"    #frozen_model.pb"  # Update this path
##    image_path = "image.jpg"    # Path to a test image (update this)
##
##    # Step 1: Load the .pb model
##    graph = load_pb_model(path_to_pb)
##    print("Model loaded successfully!")
##
##    # Step 2: Print graph details (metadata)
##    print_graph_details(graph)
##
##    calculate_parameters(graph)
##
##    # Step 3: Inspect input and output tensor names manually from the printed graph
##    # Example:
##    #   Input tensor: "images:0" (change based on your actual model input tensor name)
##    #   Output tensor: "features:0" (output tensor for the feature vector)
##
##    input_tensor_name = "images:0"   # Update with your model's input tensor name
##    output_tensor_name = "Identity:0"  # Update with your model's output tensor name
##
##    # Step 4: Preprocess an image
##    image = preprocess_image(image_path)
##    print(f"\nPreprocessed Image Shape: {image.shape}")
##
##    # Step 5: Run the model on the image
##    output = test_model(graph, image, input_tensor_name, output_tensor_name)
##    print("\nModel Output:")
##    print(output)
##    print("Output Shape:", output.shape)
#