###################################################################################################
# This script is used to test the embedding vector norms of a frozen tensorflow model (.pb file).
# It loads the model, runs a random input through it, and prints the norms of the output embeddings.
# This is useful for debugging and ensuring the model behaves as expected.
###################################################################################################

import tensorflow as tf
import numpy as np

# Load the model
with tf.io.gfile.GFile("./output/frozen_model_cnn.pb", "rb") as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())

# Create a TensorFlow graph
with tf.Graph().as_default() as graph:
    tf.import_graph_def(graph_def, name="")

# Get input/output tensors (verify these names!)
input_tensor = graph.get_tensor_by_name("images:0")  # Common input name
output_tensor = graph.get_tensor_by_name("Identity:0")  # Common output name

# Test with random noise (should NOT output unit vectors)
with tf.compat.v1.Session(graph=graph) as sess:
    random_input = np.random.rand(1, 128, 64, 3).astype(np.float32)
    #random_input = np.random.randint(0, 256, (1, 128, 64, 3), dtype=np.uint8).astype(np.float32)
    features = sess.run(output_tensor, feed_dict={input_tensor: random_input})
    print("Feature norms:", features) 