from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import base64
import json
import numpy as np

# Config
input_file = './sample_inputs/170402_212445-pix-08-in.png'
pix2pix_model = '170330_pix-05-edges2daisies-200e'

# Load image, serialized, convert to dict
with open(input_file, 'rb') as f:
    input_data = f.read()

input_instance = dict(input=base64.urlsafe_b64encode(input_data), key="0")

# Restore Pix2Pix model from checkpoint
loaded_graph = tf.Graph()
checkpoint = pix2pix_model + '/model_export/export'
checkpoint_meta_graph = checkpoint + '.meta'

with tf.compat.v1.Session(graph=loaded_graph) as sess:

    loader = tf.compat.v1.train.import_meta_graph(checkpoint_meta_graph)
    loader.restore(sess, checkpoint)

    input_vars = json.loads(tf.compat.v1.get_collection("inputs")[0])
    output_vars = json.loads(tf.compat.v1.get_collection("outputs")[0])
    
    input = loaded_graph.get_tensor_by_name(input_vars["input"])
    output = loaded_graph.get_tensor_by_name(output_vars["output"])

    input_value = np.array(input_instance["input"])
    output_value = sess.run(output, feed_dict={input: np.expand_dims(input_value, axis=0)})[0]

# Write output image
output_instance = dict(output=output_value, key="0")
b64data = output_instance["output"]
output_data = base64.urlsafe_b64decode(b64data)

with open('bin/' + input_file.split('/')[-1] + 'output.png', "wb") as f:
   f.write(output_data)