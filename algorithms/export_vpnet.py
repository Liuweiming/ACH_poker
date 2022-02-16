# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""An example of building and exporting a Tensorflow graph.

Adapted from export_graph.py. This one exports a simple network with value and
policy heads.

Adapted from the Travis Ebesu's blog post:
https://tebesu.github.io/posts/Training-a-TensorFlow-graph-in-C++-API
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

import algorithms.deep_cfr_model as model_lib

FLAGS = flags.FLAGS
flags.DEFINE_multi_integer("tensor_shape", [2, 3, 10], "tensor size")
flags.DEFINE_integer("num_actions", 3, "num actions")
flags.DEFINE_string("path", "/tmp/dp", "Directory to save graph")
flags.DEFINE_string("device", "/cpu:0", "Device")
flags.DEFINE_string("graph_def", "vnet.pb", "Filename for the graph")
flags.DEFINE_enum("nn_model", "ach", model_lib.CFRModel.valid_model_types,
                  "What type of model should be used?.")
flags.DEFINE_integer("nn_width", 64, "How wide should the network be.")
flags.DEFINE_integer("nn_depth", 1, "How deep should the network be.")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate used for training")
flags.DEFINE_float("weight_decay", 0.0001, "L2 regularization strength.")
flags.DEFINE_bool("verbose", False, "Print information about the model.")


def main(_):
    model = model_lib.CFRModel.build_model(
        FLAGS.nn_model, FLAGS.tensor_shape,
        FLAGS.num_actions, FLAGS.nn_width, FLAGS.nn_depth,
        FLAGS.weight_decay, FLAGS.learning_rate, FLAGS.path, FLAGS.device)
    model.write_graph(FLAGS.graph_def)

    if FLAGS.verbose:
        print("Model type: %s(%s, %s)" % (FLAGS.nn_model, FLAGS.nn_width,
                                          FLAGS.nn_depth))
        print("Model size:", model.num_trainable_variables, "variables")
        print("Variables:")
        model.print_trainable_variables()


if __name__ == "__main__":
    app.run(main)
