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
"""
An model implements Deep CFR.
Adapted from open_spiel.python.algorithms.alpha_zero.model.
"""
import collections
import functools
import os
from typing import Sequence

import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow.compat.v1.keras as tfk


def cascade(x, fns):
    for fn in fns:
        x = fn(x)
    return x


tfkl = tf.keras.layers

# ================================================================
# Flat vectors
# ================================================================


def var_shape(x):
    out = x.get_shape().as_list()
    assert all(isinstance(a, int) for a in out), \
        "shape function assumes that shape is fully known"
    return out


def numel(x):
    return intprod(var_shape(x))


def intprod(x):
    return int(np.prod(x))


def flatgrad(loss, var_list, clip_norm=None):
    grads = tf.gradients(loss, var_list)
    if clip_norm is not None:
        grads = [tf.clip_by_norm(grad, clip_norm=clip_norm)
                 if grad is not None else grad for grad in grads]
    return tf.concat(axis=0, values=[
        tf.reshape(grad if grad is not None else tf.zeros_like(v), [numel(v)])
        for (v, grad) in zip(var_list, grads)
    ])


def set_flat(var_list, dtype=tf.float32):
    assigns = []
    shapes = list(map(var_shape, var_list))
    total_size = np.sum([intprod(shape) for shape in shapes])

    theta = tf.placeholder(dtype, [total_size], name="set_flat_input")
    start = 0
    assigns = []
    for (shape, v) in zip(shapes, var_list):
        size = intprod(shape)
        assigns.append(tf.assign(v, tf.reshape(
            theta[start:start + size], shape)))
        start += size
    op = tf.group(*assigns, name="set_flat")
    return op, theta


def apply_gradients(var_list, opt, name="train", dtype=tf.float32):
    shapes = list(map(var_shape, var_list))
    total_size = np.sum([intprod(shape) for shape in shapes])

    theta = tf.placeholder(dtype, [total_size], name="grad_input")
    clip_norm = tf.placeholder(tf.float32, None, name="clip_norm")
    start = 0
    grads = []
    for (shape, v) in zip(shapes, var_list):
        size = intprod(shape)
        grads.append(tf.reshape(theta[start:start + size], shape))
        start += size
    # clipped_grads, _ = tf.clip_by_global_norm(grads, clip_norm)
    clipped_grads = [tf.clip_by_norm(grad, clip_norm) for grad in grads]
    train_op = opt.apply_gradients(zip(clipped_grads, var_list), name=name)
    return train_op


def get_flat(var_list):
    return tf.concat(
        axis=0, values=[tf.reshape(v, [numel(v)]) for v in var_list], name="get_flat")


def flattenallbut0(x):
    return tf.reshape(x, [-1, intprod(x.get_shape().as_list()[1:])])


class TrainInput(collections.namedtuple(
        "TrainInput", "observation legal_actions masks target")):
    """Inputs for training the Model."""

    @staticmethod
    def stack(train_inputs):
        observation, legal_actions, masks, target = zip(*train_inputs)
        return TrainInput(
            np.array(observation, dtype=np.float32),
            np.array(legal_actions, dtype=np.int32),
            np.array(masks, dtype=np.int32),
            np.array(target))


MAX_RANK = 13
MAX_SUIT = 4
MAX_CARD = 52


class EmbLayer(tfkl.Layer):
    def __init__(self, nn_width, name):
        super(EmbLayer, self).__init__(name=name)
        self.nn_width = nn_width

    def build(self, input_shape):
        self.rank_embs = tfkl.Embedding(
            MAX_RANK, self.nn_width, name="rank_emb", input_length=input_shape[-1])
        self.suit_embs = tfkl.Embedding(
            MAX_SUIT, self.nn_width, name="suit_emb", input_length=input_shape[-1])
        self.point_embs = tfkl.Embedding(
            MAX_CARD, self.nn_width, name="point_emb", input_length=input_shape[-1])

    def call(self, cards, **kwargs):
        # -1 means "no card"
        with tf.device("/cpu:0"):
            valid = tf.expand_dims(tf.cast(cards >= 0, dtype=tf.float32), -1)
            cards_clipped = tf.clip_by_value(cards, 0, MAX_CARD)
            cards_rank = cards_clipped // MAX_SUIT
            cards_suit = cards_clipped % MAX_SUIT

        card_emb = self.rank_embs(cards_rank) + \
            self.suit_embs(cards_suit) + \
            self.point_embs(cards_clipped)
        card_emb = tf.reduce_sum(input_tensor=card_emb * valid, axis=1)
        return card_emb

    def get_config(self):
        return {"nn_width": self.nn_width}


class CFRModel(object):
    """A model for Deep CFR algorithms.
       support models for q-value, value and policy networks.
    """
    valid_model_types = ["ach", "a2c", "neurd_value",
                         "neurd_policy", "rpg_value", "rpg_policy"]

    def __init__(self, model_type, session, saver, path):
        """Init a model from build_model."""
        self.model_type = model_type
        self._session = session
        self._saver = saver
        self._path = path

        def get_var(name):
            return self._session.graph.get_tensor_by_name(name + ":0")

        def try_get_var(name):
            try:
                return get_var(name)
            except Exception:
                return None

        self._input = get_var("input")
        self._legal_actions = get_var("legal_actions")
        self._masks = get_var("masks")
        self._training = get_var("training")
        self._output = get_var("output")
        self._value = try_get_var("value")
        self._params = get_var("params")
        self._baseline = try_get_var("baseline")
        self._policy = try_get_var("policy")
        self._infer_list = [self._value, self._baseline, self._policy]
        self._infer_list = [i for i in self._infer_list if i is not None]
        self._loss = get_var("loss")
        self._targets = get_var("targets")
        self._baseline_targets = try_get_var("baseline_targets")
        self._train = self._session.graph.get_operation_by_name("sim_train")

    @classmethod
    def build_model(cls, model_type, input_shape, output_size,
                    nn_width, nn_depth, weight_decay, learning_rate, path, device):
        if model_type not in cls.valid_model_types:
            raise ValueError("Invalid model type")
        g = tf.Graph()
        with g.as_default():
            with tf.device(device):
                cls._define_graph(model_type, input_shape, output_size,
                                  nn_width, nn_depth, weight_decay, learning_rate)
                init = tf.variables_initializer(
                    tf.global_variables(), name="init_all_vars_op")
                get_flat_op = get_flat(tf.trainable_variables())
                set_flat_op, _ = set_flat(tf.trainable_variables())
                with tf.device("/cpu:0"):
                    saver = tf.train.Saver()
            session = tf.Session(
                graph=g, config=tf.ConfigProto(allow_soft_placement=True))
            session.__enter__()
            session.run(init)
        return cls(model_type, session, saver, path)

    @classmethod
    def from_checkpoint(cls, checkpoint, path=None):
        """Load a model from a checkpoint."""
        model = cls.from_graph(checkpoint, path)
        model.load_checkpoint(checkpoint)
        return model

    @classmethod
    def from_graph(cls, metagraph, path=None):
        """Load only the model from a graph or checkpoint."""
        if not os.path.exists(metagraph):
            metagraph += ".meta"
        if not path:
            path = os.path.dirname(metagraph)
        g = tf.Graph()  # Allow multiple independent models and graphs.
        with g.as_default():
            saver = tf.train.import_meta_graph(metagraph)
        session = tf.Session(graph=g)
        session.__enter__()
        session.run("init_all_vars_op")
        return cls("normal", session, saver, path)

    def __del__(self):
        if hasattr(self, "_session") and self._session:
            self._session.close()

    @staticmethod
    def _define_graph(model_type, input_shape, output_size,
                      nn_width, nn_depth, weight_decay, learning_rate):
        # NOTE: nn_depth has no effect here.
        # Inference inputs
        num_cards = input_shape[:-1]
        bets_size = input_shape[-1]
        num_cards = [nc for nc in num_cards if nc]
        input_size = int(np.sum(input_shape))
        print("input_shape = ", input_shape)
        print("num_cards = ", num_cards)
        print("bets_size = ", bets_size)
        observations = tf.placeholder(
            tf.float32, [None, input_size], name="input")
        obs_splits = tf.split(
            observations, num_cards + [bets_size], axis=-1)
        cards = obs_splits[:-1]
        bets = obs_splits[-1]
        masks = tf.placeholder(tf.bool, [None, output_size],
                               name="masks")
        masks = tf.cast(masks, tf.float32)
        legal_actions = tf.placeholder(tf.bool, [None, output_size],
                                       name="legal_actions")
        legal_mask = tf.cast(legal_actions, tf.float32)
        training = tf.placeholder(tf.bool, None, name="training")
        params = tf.placeholder(tf.float32, [6], name="params")
        learning_rate, eta, alpha, beta, thres, epsilon = tf.unstack(params)

        # target placeholder
        if model_type == "ach" or model_type == "a2c":
            targets = tf.placeholder(dtype=tf.float32, shape=[
                                     None, 3], name="targets")
            advantage_targets, value_targets, old_policies = tf.split(
                targets, [1, 1, 1], axis=-1)
        elif model_type == "neurd_value" or model_type == "rpg_value":
            targets = tf.placeholder(dtype=tf.float32, shape=[
                None, 1], name="targets")
        elif model_type == "neurd_policy" or model_type == "rpg_policy":
            targets = tf.placeholder(dtype=tf.float32, shape=[
                None, output_size], name="targets")

        card_embs = []
        for i, card_group in enumerate(cards):
            card_embs.append(
                EmbLayer(nn_width, name="emb_" + str(i))(card_group))

        card_embs = tf.concat(card_embs, axis=1)

        x = tf.nn.relu(tfkl.Dense(
            nn_width * 3, name="card_1")(card_embs))
        x = tf.nn.relu(tfkl.Dense(nn_width * 3, name="card_2")(x))
        x = tf.nn.relu(tfkl.Dense(nn_width, name="card_3")(x))
        # -1 means didn"t reach yet.
        bet_occurred = tf.cast(bets >= 0, tf.float32)
        bet_size = tf.clip_by_value(bets, 0, 1e6)
        bet_feats = tf.concat([bet_size, bet_occurred], axis=1)
        y = tf.nn.relu(tfkl.Dense(nn_width, name="bet_1")(bet_feats))
        y = tf.nn.relu(tfkl.Dense(nn_width, name="bet_2")(y) + y)

        z = tf.concat([x, y], axis=1)

        z = tf.nn.relu(tfkl.Dense(nn_width, name="comb_1")(z))
        z = tf.nn.relu(tfkl.Dense(nn_width, name="comb_2")(z) + z)
        z = tf.nn.relu(tfkl.Dense(nn_width, name="comb_3")(z) + z)

        def _output_head(input_z, output_s, prefix=""):
            norm_z = tfkl.LayerNormalization()(input_z)
            return tfkl.Dense(output_s, name=prefix + "logit")(norm_z)

        def _baseline_output_head(input_z, prefix=""):
            return tfkl.Dense(output_size, name=prefix + "logit")(input_z), tfkl.Dense(1, name=prefix + "baseline")(input_z)

        def _ach_output_head(input_z, prefix=""):
            norm_z = tfkl.LayerNormalization()(input_z)
            policy_logits = tfkl.Dense(
                output_size, name=prefix + "policy_logits")(norm_z)
            value_logits = tfkl.Dense(
                1, name=prefix + "value_logits")(norm_z)
            return policy_logits, value_logits

        if model_type == "ach" or model_type == "a2c":
            policy_logits, value_logits = _ach_output_head(z, "ach_")
            policy_logits = tf.where(
                legal_actions, policy_logits, -1e15 * tf.ones_like(policy_logits))
            # mean_policy_logits = tf.stop_gradient(tf.reduce_mean(
            #     legal_mask * policy_logits, axis=-1, keepdims=True))
            # policy_logits_diff = policy_logits - mean_policy_logits
            # policy_logits_diff = tf.identity(
            #     policy_logits_diff, "policy_logits_diff")
            value = tf.identity(value_logits, name="value")
            policy_logits = tf.identity(policy_logits, name="policy_logits")
            policy = tf.nn.softmax(policy_logits)
            output = tf.concat([policy_logits, value],
                               axis=-1, name="output")
        elif model_type == "neurd_policy" or model_type == "rpg_policy":
            policy_logits = _output_head(z, output_size, "policy_")
            policy_logits = tf.where(
                legal_actions, policy_logits, -1e15 * tf.ones_like(policy_logits))
            policy_logits = tf.identity(
                policy_logits, name="policy_logits")
            policy = tf.nn.softmax(policy_logits, name="policy")
            output = tf.identity(policy_logits, name="output")
        elif model_type == "neurd_value" or model_type == "rpg_value":
            value_logits = _output_head(z, output_size, "value_")
            value = tf.identity(value_logits, name="value")
            output = tf.identity(value, name="output")

        # losses
        if model_type == "ach":
            policy_logits_1 = tf.reduce_sum(
                masks * policy_logits, axis=-1, keepdims=True)
            # policy_logits_diff_1 = tf.reduce_sum(
            #     masks * policy_logits_diff, axis=-1, keepdims=True)
            policy_1 = tf.reduce_sum(
                masks * policy, axis=-1, keepdims=True)
            c_positive = tf.math.logical_and(tf.math.logical_and(policy_1 / old_policies < 1 + epsilon,
                                                                 policy_logits_1 < thres), advantage_targets >= 0)
            c_negtive = tf.math.logical_and(tf.math.logical_and(policy_1 / old_policies > 1 - epsilon,
                                                                policy_logits_1 > -thres), advantage_targets < 0)
            c = tf.stop_gradient(
                tf.cast(tf.math.logical_or(c_positive, c_negtive), tf.float32), name="c")
            value_loss = 0.5 * tf.reduce_mean(
                tf.squared_difference(value, value_targets), name="value_loss")
            # op = tf.identity(policy_logits_1, name="old_p")
            # adv = tf.identity(advantage_targets, name="adv")
            policy_loss = -tf.reduce_mean(
                c * policy_logits_1 * advantage_targets / old_policies, name="policy_loss")
            entropy_loss = tf.reduce_mean(tf.reduce_sum(
                policy * tf.nn.log_softmax(policy_logits), axis=-1), name="entropy_loss")
            loss = eta * policy_loss + alpha * \
                value_loss + beta * entropy_loss
            loss = tf.identity(loss, "loss")
        elif model_type == "a2c":
            log_policy_logits_1 = tf.reduce_sum(
                masks * tf.nn.log_softmax(policy_logits), axis=-1, keepdims=True)
            value_loss = 0.5 * tf.reduce_mean(
                tf.squared_difference(value, value_targets), name="value_loss")
            policy_loss = -tf.reduce_mean(
                log_policy_logits_1 * advantage_targets, name="policy_loss")
            entropy_loss = tf.reduce_mean(tf.reduce_sum(
                policy * tf.nn.log_softmax(policy_logits), axis=-1), name="entropy_loss")
            loss = eta * policy_loss + alpha * \
                value_loss + beta * entropy_loss
            loss = tf.identity(loss, "loss")
        elif model_type == "neurd_policy":
            mean_targets = tf.reduce_sum(
                masks * policy * targets, axis=-1, keepdims=True)
            advantages = tf.stop_gradient(targets - mean_targets)
            # mt = tf.identity(mean_targets, "mt")
            # adv = tf.identity(advantages, "adv")
            c_positive = tf.math.logical_and(
                policy_logits < thres, advantages >= 0)
            c_negtive = tf.math.logical_and(
                policy_logits > -thres, advantages < 0)
            c = tf.stop_gradient(
                tf.cast(tf.math.logical_or(c_positive, c_negtive), tf.float32), name="c")
            policy_loss = -tf.reduce_mean(
                tf.reduce_sum(c * policy_logits * advantages, axis=-1), name="policy_loss")
            entropy_loss = tf.reduce_mean(tf.reduce_sum(
                policy * tf.nn.log_softmax(policy_logits), axis=-1), name="entropy_loss")
            loss = eta * policy_loss + beta * entropy_loss
            loss = tf.identity(loss, "loss")
        elif model_type == "rpg_policy":
            mean_targets = tf.reduce_sum(
                masks * policy * targets, axis=-1, keepdims=True)
            policy_loss = tf.reduce_mean(
                tf.reduce_sum(tf.nn.relu(targets - mean_targets), axis=-1), name="policy_loss")
            entropy_loss = tf.reduce_mean(tf.reduce_sum(
                policy * tf.nn.log_softmax(policy_logits), axis=-1), name="entropy_loss")
            loss = eta * policy_loss + beta * entropy_loss
            loss = tf.identity(loss, "loss")
        elif model_type == "neurd_value" or model_type == "rpg_value":
            value_1 = tf.reduce_sum(masks * value, axis=-1, keepdims=True)
            value_loss = 0.5 * tf.reduce_mean(
                tf.squared_difference(value_1, targets), name="value_loss")
            loss = tf.identity(value_loss, "loss")

        # optimzer
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        tvs = tf.trainable_variables()
        grads = tf.gradients(loss, tvs)
        # clipped_grads, _ = tf.clip_by_global_norm(grads, 1.0)
        sim_train_op = optimizer.apply_gradients(
            zip(grads, tvs), name="sim_train")

    def inference(self, obs, legal, masks):
        return self._session.run(self._output,
                                 feed_dict={self._input: obs, self._legal_actions: legal, self._masks: masks})

    def update(self, train_inputs, params):
        batch = TrainInput.stack(train_inputs)
        for i in range(1):
            _, loss = self._session.run(["sim_train", self._loss],
                                        feed_dict={self._input: batch.observation,
                                                   self._legal_actions: batch.legal_actions,
                                                   self._masks: batch.masks,
                                                   self._targets: batch.target,
                                                   self._params: params})
        return loss

    @property
    def num_trainable_variables(self):
        return sum(np.prod(v.shape) for v in tf.trainable_variables())

    def print_trainable_variables(self):
        for v in tf.trainable_variables():
            print("{}: {}".format(v.name, v.shape))

    def write_graph(self, filename):
        full_path = os.path.join(self._path, filename)
        tf.train.export_meta_graph(
            graph_def=self._session.graph_def, saver_def=self._saver.saver_def,
            filename=full_path, as_text=False)
        # tf.io.write_graph(self._session.graph_def, self._path, filename, as_text=False)
        return full_path

    def save_checkpoint(self, step):
        return self._saver.save(
            self._session,
            os.path.join(self._path, "checkpoint"),
            global_step=step)

    def load_checkpoint(self, path):
        return self._saver.restore(self._session, path)
