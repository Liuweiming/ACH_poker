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

"""Tests for open_spiel.python.algorithms.dqn."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf

from open_spiel.python import rl_environment
from open_spiel.python.algorithms import dqn
import pyspiel


class DQNTest(tf.test.TestCase):

  def test_run_tic_tac_toe(self):
    env = rl_environment.Environment("tic_tac_toe")
    state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]

    with self.session() as sess:
      agents = [
          dqn.DQN(  # pylint: disable=g-complex-comprehension
              sess,
              player_id,
              state_representation_size=state_size,
              num_actions=num_actions,
              hidden_layers_sizes=[16],
              replay_buffer_capacity=10,
              batch_size=5) for player_id in [0, 1]
      ]
      sess.run(tf.global_variables_initializer())
      time_step = env.reset()
      while not time_step.last():
        current_player = time_step.observations["current_player"]
        current_agent = agents[current_player]
        agent_output = current_agent.step(time_step)
        time_step = env.step([agent_output.action])

      for agent in agents:
        agent.step(time_step)

  def test_run_hanabi(self):
    # Hanabi is an optional game, so check we have it before running the test.
    game = "hanabi"
    if game not in pyspiel.registered_names():
      return

    num_players = 3
    env_configs = {
        "players": num_players,
        "max_life_tokens": 1,
        "colors": 2,
        "ranks": 3,
        "hand_size": 2,
        "max_information_tokens": 3,
        "discount": 0.
    }
    env = rl_environment.Environment(game, **env_configs)
    state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]

    with self.session() as sess:
      agents = [
          dqn.DQN(  # pylint: disable=g-complex-comprehension
              sess,
              player_id,
              state_representation_size=state_size,
              num_actions=num_actions,
              hidden_layers_sizes=[16],
              replay_buffer_capacity=10,
              batch_size=5) for player_id in range(num_players)
      ]
      sess.run(tf.global_variables_initializer())
      time_step = env.reset()
      while not time_step.last():
        current_player = time_step.observations["current_player"]
        agent_output = [agent.step(time_step) for agent in agents]
        time_step = env.step([agent_output[current_player].action])

      for agent in agents:
        agent.step(time_step)


class ReplayBufferTest(tf.test.TestCase):

  def test_replay_buffer_add(self):
    replay_buffer = dqn.ReplayBuffer(replay_buffer_capacity=10)
    self.assertEqual(len(replay_buffer), 0)
    replay_buffer.add("entry1")
    self.assertEqual(len(replay_buffer), 1)
    replay_buffer.add("entry2")
    self.assertEqual(len(replay_buffer), 2)

    self.assertIn("entry1", replay_buffer)
    self.assertIn("entry2", replay_buffer)

  def test_replay_buffer_max_capacity(self):
    replay_buffer = dqn.ReplayBuffer(replay_buffer_capacity=2)
    replay_buffer.add("entry1")
    replay_buffer.add("entry2")
    replay_buffer.add("entry3")
    self.assertEqual(len(replay_buffer), 2)

    self.assertIn("entry2", replay_buffer)
    self.assertIn("entry3", replay_buffer)

  def test_replay_buffer_sample(self):
    replay_buffer = dqn.ReplayBuffer(replay_buffer_capacity=3)
    replay_buffer.add("entry1")
    replay_buffer.add("entry2")
    replay_buffer.add("entry3")

    samples = replay_buffer.sample(3)

    self.assertIn("entry1", samples)
    self.assertIn("entry2", samples)
    self.assertIn("entry3", samples)


if __name__ == "__main__":
  tf.test.main()
