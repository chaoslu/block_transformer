# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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
"""The main BERT model and related functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import json
import math
import re
import numpy as np
import six
import tensorflow as tf


class BertConfig(object):
	"""Configuration for `BertModel`."""

	def __init__(self,
				 vocab_size,
				 hidden_size=120,
				 num_encoding_layers=3,
				 num_interaction_layers=2,
				 num_attention_heads=4,
				 gaussian_prior_factor=0.1,
				 gaussian_prior_bias=0.1,
				 dependency_size=64,
				 hidden_act="gelu",
				 hidden_dropout_prob=0.1,
				 attention_probs_dropout_prob=0.1,
				 initializer_range=0.02):
		"""Constructs BertConfig.

		Args:
			vocab_size: Vocabulary size of `inputs_ids` in `BertModel`.
			hidden_size: Size of the encoder layers and the pooler layer.
			num_hidden_layers: Number of hidden layers in the Transformer encoder.
			num_attention_heads: Number of attention heads for each attention layer in
				the Transformer encoder.
			intermediate_size: The size of the "intermediate" (i.e., feed-forward)
				layer in the Transformer encoder.
			hidden_act: The non-linear activation function (function or string) in the
				encoder and pooler.
			hidden_dropout_prob: The dropout probability for all fully connected
				layers in the embeddings, encoder, and pooler.
			attention_probs_dropout_prob: The dropout ratio for the attention
				probabilities.
			max_position_embeddings: The maximum sequence length that this model might
				ever be used with. Typically set this to something large just in case
				(e.g., 512 or 1024 or 2048).
			type_vocab_size: The vocabulary size of the `token_type_ids` passed into
				`BertModel`.
			initializer_range: The stdev of the truncated_normal_initializer for
				initializing all weight matrices.
			continue_pretraining: the usage of the model -- the first time for pretraining
				or fine-tuning don't have optimizer state stored in the model
		"""
		self.vocab_size = vocab_size
		self.hidden_size = hidden_size
		self.num_encoding_layers = num_encoding_layers
		self.num_interaction_layers = num_interaction_layers
		self.num_attention_heads = num_attention_heads
		self.dependency_size = dependency_size
		self.hidden_act = hidden_act
		self.intermediate_size = hidden_size
		self.hidden_dropout_prob = hidden_dropout_prob
		self.attention_probs_dropout_prob = attention_probs_dropout_prob
		self.initializer_range = initializer_range

	@classmethod
	def from_dict(cls, json_object):
		"""Constructs a `BertConfig` from a Python dictionary of parameters."""
		config = BertConfig(vocab_size=None)
		for (key, value) in six.iteritems(json_object):
			config.__dict__[key] = value
		return config

	@classmethod
	def from_json_file(cls, json_file):
		"""Constructs a `BertConfig` from a json file of parameters."""
		with tf.gfile.GFile(json_file, "r") as reader:
			text = reader.read()
		return cls.from_dict(json.loads(text))

	def to_dict(self):
		"""Serializes this instance to a Python dictionary."""
		output = copy.deepcopy(self.__dict__)
		return output

	def to_json_string(self):
		"""Serializes this instance to a JSON string."""
		return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class BlockBertModel(object):
	"""BERT model ("Bidirectional Encoder Representations from Transformers").

	Example usage:

	```python
	# Already been converted into WordPiece token ids
	input_ids = tf.constant([[31, 51, 99], [15, 5, 0]])
	input_mask = tf.constant([[1, 1, 1], [1, 1, 0]])
	token_type_ids = tf.constant([[0, 0, 1], [0, 2, 0]])

	config = modeling.BertConfig(vocab_size=32000, hidden_size=512,
		num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

	model = modeling.BertModel(config=config, is_training=True,
		input_ids=input_ids, input_mask=input_mask, token_type_ids=token_type_ids)

	label_embeddings = tf.get_variable(...)
	pooled_output = model.get_pooled_output()
	logits = tf.matmul(pooled_output, label_embeddings)
	...
	```
	"""

	def __init__(self,
				 config,
				 is_training,
				 premise_input_ids,
				 hypothesis_input_ids,
				 premise_input_chars_ids=None,
				 hypothesis_input_chars_ids=None,
				 premise_input_mask=None,
				 hypothesis_input_mask=None,
				 use_one_hot_embeddings=False,
				 use_pretraining=False):
		"""Constructor for BertModel.

		Args:
			config: `BertConfig` instance.
			is_training: bool. true for training model, false for eval model. Controls
				whether dropout will be applied.
			input_ids: int32 Tensor of shape [batch_size, seq_length].
			input_mask: (optional) int32 Tensor of shape [batch_size, seq_length].
			token_type_ids: (optional) int32 Tensor of shape [batch_size, seq_length].
			use_one_hot_embeddings: (optional) bool. Whether to use one-hot word
				embeddings or tf.embedding_lookup() for the word embeddings.
			scope: (optional) variable scope. Defaults to "bert".

		Raises:
			ValueError: The config is invalid or one of the input tensor shapes
				is invalid.
		"""
		config = copy.deepcopy(config)
		if not is_training:
			config.hidden_dropout_prob = 0.0
			config.attention_probs_dropout_prob = 0.0

		input_shape = get_shape_list(premise_input_ids, expected_rank=2)
		batch_size = input_shape[0]
		seq_length = input_shape[1]

		if premise_input_mask is None:
			premise_input_mask = tf.ones(shape=[batch_size, seq_length], dtype=tf.int32)

		if hypothesis_input_mask is None:
			hypothesis_input_mask = tf.ones(shape=[batch_size, seq_length], dtype=tf.int32)

			with tf.variable_scope("embeddings"):
				# Perform embedding lookup on the word ids.
				(self.premise_embedding_output,self.hypothesis_embedding_output,self.embedding_table) = embedding_lookup(
						premise_input_ids=premise_input_ids,
						hypothesis_input_ids=hypothesis_input_ids,
						premise_input_chars_ids=premise_input_chars_ids,
				 		hypothesis_input_chars_ids=hypothesis_input_chars_ids,
						vocab_size=config.vocab_size,
						embedding_size=config.hidden_size,
						initializer_range=config.initializer_range,
						word_embedding_name="word_embeddings",
						use_one_hot_embeddings=use_one_hot_embeddings,
						use_pretraining=use_pretraining)
				

				# Add positional embeddings and token type embeddings, then layer
				# normalize and perform dropout.
				self.premise_embedding_output = embedding_position_processor(
						input_tensor=self.premise_embedding_output,
						use_position_embeddings=True,
						position_embedding_name="position_embeddings",
						initializer_range=config.initializer_range,
						dropout_prob=config.hidden_dropout_prob)

				self.hypothesis_embedding_output = embedding_position_processor(
						input_tensor=self.hypothesis_embedding_output,
						use_position_embeddings=True,
						position_embedding_name="position_embeddings",
						initializer_range=config.initializer_range,
						dropout_prob=config.hidden_dropout_prob)

			# This converts a 2D mask of shape [batch_size, seq_length] to a 3D
			# mask of shape [batch_size, seq_length, seq_length] which is used
			# for the attention scores.
			attention_mask_2p = create_attention_mask_from_input_mask(
					premise_input_ids, premise_input_mask)
			attention_mask_2h = create_attention_mask_from_input_mask(
					hypothesis_input_ids, hypothesis_input_mask)

			with tf.variable_scope("encoder"):
				# Run the stacked transformer.
				# `sequence_output` shape = [batch_size, seq_length, hidden_size].

				# The encoding blocks, two streams of premise and hypothesis operated seperately
				(self.all_encoder_layers_p, self.encoding_attention_scores_p) = encoding_transformer_model(
						input_tensor=self.premise_embedding_output,
						attention_mask=attention_mask_2p,
						dependency_size=config.dependency_size,
						hidden_size=config.hidden_size,
						num_encoding_layers=config.num_encoding_layers,
						num_attention_heads=config.num_attention_heads,
						intermediate_size=config.intermediate_size,
						intermediate_act_fn=get_activation(config.hidden_act),
						hidden_dropout_prob=config.hidden_dropout_prob,
						attention_probs_dropout_prob=config.attention_probs_dropout_prob,
						initializer_range=config.initializer_range,
						do_return_all_layers=True,
						gaussian_prior_factor=config.gaussian_prior_factor,
						gaussian_prior_bias=config.gaussian_prior_bias)

				(self.all_encoder_layers_h, self.encoding_attention_scores_h) = encoding_transformer_model(
						input_tensor=self.hypothesis_embedding_output,
						attention_mask=attention_mask_2h,
						dependency_size=config.dependency_size,
						hidden_size=config.hidden_size,
						num_encoding_layers=config.num_encoding_layers,
						num_attention_heads=config.num_attention_heads,
						intermediate_size=config.intermediate_size,
						intermediate_act_fn=get_activation(config.hidden_act),
						hidden_dropout_prob=config.hidden_dropout_prob,
						attention_probs_dropout_prob=config.attention_probs_dropout_prob,
						initializer_range=config.initializer_range,
						do_return_all_layers=True,
						gaussian_prior_factor=gaussian_prior_factor,
						gaussian_prior_bias=gaussian_prior_bias)

			self.encoder_output_p = self.all_encoder_layers_p[-1]
			self.encoder_output_h = self.all_encoder_layers_h[-1]

			with tf.variable_scope("interactor"):
				# adding the positional encoding to encoder output
				self.interaction_input_p = embedding_position_processor(
							input_tensor=self.encoder_output_p,
							position_embedding_name="position_embeddings",
							initializer_range=config.initializer_range,
							dropout_prob=config.hidden_dropout_prob)

				self.interaction_input_h = embedding_position_processor(
							input_tensor=self.encoder_output_h,
							position_embedding_name="position_embeddings",
							initializer_range=config.initializer_range,
							dropout_prob=config.hidden_dropout_prob)

				(self.all_interaction_layers_p, self.all_interaction_layers_h, self.inter_attetnion_scores_p,
					self.inter_attetnion_scores_h) = interaction_transformer_model(
							premise_input_tensor=self.interaction_input_p,
							hypothesis_input_tensor=self.interaction_input_h,
							attention_mask_premise=attention_mask_2p,
							attention_mask_hypothesis=attention_mask_2h,
							dependency_size=64,
							hidden_size=config.hidden_size,
							num_interaction_layers=config.num_interaction_layers,
							num_attention_heads=config.num_attention_heads,
							intermediate_size=config.intermediate_size,
							intermediate_act_fn=gelu,
							hidden_dropout_prob=0.1,
							attention_probs_dropout_prob=0.1,
							initializer_range=0.02,
							do_return_all_layers=True,
							gaussian_prior_factor=config.gaussian_prior_factor,
							gaussian_prior_bias=config.gaussian_prior_bias)

			self.interaction_output_p = self.all_interaction_layers_p[-1]
			self.interaction_output_h = self.all_interaction_layers_h[-1]

			with tf.variable_scope("comparison"):
				self.comparison_output = comparison_layer(
							premise_output=self.interaction_output_p,
							hypothesis_output=self.inter_attention_output_h,
							premise_input_tensor=self.interaction_input_p,
							hypothesis_input_tensor=self.interaction_input_h,
							initializer_range=0.02)
			# The "pooler" converts the encoded sequence tensor of shape
			# [batch_size, seq_length, hidden_size] to a tensor of shape
			# [batch_size, hidden_size]. This is necessary for segment-level
			# (or segment-pair-level) classification tasks where we need a fixed
			# dimensional representation of the segment.
			

	def get_comparision_output(self):
		return self.comparison_output

	def get_all_encoding_layers(self,premise=True):
		"""Gets final hidden layer of encoder.

		Returns:
			float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
			to the final hidden of the transformer encoder.
		"""
		if premise:
			return self.all_encoder_layers_p
		else:
			return self.all_encoder_layers_h

	def get_all_interaction_layers(self,premise=True):
		if premise:
			return self.all_interaction_layers_p
		else:
			return self.all_interaction_layers_h

	def get_embedding_output(self):
		"""Gets output of the embedding lookup (i.e., input to the transformer).

		Returns:
			float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
			to the output of the embedding layer, after summing the word
			embeddings with the positional embeddings and the token type embeddings,
			then performing layer normalization. This is the input to the transformer.
		"""
		return self.embedding_output

	def get_embedding_table(self):
		return self.embedding_table

	def get_query_filter(self):
		return self.query_filter

	def get_key_filter(self):
		return self.key_filter

	def get_attention_scores(self,name):
		attention_dict = {"enc-h":self.encoding_attention_scores_h,
						  "enc-p":self.encoding_attention_scores_p,
						  "inter-h":self.inter_attention_scores_h,
						  "inter-p":self.inter_attention_scores_p}
		return attention_dict[name]



def gelu(x):
	"""Gaussian Error Linear Unit.

	This is a smoother version of the RELU.
	Original paper: https://arxiv.org/abs/1606.08415
	Args:
		x: float Tensor to perform activation.

	Returns:
		`x` with the GELU activation applied.
	"""
	cdf = 0.5 * (1.0 + tf.tanh(
			(np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
	return x * cdf


def get_activation(activation_string):
	"""Maps a string to a Python function, e.g., "relu" => `tf.nn.relu`.

	Args:
		activation_string: String name of the activation function.

	Returns:
		A Python function corresponding to the activation function. If
		`activation_string` is None, empty, or "linear", this will return None.
		If `activation_string` is not a string, it will return `activation_string`.

	Raises:
		ValueError: The `activation_string` does not correspond to a known
			activation.
	"""

	# We assume that anything that"s not a string is already an activation
	# function, so we just return it.
	if not isinstance(activation_string, six.string_types):
		return activation_string

	if not activation_string:
		return None

	act = activation_string.lower()
	if act == "linear":
		return None
	elif act == "relu":
		return tf.nn.relu
	elif act == "gelu":
		return gelu
	elif act == "tanh":
		return tf.tanh
	else:
		raise ValueError("Unsupported activation: %s" % act)


def get_assignment_map_from_checkpoint(tvars, init_checkpoint, first_pretraining):
	"""Compute the union of the current variables and checkpoint variables."""
	assignment_map = {}
	initialized_variable_names = {}

	name_to_variable = collections.OrderedDict()
	for var in tvars:
		name = var.name
		m = re.match("^(.*):\\d+$", name)
		if m is not None:
			name = m.group(1)
		name_to_variable[name] = var

	init_vars = tf.train.list_variables(init_checkpoint)

	assignment_map = collections.OrderedDict()
	for x in init_vars:
		(name, var) = (x[0], x[1])
		if name not in name_to_variable:

			'''
			new_name = ""
			#tf.logging.info("%s\n" %name)

			# for the attention layers, the original linear layers for Q,K,V are intialized
			# from the vanilla bert model. while the filters are not.

			if first_pretraining:				
				if "attention" in name:
					scopes = re.split("/",name)
					assert "layer" in scopes[2]
					temp = scopes[2]
					scopes[2] = scopes[3]
					scopes[3] = scopes[4]
					scopes[4] = temp
					assert "attention" in scopes[2]
					assert "layer_" in scopes[4]
					new_name = "/".join(scopes)
					assert new_name in name_to_variable

					assignment_map[name] = new_name
					initialized_variable_names[new_name] = 1
					initialized_variable_names[new_name + ":0"] = 1
				else:
					continue
			else:
				continue
			'''
			continue
		else:
			assignment_map[name] = name
			initialized_variable_names[name] = 1
			initialized_variable_names[name + ":0"] = 1

	return (assignment_map, initialized_variable_names)


def dropout(input_tensor, dropout_prob):
	"""Perform dropout.

	Args:
		input_tensor: float Tensor.
		dropout_prob: Python float. The probability of dropping out a value (NOT of
			*keeping* a dimension as in `tf.nn.dropout`).

	Returns:
		A version of `input_tensor` with dropout applied.
	"""
	if dropout_prob is None or dropout_prob == 0.0:
		return input_tensor

	output = tf.nn.dropout(input_tensor, 1.0 - dropout_prob)
	return output


def layer_norm(input_tensor, name=None):
	"""Run layer normalization on the last dimension of the tensor."""
	return tf.contrib.layers.layer_norm(
			inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)


def layer_norm_and_dropout(input_tensor, dropout_prob, name=None):
	"""Runs layer normalization followed by dropout."""
	output_tensor = layer_norm(input_tensor, name)
	output_tensor = dropout(output_tensor, dropout_prob)
	return output_tensor


def create_initializer(initializer_range=0.02):
	"""Creates a `truncated_normal_initializer` with the given range."""
	return tf.truncated_normal_initializer(stddev=initializer_range)

def get_timing_signal_1d(length,
						 channels,
						 min_timescale=1.0,
						 max_timescale=1.0e4,
						 start_index=0):
	"""Gets a bunch of sinusoids of different frequencies.
	Each channel of the input Tensor is incremented by a sinusoid of a different
	frequency and phase.
	This allows attention to learn to use absolute and relative positions.
	Timing signals should be added to some precursors of both the query and the
	memory inputs to attention.
	The use of relative position is possible because sin(x+y) and cos(x+y) can be
	expressed in terms of y, sin(x) and cos(x).
	In particular, we use a geometric sequence of timescales starting with
	min_timescale and ending with max_timescale.  The number of different
	timescales is equal to channels / 2. For each timescale, we
	generate the two sinusoidal signals sin(timestep/timescale) and
	cos(timestep/timescale).  All of these sinusoids are concatenated in
	the channels dimension.
	Args:
	length: scalar, length of timing signal sequence.
	channels: scalar, size of timing embeddings to create. The number of
		different timescales is equal to channels / 2.
	min_timescale: a float
	max_timescale: a float
	start_index: index of first position
	Returns:
	a Tensor of timing signals [1, length, channels]
	"""
	position = tf.to_float(tf.range(length) + start_index)
	num_timescales = channels // 2
	log_timescale_increment = (math.log(float(max_timescale) / float(min_timescale)) / tf.maximum(tf.to_float(num_timescales) - 1, 1))
	inv_timescales = min_timescale * tf.exp(tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
	scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
	# Please note that this slightly differs from the published paper.
	# See a discussion here: https://github.com/tensorflow/tensor2tensor/pull/177
	signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
	signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
	signal = tf.reshape(signal, [1, length, channels])
	return signal


def load_embedding_table(fname):
	_,_,embedding_table = cPickle.load(open(fname,"rb"))
	return embedding_table

def load_chars_embedding_table(fname):
	_,_,embedding_table = cPickle.load(open(fname,"rb"))
	return embedding_table

def embedding_lookup(premise_input_ids,
					 hypothesis_input_ids,
					 premise_input_chars_ids,
				 	 hypothesis_input_chars_ids,
					 vocab_size,
					 embedding_size=300,
					 initializer_range=0.02,
					 word_embedding_name="word_embeddings",
					 use_one_hot_embeddings=False,
					 use_pretraining=False):
	"""Looks up words embeddings for id tensor.

	Args:
		input_ids: int32 Tensor of shape [batch_size, seq_length] containing word
			ids.
		vocab_size: int. Size of the embedding vocabulary.
		embedding_size: int. Width of the word embeddings.
		initializer_range: float. Embedding initialization range.
		word_embedding_name: string. Name of the embedding table.
		use_one_hot_embeddings: bool. If True, use one-hot method for word
			embeddings. If False, use `tf.gather()`.

	Returns:
		float Tensor of shape [batch_size, seq_length, embedding_size].
	"""
	# This function assumes that the input is of shape [batch_size, seq_length,
	# num_inputs].
	#
	# If the input is a 2D tensor of shape [batch_size, seq_length], we
	# reshape to [batch_size, seq_length, 1].
	if premise_input_ids.ndims == 2:
		premise_input_ids = tf.expand_dims(premise_input_ids, axis=[-1])
		hypothesis_input_ids = tf.expand_dims(hypothesis_input_ids, axis=[-1])
		premise_input_chars_ids = tf.expand_dims(premise_input_chars_ids, axis=[-1])
		hypothesis_input_chars_ids = tf.expand_dims(hypothesis_input_chars_ids, axis=[-1])

	if use_pretraining:
		embedding_table = load_embedding_table(embedding_file)
		embedding_table = tf.constant(embedding_table)
	else:
		embedding_table = tf.get_variable(
				name=word_embedding_name,
				shape=[vocab_size, embedding_size],
				initializer=create_initializer(initializer_range))


	flat_input_ids_p = tf.reshape(premise_input_ids, [-1])
	flat_input_ids_h = tf.reshape(hypothesis_input_ids, [-1])

	

	def transform_to_dense(ids_p,ids_h,emb_table,vocab_size,use_one_hot_embeddings):
		if use_one_hot_embeddings:
			one_hot_input_ids_p = tf.one_hot(ids_p, depth=vocab_size)
			one_hot_input_ids_h = tf.one_hot(ids_h, depth=vocab_size)
			output_p = tf.matmul(one_hot_input_ids_p, embedding_table)
			output_h = tf.matmul(one_hot_input_ids_h, embedding_table)
		else:
			output_p = tf.gather(embedding_table, flat_input_ids_p)
			output_h = tf.gather(embedding_table, flat_input_ids_h)

		return output_p,output_h


	word_output_p,word_output_h = transform_to_dense(flat_input_ids_p,flat_input_ids_h,embedding_table,
													vocab_size,use_one_hot_embeddings)

	if use_pretraining:
		chars_embedding_table = load_chars_embedding_table(chars_embedding_file)
		chars_vocab_size,chars_embedding_size = chars_embedding_table.shape
		
		flat_input_chars_ids_p = tf.reshape(premise_input_chars_ids, [-1])
		flat_input_chars_ids_h = tf.reshape(hypothesis_input_chars_ids, [-1])
		
		chars_output_p,chars_output_h = transform_to_dense(flat_input_chars_ids_p,flat_input_chars_ids_h,
										chars_embedding_table,chars_vocab_size,use_one_hot_embeddings)
		# concatenate word representation and character representation
		output_p = tf.concat(word_output_p, chars_output_p, axis=-1)
		output_h = tf.concat(word_output_h, chars_output_h, axis=-1)

		#enlarge embedding size for output reshape
		embedding_size += chars_embedding_size

	input_shape = get_shape_list(premise_input_ids)

	output_p = tf.reshape(output_p, input_shape[0:-1] + [input_shape[-1] * embedding_size])
	output_h = tf.reshape(output_h, input_shape[0:-1] + [input_shape[-1] * embedding_size])
	return (output_p, output_h, embedding_table)


def embedding_position_processor(input_tensor,
							position_embedding_name="position_embeddings",
							initializer_range=0.02,
							dropout_prob=0.1):
	"""Performs various post-processing on a word embedding tensor.

	Args:
		input_tensor: float Tensor of shape [batch_size, seq_length,
			embedding_size].
		use_token_type: bool. Whether to add embeddings for `token_type_ids`.
		token_type_ids: (optional) int32 Tensor of shape [batch_size, seq_length].
			Must be specified if `use_token_type` is True.
		token_type_vocab_size: int. The vocabulary size of `token_type_ids`.
		token_type_embedding_name: string. The name of the embedding table variable
			for token type ids.
		use_position_embeddings: bool. Whether to add position embeddings for the
			position of each token in the sequence.
		position_embedding_name: string. The name of the embedding table variable
			for positional embeddings.
		initializer_range: float. Range of the weight initialization.
		max_position_embeddings: int. Maximum sequence length that might ever be
			used with this model. This can be longer than the sequence length of
			input_tensor, but cannot be shorter.
		dropout_prob: float. Dropout probability applied to the final output tensor.

	Returns:
		float tensor with same shape as `input_tensor`.

	Raises:
		ValueError: One of the tensor shapes or input values is invalid.
	"""
	input_shape = get_shape_list(input_tensor, expected_rank=3)
	batch_size = input_shape[0]
	seq_length = input_shape[1]
	width = input_shape[2]

	output = input_tensor	
	position_embeddings = get_timing_signal_1d(seq_length + 1, width)
	output += position_embeddings

	output = layer_norm_and_dropout(output, dropout_prob)
	return output


def create_attention_mask_from_input_mask(from_tensor, to_mask):
	"""Create 3D attention mask from a 2D tensor mask.

	Args:
		from_tensor: 2D or 3D Tensor of shape [batch_size, from_seq_length, ...].
		to_mask: int32 Tensor of shape [batch_size, to_seq_length].

	Returns:
		float Tensor of shape [batch_size, from_seq_length, to_seq_length].
	"""
	from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
	batch_size = from_shape[0]
	from_seq_length = from_shape[1]

	to_shape = get_shape_list(to_mask, expected_rank=2)
	to_seq_length = to_shape[1]

	to_mask = tf.cast(
			tf.reshape(to_mask, [batch_size, 1, to_seq_length]), tf.float32)

	# We don't assume that `from_tensor` is a mask (although it could be). We
	# don't actually care if we attend *from* padding tokens (only *to* padding)
	# tokens so we create a tensor of all ones.
	#
	# `broadcast_ones` = [batch_size, from_seq_length, 1]
	broadcast_ones = tf.ones(
			shape=[batch_size, from_seq_length, 1], dtype=tf.float32)

	# Here we broadcast along two dimensions to create the mask.
	mask = broadcast_ones * to_mask

	return mask

'''
def create_sentences_attention_mask(sent_wise_mask):
	
	#creating sentencewise masks which allow only intra-sentence attention
	
	shape = get_shape_list(sent_wise_mask)
	batch_size = shape[0]
	seq_length = shape[1]

	from_mask = tf.reshape(sent_wise_mask,[batch_size,seq_length,1])
	to_mask = tf.reshape(sent_wise_mask,[batch_size,1,seq_length])
	mask = tf.cast(tf.math.equal(from_mask,to_mask),tf.int64)

	return mask
'''


def attention_layer(from_tensor,
					to_tensor,
					layer_idx,
					attention_mask=None,
					num_attention_heads=1,
					size_per_head=512,
					dependency_size=64,
					query_act=None,
					key_act=None,
					value_act=None,
					attention_probs_dropout_prob=0.0,
					initializer_range=0.02,
					do_return_2d_tensor=False,
					batch_size=None,
					from_seq_length=None,
					to_seq_length=None):
	"""Performs multi-headed attention from `from_tensor` to `to_tensor`.

	This is an implementation of multi-headed attention based on "Attention
	is all you Need". If `from_tensor` and `to_tensor` are the same, then
	this is self-attention. Each timestep in `from_tensor` attends to the
	corresponding sequence in `to_tensor`, and returns a fixed-with vector.

	This function first projects `from_tensor` into a "query" tensor and
	`to_tensor` into "key" and "value" tensors. These are (effectively) a list
	of tensors of length `num_attention_heads`, where each tensor is of shape
	[batch_size, seq_length, size_per_head].

	Then, the query and key tensors are dot-producted and scaled. These are
	softmaxed to obtain attention probabilities. The value tensors are then
	interpolated by these probabilities, then concatenated back to a single
	tensor and returned.

	In practice, the multi-headed attention are done with transposes and
	reshapes rather than actual separate tensors.

	Args:
		from_tensor: float Tensor of shape [batch_size, from_seq_length,
			from_width].
		to_tensor: float Tensor of shape [batch_size, to_seq_length, to_width].
		attention_mask: (optional) int32 Tensor of shape [batch_size,
			from_seq_length, to_seq_length]. The values should be 1 or 0. The
			attention scores will effectively be set to -infinity for any positions in
			the mask that are 0, and will be unchanged for positions that are 1.
		num_attention_heads: int. Number of attention heads.
		size_per_head: int. Size of each attention head.
		query_act: (optional) Activation function for the query transform.
		key_act: (optional) Activation function for the key transform.
		value_act: (optional) Activation function for the value transform.
		attention_probs_dropout_prob: (optional) float. Dropout probability of the
			attention probabilities.
		initializer_range: float. Range of the weight initializer.
		do_return_2d_tensor: bool. If True, the output will be of shape [batch_size
			* from_seq_length, num_attention_heads * size_per_head]. If False, the
			output will be of shape [batch_size, from_seq_length, num_attention_heads
			* size_per_head].
		batch_size: (Optional) int. If the input is 2D, this might be the batch size
			of the 3D version of the `from_tensor` and `to_tensor`.
		from_seq_length: (Optional) If the input is 2D, this might be the seq length
			of the 3D version of the `from_tensor`.
		to_seq_length: (Optional) If the input is 2D, this might be the seq length
			of the 3D version of the `to_tensor`.

	Returns:
		float Tensor of shape [batch_size, from_seq_length,
			num_attention_heads * size_per_head]. (If `do_return_2d_tensor` is
			true, this will be of shape [batch_size * from_seq_length,
			num_attention_heads * size_per_head]).

	Raises:
		ValueError: Any of the arguments or tensor shapes are invalid.
	"""

	def transpose_for_scores(input_tensor, batch_size, num_attention_heads,
													 seq_length, width):
		output_tensor = tf.reshape(
				input_tensor, [batch_size, seq_length, num_attention_heads, width])

		output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
		return output_tensor

	from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
	to_shape = get_shape_list(to_tensor, expected_rank=[2, 3])

	if len(from_shape) != len(to_shape):
		raise ValueError(
				"The rank of `from_tensor` must match the rank of `to_tensor`.")

	if len(from_shape) == 3:
		batch_size = from_shape[0]
		from_seq_length = from_shape[1]
		to_seq_length = to_shape[1]
	elif len(from_shape) == 2:
		if (batch_size is None or from_seq_length is None or to_seq_length is None):
			raise ValueError(
					"When passing in rank 2 tensors to attention_layer, the values "
					"for `batch_size`, `from_seq_length`, and `to_seq_length` "
					"must all be specified.")

	#for gaussian prior of attention matrix
	def create_distance_tensor(batch_size,from_seq_length,to_seq_length):
		distance_from_tensor = tf.ones(shape=[batch_size,from_seq_length])
		distance_from_tensor = tf.math.cumsum(distance_from_tensor,axis=-1)
		distance_from_tensor = tf.expand_dims(distance_from_tensor,[1])

		distance_to_tensor = tf.ones(shape=[batch_size,to_seq_length])
		distance_to_tensor = tf.math.cumsum(distance_to_tensor,axis=-1)
		distance_to_tensor = tf.expand_dims(distance_to_tensor,[2])

		distance_tensor = (distance_from_tensor - distance_to_tensor) * (distance_from_tensor - distance_to_tensor)
		distance_tensor = tf.cast(distance_tensor, tf.float32)
		return distance_tensor



	'''my bert modification with master filter, shuchao 
	perform cumsum on filters generated by each position in a sequence, two filters
	correspond to the upperbound and lowerbound of a token's dependcy-level in the 
	sequence. Only tokens of similiar dependency-levels attend to each other. 
	

	'''

	# Scalar dimensions referenced here:
	#   B = batch size (number of sequences)
	#   F = `from_tensor` sequence length
	#   T = `to_tensor` sequence length
	#   N = `num_attention_heads`
	#   H = `size_per_head`

	from_tensor_2d = reshape_to_matrix(from_tensor)
	to_tensor_2d = reshape_to_matrix(to_tensor)

	
	with tf.variable_scope("layer_%d" % layer_idx):
		# `query_layer` = [B*F, N*H]
		query_layer = tf.layers.dense(
			from_tensor_2d,
			num_attention_heads * size_per_head,
			activation=query_act,
			name="query",
			kernel_initializer=create_initializer(initializer_range),
			reuse=tf.AUTO_REUSE)

		# `key_layer` = [B*T, N*H]
		key_layer = tf.layers.dense(
			to_tensor_2d,
			num_attention_heads * size_per_head,
			activation=key_act,
			name="key",
			kernel_initializer=create_initializer(initializer_range),
			reuse=tf.AUTO_REUSE)

		# `value_layer` = [B*T, N*H]
		value_layer = tf.layers.dense(
			to_tensor_2d,
			num_attention_heads * size_per_head,
			activation=value_act,
			name="value",
			kernel_initializer=create_initializer(initializer_range),
			reuse=tf.AUTO_REUSE)
	
	'''
	# query filters
	query_filter_upper = tf.layers.dense(
		from_tensor_2d,
		dependency_size * num_attention_heads,
		activation=query_act,
		name="query_filter_upper",
		kernel_initializer=create_initializer(initializer_range),
		reuse=tf.AUTO_REUSE)

	query_filter_lower = tf.layers.dense(
		from_tensor_2d,
		dependency_size * num_attention_heads,
		activation=query_act,
		name="query_filter_lower",
		kernel_initializer=create_initializer(initializer_range),
		reuse=tf.AUTO_REUSE)

	# key filters
	key_filter_upper = tf.layers.dense(
		to_tensor_2d,
		dependency_size * num_attention_heads,
		activation=key_act,
		name="key_filter_upper",
		kernel_initializer=create_initializer(initializer_range),
		reuse=tf.AUTO_REUSE)

	key_filter_lower = tf.layers.dense(
		to_tensor_2d,
		dependency_size * num_attention_heads,
		activation=key_act,
		name="key_filter_lower",
		kernel_initializer=create_initializer(initializer_range),
		reuse=tf.AUTO_REUSE)


	query_filter_upper = transpose_for_scores(query_filter_upper, batch_size, num_attention_heads, 
									   from_seq_length, dependency_size)
	query_filter_lower = transpose_for_scores(query_filter_lower, batch_size, num_attention_heads, 
									   from_seq_length, dependency_size)
	key_filter_upper = transpose_for_scores(key_filter_upper, batch_size, num_attention_heads,
									 to_seq_length, dependency_size)
	key_filter_lower = transpose_for_scores(key_filter_lower, batch_size, num_attention_heads,
									 to_seq_length, dependency_size)


	query_filter_upper = tf.nn.softmax(query_filter_upper)
	query_filter_upper = tf.math.cumsum(query_filter_upper,axis=-1,reverse=True)
	query_filter_lower = tf.nn.softmax(query_filter_lower)
	query_filter_lower = tf.math.cumsum(query_filter_lower,axis=-1,reverse=True)

	query_filter = (1.0 - query_filter_upper) * query_filter_lower + (1.0 - query_filter_lower) * query_filter_upper
	#query_filter = tf.tile(tf.expand_dims(query_filter,axis=-1),[1,1,num_attention_heads])
	#query_filter = tf.reshape(query_filter,[batch_size * from_seq_length,-1])
	#query_layer = query_filter * query_layer
	
	key_filter_upper = tf.nn.softmax(key_filter_upper)
	key_filter_upper = tf.math.cumsum(key_filter_upper,axis=-1,reverse=True)
	key_filter_lower = tf.nn.softmax(key_filter_lower)
	key_filter_lower = tf.math.cumsum(key_filter_lower,axis=-1,reverse=True)

	key_filter = (1.0 - key_filter_upper) * key_filter_lower + (1.0 - key_filter_lower) * key_filter_upper
	#key_filter = tf.tile(tf.expand_dims(key_filter,axis=-1),[1,1,num_attention_heads])
	#key_filter = tf.reshape(key_filter,[batch_size * from_seq_length,-1])
	#key_layer = key_filter * key_layer
	'''
	
	
	# `query_layer` = [B, N, F, H]
	query_layer = transpose_for_scores(query_layer, batch_size, num_attention_heads, 
									   from_seq_length, size_per_head)
	# `key_layer` = [B, N, T, H]
	key_layer = transpose_for_scores(key_layer, batch_size, num_attention_heads,
									 to_seq_length, size_per_head)

	'''
	query_filter = transpose_for_scores(query_filter, batch_size, num_attention_heads, 
									   from_seq_length, size_per_head)
	# `key_layer` = [B, N, T, H]
	key_filter = transpose_for_scores(key_filter, batch_size, num_attention_heads,
									 to_seq_length, size_per_head)
	
	'''



	# Take the dot product between "query" and "key" to get the raw
	# attention scores.
	# `attention_scores` = [B, N, F, T]
	attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
	attention_scores = tf.multiply(attention_scores,
									1.0 / math.sqrt(float(size_per_head)))
	
	#attention_scores = tf.matmul(query_filter, key_filter, transpose_b=True)
	#attention_scores = tf.multiply(attention_scores,
	#								1.0 / math.sqrt(float(size_per_head)))

	# attention_scores_unmasked = attention_scores_unmasked + attention_filter
	
	if gaussian_prior_factor is not None:
		distance_mask = create_distance_tensor(batch_size,from_seq_length,to_seq_length)
		distance_mask = tf.math.abs(tf.multiply(distance_mask,gaussian_prior_factor) + gaussian_prior_bias) * -1.0
		distance_mask = tf.expand_dims(distance_mask, axis=[1])
		attention_scores = attention_scores + distance_mask

	if attention_mask is not None:
		# `attention_mask` = [B, 1, F, T]
		attention_mask = tf.expand_dims(attention_mask, axis=[1])

		# Since attention_mask is 1.0 for positions we want to attend and 0.0 for
		# masked positions, this operation will create a tensor which is 0.0 for
		# positions we want to attend and -10000.0 for masked positions.
		adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0

		# Since we are adding it to the raw scores before the softmax, this is
		# effectively the same as removing these entirely.
		attention_scores = attention_scores + adder


	# Normalize the attention scores to probabilities.
	# `attention_probs` = [B, N, F, T]
	attention_probs = tf.nn.softmax(attention_scores)

	# This is actually dropping out entire tokens to attend to, which might
	# seem a bit unusual, but is taken from the original Transformer paper.
	attention_probs = dropout(attention_probs, attention_probs_dropout_prob)

	# `value_layer` = [B, T, N, H]
	value_layer = tf.reshape(
			value_layer,
			[batch_size, to_seq_length, num_attention_heads, size_per_head])

	# `value_layer` = [B, N, T, H]
	value_layer = tf.transpose(value_layer, [0, 2, 1, 3])

	# `context_layer` = [B, N, F, H]
	context_layer = tf.matmul(attention_probs, value_layer)

	# `context_layer` = [B, F, N, H]
	context_layer = tf.transpose(context_layer, [0, 2, 1, 3])


	if do_return_2d_tensor:
		# `context_layer` = [B*F, N*H]
		context_layer = tf.reshape(
				context_layer,
				[batch_size * from_seq_length, num_attention_heads * size_per_head])
	else:
		# `context_layer` = [B, F, N*H]
		context_layer = tf.reshape(
				context_layer,
				[batch_size, from_seq_length, num_attention_heads * size_per_head])

	return (context_layer,attention_scores)


def attention_sublayer(from_tensor,
						to_tensor,
						layer_idx,
						attention_mask=None,
						num_attention_heads=1,
						size_per_head=512,
						dependency_size=64,
						query_act=None,
						key_act=None,
						value_act=None,
						attention_probs_dropout_prob=0.0,
						initializer_range=0.02,
						do_return_2d_tensor=False,
						batch_size=None,
						from_seq_length=None,
						to_seq_length=None,
						gaussian_prior_factor=None,
						gaussian_prior_bias=None):

	attention_heads = []
	(attention_heads,attention_scores) = attention_layer(
						from_tensor=from_tensor,
						to_tensor=to_tensor,
						layer_idx=layer_idx,
						attention_mask=attention_mask,
						num_attention_heads=num_attention_heads,
						size_per_head=size_per_head,
						dependency_size=dependency_size,
						query_act=query_act,
						key_act=key_act,
						value_act=value_act,
						attention_probs_dropout_prob=attention_probs_dropout_prob,
						initializer_range=initializer_range,
						do_return_2d_tensor=do_return_2d_tensor,
						batch_size=batch_size,
						from_seq_length=from_seq_length,
						to_seq_length=to_seq_length,
						gaussian_prior_factor=gaussian_prior_factor,
						gaussian_prior_bias=gaussian_prior_bias)
	attention_output = None
	if len(attention_heads) == 1:
		attention_output = attention_heads[0]
	else:
		# In the case where we have other sequences, we just concatenate
		# them to the self-attention head before the projection.
		attention_output = tf.concat(attention_heads, axis=-1)	

	# Run a linear projection of `hidden_size` then add a residual
	# with `layer_input`.
	with tf.variable_scope("output/layer_%d" % layer_idx):
		attention_output = tf.layers.dense(
					attention_output,
					hidden_size,
					kernel_initializer=create_initializer(initializer_range),
					reuse=tf.AUTO_REUSE)
		attention_output = dropout(attention_output, hidden_dropout_prob)
		attention_output = layer_norm(attention_output + from_tensor)
		
	return (attention_output,attention_scores)	

def ffn_sublayer(input_tensor,
				layer_idx,
				intermediate_size=3072,
				intermediate_act_fn=tf.nn.relu,
				initializer_range=0.02,
				hidden_size=768,
				hidden_dropout_prob=0.1):
	# The activation is only applied to the "intermediate" hidden layer.
	with tf.variable_scope("layer_%d/intermediate" % layer_idx):
		intermediate_output = tf.layers.dense(
					input_tensor,
					intermediate_size,
					activation=intermediate_act_fn,
					kernel_initializer=create_initializer(initializer_range),
					reuse=tf.AUTO_REUSE)

		# Down-project back to `hidden_size` then add the residual.
	with tf.variable_scope("layer_%d/output" % layer_idx):
		layer_output = tf.layers.dense(
					intermediate_output,
					hidden_size,
					kernel_initializer=create_initializer(initializer_range),
					reuse=tf.AUTO_REUSE)
		layer_output = dropout(layer_output, hidden_dropout_prob)
		layer_output = layer_norm(layer_output + input_tensor)

	return layer_output


def encoding_transformer_model(input_tensor,
						attention_mask=None,
						dependency_size=64,
						hidden_size=120,
						num_hidden_layers=3,
						num_attention_heads=4,
						intermediate_size=120,
						intermediate_act_fn=gelu,
						hidden_dropout_prob=0.1,
						attention_probs_dropout_prob=0.1,
						initializer_range=0.02,
						do_return_all_layers=False,
						gaussian_prior_factor=None,
						gaussian_prior_bias=None):
	"""Multi-headed, multi-layer Transformer from "Attention is All You Need".

	This is almost an exact implementation of the original Transformer encoder.

	See the original paper:
	https://arxiv.org/abs/1706.03762

	Also see:
	https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py

	Args:
		input_tensor: float Tensor of shape [batch_size, seq_length, hidden_size].
		attention_mask: (optional) int32 Tensor of shape [batch_size, seq_length,
			seq_length], with 1 for positions that can be attended to and 0 in
			positions that should not be.
		hidden_size: int. Hidden size of the Transformer.
		num_hidden_layers: int. Number of layers (blocks) in the Transformer.
		num_attention_heads: int. Number of attention heads in the Transformer.
		intermediate_size: int. The size of the "intermediate" (a.k.a., feed
			forward) layer.
		intermediate_act_fn: function. The non-linear activation function to apply
			to the output of the intermediate/feed-forward layer.
		hidden_dropout_prob: float. Dropout probability for the hidden layers.
		attention_probs_dropout_prob: float. Dropout probability of the attention
			probabilities.
		initializer_range: float. Range of the initializer (stddev of truncated
			normal).
		do_return_all_layers: Whether to also return all layers or just the final
			layer.

	Returns:
		float Tensor of shape [batch_size, seq_length, hidden_size], the final
		hidden layer of the Transformer.

	Raises:
		ValueError: A Tensor shape or parameter is invalid.
	"""
	if hidden_size % num_attention_heads != 0:
		raise ValueError(
				"The hidden size (%d) is not a multiple of the number of attention "
				"heads (%d)" % (hidden_size, num_attention_heads))

	attention_head_size = int(hidden_size / num_attention_heads)

	'''
	if attention_head_size % smoothness != 0:
		raise ValueError(
				"The attention head size (%d) is not a multiple of the smoothness "
				"heads (%d)" % (attention_head_size, smoothness))
	'''

	input_shape = get_shape_list(input_tensor, expected_rank=3)
	batch_size = input_shape[0]
	seq_length = input_shape[1]
	input_width = input_shape[2]

	# The Transformer performs sum residuals on all layers so the input needs
	# to be the same as the hidden size.
	if input_width != hidden_size:
		raise ValueError("The width of the input tensor (%d) != hidden size (%d)" %
										 (input_width, hidden_size))

	# We keep the representation as a 2D tensor to avoid re-shaping it back and
	# forth from a 3D tensor to a 2D tensor. Re-shapes are normally free on
	# the GPU/CPU but may not be free on the TPU, so we want to minimize them to
	# help the optimizer.
	prev_output = reshape_to_matrix(input_tensor)

	all_layer_outputs = []
	all_layer_attentions = []

	'''
	all_layer_queries = []
	all_layer_keys = []
	all_layer_attention_filters = []
	'''

	with tf.variable_scope("encoding"):
		for layer_idx in range(num_hidden_layers):
			#with tf.variable_scope("layer_%d" % layer_idx):
			layer_input = prev_output
			with tf.variable_scope("attention"):
				
				(attention_output,attention_scores) = attention_sublayer(
							from_tensor=layer_input,
							to_tensor=layer_input,
							layer_idx=layer_idx,
							attention_mask=attention_mask,
							num_attention_heads=num_attention_heads,
							size_per_head=attention_head_size,
							dependency_size=dependency_size,
							attention_probs_dropout_prob=attention_probs_dropout_prob,
							initializer_range=initializer_range,
							do_return_2d_tensor=True,
							batch_size=batch_size,
							from_seq_length=seq_length,
							to_seq_length=seq_length,
							gaussian_prior_factor=gaussian_prior_factor,
							gaussian_prior_bias=gaussian_prior_bias)

			with tf.variable_scope("ffn"):
				layer_output = ffn_sublayer(
							layer_idx=layer_idx,
							input_tensor=attention_output,
							intermediate_size=intermediate_size,
							intermediate_act_fn=intermediate_act_fn,
							hidden_size=hidden_size,
							initializer_range=initializer_range,
							hidden_dropout_prob=hidden_dropout_prob)

			
				
			
			prev_output = layer_output
			all_layer_outputs.append(layer_output)
			all_layer_attentions.append(attention_scores)
			'''
			all_layer_queries.append(query_filter)
			all_layer_keys.append(key_filter)
			'''

	if do_return_all_layers:
		final_outputs = []
		for layer_output in all_layer_outputs:
			final_output = reshape_from_matrix(layer_output, input_shape)
			final_outputs.append(final_output)
		return (final_outputs,all_layer_attentions)
	else:
		final_output = reshape_from_matrix(prev_output, input_shape)
		return (final_output,all_layer_attentions)

def interaction_transformer_model(premise_input_tensor,
								hypothesis_input_tensor,
								attention_mask_premise=None,
								attention_mask_hypothesis=None,
								dependency_size=64,
								hidden_size=120,
								num_hidden_layers=2,
								num_attention_heads=4,
								intermediate_size=120,
								intermediate_act_fn=gelu,
								hidden_dropout_prob=0.1,
								attention_probs_dropout_prob=0.1,
								initializer_range=0.02,
								do_return_all_layers=False,
								gaussian_prior_factor=None,
								gaussian_prior_bias=None):
	"""Multi-headed, multi-layer Transformer from "Attention is All You Need".

	This is almost an exact implementation of the original Transformer encoder.

	See the original paper:
	https://arxiv.org/abs/1706.03762

	Also see:
	https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py

	Args:
		input_tensor: float Tensor of shape [batch_size, seq_length, hidden_size].
		attention_mask: (optional) int32 Tensor of shape [batch_size, seq_length,
			seq_length], with 1 for positions that can be attended to and 0 in
			positions that should not be.
		hidden_size: int. Hidden size of the Transformer.
		num_hidden_layers: int. Number of layers (blocks) in the Transformer.
		num_attention_heads: int. Number of attention heads in the Transformer.
		intermediate_size: int. The size of the "intermediate" (a.k.a., feed
			forward) layer.
		intermediate_act_fn: function. The non-linear activation function to apply
			to the output of the intermediate/feed-forward layer.
		hidden_dropout_prob: float. Dropout probability for the hidden layers.
		attention_probs_dropout_prob: float. Dropout probability of the attention
			probabilities.
		initializer_range: float. Range of the initializer (stddev of truncated
			normal).
		do_return_all_layers: Whether to also return all layers or just the final
			layer.

	Returns:
		float Tensor of shape [batch_size, seq_length, hidden_size], the final
		hidden layer of the Transformer.

	Raises:
		ValueError: A Tensor shape or parameter is invalid.
	"""
	if hidden_size % num_attention_heads != 0:
		raise ValueError(
				"The hidden size (%d) is not a multiple of the number of attention "
				"heads (%d)" % (hidden_size, num_attention_heads))

	attention_head_size = int(hidden_size / num_attention_heads)


	input_shape = get_shape_list(premise_input_tensor, expected_rank=3)
	batch_size = input_shape[0]
	seq_length = input_shape[1]
	input_width = input_shape[2]

	# The Transformer performs sum residuals on all layers so the input needs
	# to be the same as the hidden size.
	if input_width != hidden_size:
		raise ValueError("The width of the input tensor (%d) != hidden size (%d)" %
										 (input_width, hidden_size))

	# We keep the representation as a 2D tensor to avoid re-shaping it back and
	# forth from a 3D tensor to a 2D tensor. Re-shapes are normally free on
	# the GPU/CPU but may not be free on the TPU, so we want to minimize them to
	# help the optimizer.
	prev_output_premise = reshape_to_matrix(premise_input_tensor)
	prev_output_hypothesis = reshape_to_matrix(prev_output_hypothesis)

	attention_mask_premise = attention_mask[0]
	attention_mask_hypothesis = attention_mask[1]

	all_layer_outputs_premise = []
	all_layer_outputs_hypothesis = []
	all_layer_attentions_premise =[]
	all_layer_attentions_hypothesis =[]

	'''
	all_layer_queries = []
	all_layer_keys = []
	all_layer_attentions = []
	all_layer_attention_filters = []
	'''

	with tf.variable_scope("interaction"):
		for layer_idx in range(num_hidden_layers):
			#with tf.variable_scope("layer_%d" % layer_idx):
			premise_layer_input = prev_output_premise
			hypothesis_layer_input = prev_output_hypothesis
			
			with tf.variable_scope("intra_attention"):
				(self_attention_output_p,self_attention_scores_p) = attention_sublayer(
							from_tensor=premise_layer_input,
							to_tensor=premise_layer_input,
							layer_idx=layer_idx,
							attention_mask=attention_mask_premise,
							num_attention_heads=num_attention_heads,
							size_per_head=attention_head_size,
							dependency_size=dependency_size,
							attention_probs_dropout_prob=attention_probs_dropout_prob,
							initializer_range=initializer_range,
							do_return_2d_tensor=True,
							batch_size=batch_size,
							from_seq_length=seq_length,
							to_seq_length=seq_length,
							gaussian_prior_factor=gaussian_prior_factor,
							gaussian_prior_bias=gaussian_prior_bias)

				(self_attention_output_h,self_attention_scores_h) = attention_sublayer(
							from_tensor=hypothesis_layer_input,
							to_tensor=hypothesis_layer_input,
							layer_idx=layer_idx,
							attention_mask=attention_mask_hypothesis,
							num_attention_heads=num_attention_heads,
							size_per_head=attention_head_size,
							dependency_size=dependency_size,
							attention_probs_dropout_prob=attention_probs_dropout_prob,
							initializer_range=initializer_range,
							do_return_2d_tensor=True,
							batch_size=batch_size,
							from_seq_length=seq_length,
							to_seq_length=seq_length,
							gaussian_prior_factor=gaussian_prior_factor,
							gaussian_prior_bias=gaussian_prior_bias)


			with tf.variable_scope("inter_attention"):
					(inter_attention_output_p,inter_attention_scores_p) = attention_sublayer(
								from_tensor=self_attention_output_p,
								to_tensor=self_attention_output_h,
								layer_idx=layer_idx,
								attention_mask=attention_mask_hypothesis,
								num_attention_heads=num_attention_heads,
								size_per_head=attention_head_size,
								dependency_size=dependency_size,
								attention_probs_dropout_prob=attention_probs_dropout_prob,
								initializer_range=initializer_range,
								do_return_2d_tensor=True,
								batch_size=batch_size,
								from_seq_length=seq_length,
								to_seq_length=seq_length,
								gaussian_prior_factor=gaussian_prior_factor,
								gaussian_prior_bias=gaussian_prior_bias)

					(inter_attention_output_h,inter_attention_scores_h) = attention_sublayer(
								from_tensor=self_attention_output_h,
								to_tensor=self_attention_output_p,
								layer_idx=layer_idx,
								attention_mask=attention_mask_premise,
								num_attention_heads=num_attention_heads,
								size_per_head=attention_head_size,
								dependency_size=dependency_size,
								attention_probs_dropout_prob=attention_probs_dropout_prob,
								initializer_range=initializer_range,
								do_return_2d_tensor=True,
								batch_size=batch_size,
								from_seq_length=seq_length,
								to_seq_length=seq_length,
								gaussian_prior_factor=gaussian_prior_factor,
								gaussian_prior_bias=gaussian_prior_bias)


			# The activation is only applied to the "intermediate" hidden layer.
			with tf.variable_scope("ffn"):
				premise_layer_output = ffn_sublayer(
							layer_idx=layer_idx,
							input_tensor=inter_attention_output_p,
							intermediate_size=intermediate_size,
							intermediate_act_fn=intermediate_act_fn,
							hidden_size=hidden_size,
							initializer_range=initializer_range,
							hidden_dropout_prob=hidden_dropout_prob)

			# The activation is only applied to the "intermediate" hidden layer.
			with tf.variable_scope("ffn"):
				hypothesis_layer_output = ffn_sublayer(
							layer_idx=layer_idx,
							input_tensor=inter_attention_output_h,
							intermediate_size=intermediate_size,
							intermediate_act_fn=intermediate_act_fn,
							hidden_size=hidden_size,
							initializer_range=initializer_range,
							hidden_dropout_prob=hidden_dropout_prob)

			prev_output_premise = premise_layer_output
			prev_output_hypothesis = hypothesis_layer_output

			all_layer_outputs_premise.append(premise_layer_output)
			all_layer_outputs_hypothesis.append(hypothesis_layer_output)

			# each attention score in an interaction sublayer comprises the self attention part
			# and the inter attetnion part, thus a 2-tuple

			all_layer_attentions_premise.append((self_attention_scores_p,inter_attention_scores_p))
			all_layer_attentions_hypothesis.append(self_attention_scores_h,inter_attention_scores_h)


	if do_return_all_layers:
		final_outputs_p = []
		final_outputs_h = []
		for layer_output in all_layer_outputs_premise:
			final_output_p = reshape_from_matrix(layer_output, input_shape)
			final_outputs_p.append(final_output_p)
		for layer_output in all_layer_outputs_hypothesis:
			final_output_h = reshape_from_matrix(layer_output, input_shape)
			final_outputs_h.append(final_output_h)
		return (final_outputs_p,final_outputs_h)
	
	else:
		final_output_p = reshape_from_matrix(prev_output_premise, input_shape)
		final_output_h = reshape_from_matrix(prev_output_hypothesis, input_shape)
		return (final_output_p,final_output_h,all_layer_attentions_premise,all_layer_outputs_hypothesis)


def comparison_layer(premise_output,
					hypothesis_output,
					premise_input_tensor,
					hypothesis_input_tensor,
					initializer_range):
	input_shape = get_shape_list(premise_input_tensor, expected_rank=3)
	seq_length = input_shape[1]
	hidden_size = input_shape[2]
	
	premise = tf.concat(premise_input,premise_output,-1)
	hypothesis = tf.concat(hypothesis_input,hypothesis_output,-1)

	premise = tf.layers.dense(premise,hidden_size,activation=tf.nn.relu,kernel_initializer=create_initializer(initializer_range))
	premise = tf.layers.dense(premise,hidden_size,kernel_initializer=create_initializer(initializer_range))
	premise = tf.multiply(tf.reduce_sum(premise,axis=1),1.0/math.sqrt(float(seq_length)))

	hypothesis = tf.layers.dense(hypothesis,hidden_size,activation=tf.nn.relu,kernel_initializer=create_initializer(initializer_range))
	hypothesis = tf.layers.dense(hypothesis,hidden_size,kernel_initializer=create_initializer(initializer_range))
	hypothesis = tf.multiply(tf.reduce_sum(hypothesis,axis=1),1.0/math.sqrt(float(seq_length)))

	comparison_result = tf.concat(premise,hypothesis,-1)
	comparison_result = tf.layers.dense(comparison_result,hidden_size,activation=tf.nn.relu,kernel_initializer=create_initializer(initializer_range))
	comparison_result = tf.layers.dense(comparison_result,3,kernel_initializer=create_initializer(initializer_range))
	#comparison_result = tf.nn.softmax(comparison_result)

	return comparison_result





def get_shape_list(tensor, expected_rank=None, name=None):
	"""Returns a list of the shape of tensor, preferring static dimensions.

	Args:
		tensor: A tf.Tensor object to find the shape of.
		expected_rank: (optional) int. The expected rank of `tensor`. If this is
			specified and the `tensor` has a different rank, and exception will be
			thrown.
		name: Optional name of the tensor for the error message.

	Returns:
		A list of dimensions of the shape of tensor. All static dimensions will
		be returned as python integers, and dynamic dimensions will be returned
		as tf.Tensor scalars.
	"""
	if name is None:
		name = tensor.name

	if expected_rank is not None:
		assert_rank(tensor, expected_rank, name)

	shape = tensor.shape.as_list()

	non_static_indexes = []
	for (index, dim) in enumerate(shape):
		if dim is None:
			non_static_indexes.append(index)

	if not non_static_indexes:
		return shape

	dyn_shape = tf.shape(tensor)
	for index in non_static_indexes:
		shape[index] = dyn_shape[index]
	return shape


def reshape_to_matrix(input_tensor):
	"""Reshapes a >= rank 2 tensor to a rank 2 tensor (i.e., a matrix)."""
	ndims = input_tensor.shape.ndims
	if ndims < 2:
		raise ValueError("Input tensor must have at least rank 2. Shape = %s" %
										 (input_tensor.shape))
	if ndims == 2:
		return input_tensor

	width = input_tensor.shape[-1]
	output_tensor = tf.reshape(input_tensor, [-1, width])
	return output_tensor


def reshape_from_matrix(output_tensor, orig_shape_list):
	"""Reshapes a rank 2 tensor back to its original rank >= 2 tensor."""
	if len(orig_shape_list) == 2:
		return output_tensor

	output_shape = get_shape_list(output_tensor)

	orig_dims = orig_shape_list[0:-1]
	width = output_shape[-1]

	return tf.reshape(output_tensor, orig_dims + [width])


def assert_rank(tensor, expected_rank, name=None):
	"""Raises an exception if the tensor rank is not of the expected rank.

	Args:
		tensor: A tf.Tensor to check the rank of.
		expected_rank: Python integer or list of integers, expected rank.
		name: Optional name of the tensor for the error message.

	Raises:
		ValueError: If the expected shape doesn't match the actual shape.
	"""
	if name is None:
		name = tensor.name

	expected_rank_dict = {}
	if isinstance(expected_rank, six.integer_types):
		expected_rank_dict[expected_rank] = True
	else:
		for x in expected_rank:
			expected_rank_dict[x] = True

	actual_rank = tensor.shape.ndims
	if actual_rank not in expected_rank_dict:
		scope_name = tf.get_variable_scope().name
		raise ValueError(
				"For the tensor `%s` in scope `%s`, the actual rank "
				"`%d` (shape = %s) is not equal to the expected rank `%s`" %
				(name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))


