import unittest

from transformers import ZagrosConfig, ZagrosModel, ZagrosForCausalLM
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING
from transformers.testing_utils import (MODEL_TO_MAPPING, require_torch, slow, require_torch_multi_gpu)
from transformers.utils.import_utils import is_torch_available

from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor, random_attention_mask
from ...test_pipeline_mixin import PipelineTesterMixin

if is_torch_available():
    import torch

from transformers import is_torch_available

if is_torch_available():
    from transformers.models.bert.modeling_bert import BertConfig, BertModel, BertLMHeadModel
    from transformers.models.gpt2.modeling_gpt2 import GPT2Config, GPT2LMHeadModel, GPT2Model
    from transformers.models.bert.modeling_bert import BertConfig, BertModel, BertLMHeadModel
    from transformers.models.gpt2.modeling_gpt2 import GPT2Config, GPT2LMHeadModel, GPT2Model

class ZagrosModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_token_type_ids=True,
        use_labels=True,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=5,
        num_attention_heads=4,
        intermediate_size=37,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=16,
        type_sequence_label_size=2,
        initializer_range=0.02,
        num_labels=3,
        num_choices=4,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_token_type_ids = use_token_type_ids
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.scope = scope
        self.num_experts = 4
        self.num_experts_per_tok = 2

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        attention_mask = random_attention_mask([self.batch_size, self.seq_length])

        token_type_ids = None
        if self.use_token_type_ids:
            token_type_ids = ids_tensor([self.batch_size, self.seq_length], self.type_vocab_size)

        config = self.get_config()

        return config, input_ids, token_type_ids, attention_mask

    def get_config(self):
        return ZagrosConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            initializer_range=self.initializer_range,
            num_experts=self.num_experts,
            num_experts_per_tok=self.num_experts_per_tok,
        )

    def create_and_check_model(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = ZagrosModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask)
        result = model(input_ids, token_type_ids=token_type_ids, attention_mask=input_mask)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_for_causal_lm(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
    ):
        model = ZagrosForCausalLM(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, labels=sequence_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            token_type_ids,
            sequence_labels,
            token_labels,
            choice_labels,
            ) = config_and_inputs
        inputs_dict = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": input_mask,
        }
        return config, inputs_dict


@require_torch
class ZagrosModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (ZagrosModel, ZagrosForCausalLM)  # Add other classes if needed
    pipeline_model_mapping = {
        "feature-extraction": ZagrosModel,
        "text-generation": ZagrosForCausalLM,
    }
    fx_compatible = True
    test_head_masking = True
    test_pruning = True
    test_resize_embeddings = True
    test_model_parallel = True

    def setUp(self):
        self.model_tester = ZagrosModelTester(self)
        self.config_tester = ConfigTester(self, config_class=ZagrosConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_causal_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_causal_lm(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        for model_name in MODEL_TO_MAPPING.values():
            if model_name == "ZagrosModel":
                return
        super().test_model_from_pretrained()

    @slow
    def test_saved_model_configuration(self):
        for model_name in MODEL_TO_MAPPING.values():
            if model_name == "ZagrosModel":
                return
        super().test_saved_model_configuration()

if __name__ == "__main__":
    unittest.main()