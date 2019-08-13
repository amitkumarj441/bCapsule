# -*- coding: utf-8 -*-
from torch.nn import CrossEntropyLoss

from collections import OrderedDict
from capsule_net import capsnet
import torch
from torch import nn
from  pytorch_pretrained_bert  import  BertPreTrainedModel,BertModel
class BertForSequenceClassification(BertPreTrainedModel):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_labels].

    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    num_labels = 2

    model = BertForSequenceClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config, num_labels):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits


class BertForSequenceCapsuleClassification_meta(BertPreTrainedModel):
    """BERT model for classification with capsule.
        using MAML to capsule weights.
    """
    def __init__(self, config, num_labels, output_atoms=30, lr_a=0.00005):
        super(BertForSequenceCapsuleClassification_meta, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # ---

        self.capsule_layer = capsnet.CapsuleLayer(1, output_dim=num_labels, input_atoms=768, output_atoms=output_atoms, num_routing=1, leaky=True)
        self.capsule_classification = capsnet.CapsuleClassification()

        self.lr_a = lr_a

        self.apply(self.init_bert_weights)

    def forward_pass(self, fast_weights, input_ids, token_type_ids=None, attention_mask=None, labels=None):

        onehot_labels = torch.zeros(labels.shape[0], self.num_labels).to(labels.device.type).scatter_(dim=1, index=labels.unsqueeze(1), value=1.)

        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        pooled_output = pooled_output.unsqueeze(1)  # [b, 1, hid]

        capsule_layer_output = self.capsule_layer(pooled_output, fast_weights)
        logits, probs = self.capsule_classification(capsule_layer_output)
        loss = self.capsule_layer._margin_loss(onehot_labels, logits)
        return loss

    def forward(self, train_dataloader, test_data, device):

        fast_weights = OrderedDict((name, param) for (name, param) in self.capsule_layer.named_parameters())
        test_input_ids = test_data[0].to(device)
        test_token_type_ids = test_data[1].to(device)
        test_attention_mask = test_data[2].to(device)
        test_labels = test_data[3].to(device)

        for _ in range(3):  # epoch
            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch

                onehot_labels = torch.zeros(label_ids.shape[0], self.num_labels).to(label_ids.device.type).scatter_(dim=1, index=label_ids.unsqueeze(1), value=1.)

                _, pooled_output = self.bert(input_ids, segment_ids, input_mask, output_all_encoded_layers=False)
                pooled_output = self.dropout(pooled_output)
                pooled_output = pooled_output.unsqueeze(1)  # [b, 1, hid]

                if step == 0:
                    capsule_layer_output = self.capsule_layer(pooled_output)
                    logits, probs = self.capsule_classification(capsule_layer_output)
                    loss = self.capsule_layer._margin_loss(onehot_labels, logits)
                    grads = torch.autograd.grad(loss, self.capsule_layer.parameters(), create_graph=True)
                else:
                    capsule_layer_output = self.capsule_layer(pooled_output, fast_weights)
                    logits, probs = self.capsule_classification(capsule_layer_output)
                    loss = self.capsule_layer._margin_loss(onehot_labels, logits)
                    grads = torch.autograd.grad(loss, fast_weights.values(), create_graph=True)

                fast_weights = OrderedDict((name, param - self.lr_a * grad) for ((name, param), grad) in zip(fast_weights.items(), grads))

        meta_loss = self.forward_pass(fast_weights, test_input_ids, test_token_type_ids, test_attention_mask, test_labels)
        # meta_grads = torch.autograd.grad(meta_loss, self.parameters())
        # meta_grads = {name: g for ((name, _), g) in zip(self.named_parameters(), meta_grads)}

        return meta_loss   # meta_grads


class BertForSequenceCapsuleClassification(BertPreTrainedModel):
    """BERT model for classification with capsule.
    """
    def __init__(self, config, num_labels, output_atoms=30, meta_weights=None):
        super(BertForSequenceCapsuleClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # ---
        if meta_weights is None:
            self.capsule_layer = capsnet.CapsuleLayer(1, output_dim=num_labels, input_atoms=768, output_atoms=output_atoms, num_routing=1, leaky=True)
        else:
            self.capsule_layer = capsnet.CapsuleLayer(1, output_dim=num_labels, input_atoms=768,
                                                              output_atoms=output_atoms, paras=meta_weights, num_routing=1, leaky=True)
        self.capsule_classification = capsnet.CapsuleClassification()

        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        pooled_output = pooled_output.unsqueeze(1)  # [b, 1, hid]


        capsule_layer_output = self.capsule_layer(pooled_output)
        logits, probs = self.capsule_classification(capsule_layer_output)

        if labels is not None:
            onehot_labels = torch.zeros(labels.shape[0], self.num_labels).to(labels.device.type).scatter_(dim=1, index=labels.unsqueeze(1), value=1.)
            loss = self.capsule_layer._margin_loss(onehot_labels, logits)
            return loss
        else:
            return probs