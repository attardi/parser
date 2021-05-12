# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from supar.modules.scalar_mix import ScalarMix
from supar.utils.fn import pad


class TransformerEmbedding(nn.Module):
    r"""
    A module that directly utilizes the pretrained models in `transformers`_ to produce BERT representations.
    While mainly tailored to provide input preparation and post-processing for the BERT model,
    it is also compatible with other pretrained language models like XLNet, RoBERTa and ELECTRA, etc.

    Args:
        model (str):
            Path or name of the pretrained models registered in `transformers`_, e.g., ``'bert-base-cased'``.
        n_layers (int):
            The number of BERT layers to use. If 0, uses all layers.
        n_out (int):
            The requested size of the embeddings. If 0, uses the size of the pretrained embedding model. Default: 0.
        stride (int):
            A sequence longer than max length will be split into several small pieces
            with a window size of ``stride``. Default: 256.
        pooling (str):
            Pooling way to get from token piece embeddings to token embedding.
            ``first``: take the first subtoken. ``last``: take the last subtoken. ``mean``: take a mean over all.
            Default: ``mean``.
        pad_index (int):
            The index of the padding token in BERT vocabulary. Default: 0.
        dropout (float):
            The dropout ratio of BERT layers. Default: 0. This value will be passed into the :class:`ScalarMix` layer.
        requires_grad (bool):
            If ``True``, the model parameters will be updated together with the downstream task. Default: ``False``.

    .. _transformers:
        https://github.com/huggingface/transformers
    """

    def __init__(self, model, n_layers, n_out=0, stride=256, pooling='mean', pad_index=0, dropout=0, requires_grad=False,
                 use_attentions=True, # attention
                 attention_head=0, attention_layer=8):
        super().__init__()

        from transformers import AutoConfig, AutoModel, AutoTokenizer
        self.bert = AutoModel.from_pretrained(model, config=AutoConfig.from_pretrained(model, output_hidden_states=True, output_attentions=use_attentions))
        self.bert = self.bert.requires_grad_(requires_grad)

        self.model = model
        self.n_layers = n_layers or self.bert.config.num_hidden_layers
        self.hidden_size = self.bert.config.hidden_size
        self.n_out = n_out or self.hidden_size
        self.stride = stride
        self.pooling = pooling
        self.pad_index = pad_index
        self.dropout = dropout
        self.requires_grad = requires_grad
        self.max_len = int(max(0, self.bert.config.max_position_embeddings) or 1e12) - 2
        self.use_attentions = use_attentions
        self.head = attention_head
        self.attention_layer = attention_layer

        self.tokenizer = AutoTokenizer.from_pretrained(model)

        self.scalar_mix = ScalarMix(self.n_layers, dropout)
        self.projection = nn.Linear(self.hidden_size, self.n_out, False) if self.hidden_size != n_out else nn.Identity()

    def __repr__(self):
        s = f"{self.model}, n_layers={self.n_layers}, n_out={self.n_out}, "
        s += f"stride={self.stride}, pooling={self.pooling}, pad_index={self.pad_index}"
        if self.dropout > 0:
            s += f", dropout={self.dropout}"
        if self.requires_grad:
            s += f", requires_grad={self.requires_grad}"

        return f"{self.__class__.__name__}({s})"

    def forward(self, subwords):
        r"""
        Args:
            subwords (~torch.Tensor): ``[batch_size, seq_len, fix_len]``.
        Returns:
            ~torch.Tensor:
                BERT embeddings of shape ``[batch_size, seq_len, n_out]``.
        """

        batch_size, seq_len, fix_len = subwords.shape # attention
        mask = subwords.ne(self.pad_index)
        lens = mask.sum((1, 2))
        # [batch_size, n_subwords]
        subwords = pad(subwords[mask].split(lens.tolist()), self.pad_index, padding_side=self.tokenizer.padding_side)
        bert_mask = pad(mask[mask].split(lens.tolist()), 0, padding_side=self.tokenizer.padding_side)

        # return the hidden states of all layers
        # Outputs from the transformer:
        # - last_hidden_state: [batch, seq_len, hidden_size]
        # - pooler_output: [batch, hidden_size],
        # - hidden_states (optional): [[batch_size, seq_length, hidden_size]] * (1 + layers)
        # - attentions (optional): [[batch_size, num_heads, seq_length, seq_length]] * layers
        # print('<BERT, GPU MiB:', memory_allocated() // (1024*1024)) # DEBUG
        outputs = self.bert(subwords[:, :self.max_len], attention_mask=bert_mask[:, :self.max_len].float())
        bert_idx = -2 if self.use_attentions else -1
        bert = outputs[bert_idx]
        # [n_layers, batch_size, max_len, hidden_size]
        bert = bert[-self.n_layers:]
        # [batch_size, max_len, hidden_size]
        bert = self.scalar_mix(bert)
        if self.use_attentions:
            # [batch_size, num_heads, min(sent_len, max_len), min(sent_len, max_len)]
            al = outputs[-1][self.attention_layer]
            num_heads = al.shape[1]
            n_subwords = subwords.shape[1]
            # [batch_size, num_heads, seq_len, seq_len]
            attn_layer = bert.new_zeros((batch_size, num_heads,
                                         n_subwords, n_subwords))
            attn_layer[:,:,:al.shape[2],:al.shape[3]] = al
        # [batch_size, n_subwords, hidden_size]
        for i in range(self.stride, (subwords.shape[1]-self.max_len+self.stride-1)//self.stride*self.stride+1, self.stride):
            part = self.bert(subwords[:, i:i+self.max_len], attention_mask=bert_mask[:, i:i+self.max_len].float())
            bert = torch.cat((bert, self.scalar_mix(part[bert_idx][-self.n_layers:])[:, self.max_len-self.stride:]), 1)
            if self.use_attentions:
                # [batch, n_subwords, n_subwords]
                part_al = part[-1][self.attention_layer]
                attn_layer[:,:,i:i+part_al.shape[2],i:i+part_al.shape[3]] = part_al
        # [batch_size, seq_len]
        bert_lens = mask.sum(-1)
        bert_lens = bert_lens.masked_fill_(bert_lens.eq(0), 1)
        # [batch_size, seq_len, fix_len, hidden_size]
        embed = bert.new_zeros(*mask.shape, self.hidden_size).masked_scatter_(mask.unsqueeze(-1), bert[bert_mask])
        # [batch_size, seq_len, hidden_size]
        if self.pooling == 'first':
            embed = embed[:, :, 0]
        elif self.pooling == 'last':
            embed = embed.gather(2, (bert_lens-1).unsqueeze(-1).repeat(1, 1, self.hidden_size).unsqueeze(2)).squeeze(2)
        else:
            embed = embed.sum(2) / bert_lens.unsqueeze(-1)
        # Attention
        seq_attn = None
        if self.use_attentions:
            # [batch, n_subwords, n_subwords]
            attn = attn_layer[:,self.head,:,:]
            # squeeze out multiword tokens
            mask2 = ~mask
            mask2[:,:,0] = True  # keep first column
            sub_masks = pad(mask2[mask].split(lens.tolist()), 0, padding_side=self.tokenizer.padding_side)
            seq_mask = torch.einsum('bi,bj->bij', sub_masks, sub_masks)  # outer product
            seq_lens = seq_mask.sum((1,2))
            # [batch_size, seq_len, seq_len]
            sub_attn = attn[seq_mask].split(seq_lens.tolist())
            # fill a tensor [batch_size, seq_len, seq_len]
            seq_attn = attn.new_zeros(batch_size, seq_len, seq_len)
            for i, attn_i in enumerate(sub_attn):
                size = sub_masks[i].sum(0)
                attn_i = attn_i.view(size, size)
                seq_attn[i,:size,:size] = attn_i
        embed = self.projection(embed)

        return embed, seq_attn  # Attardi
