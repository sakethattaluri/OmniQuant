import transformers
import torch
from .models_utils import BaseLM, find_layers
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import torch.nn.functional as F
from torch import nn
import torch
import numpy as np
from tqdm import tqdm
import pdb


class LMClass(BaseLM):
    def __init__(self, args):

        super().__init__()

        self.args = args
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = args.model
        self.batch_size_per_gpu = args.batch_size

        self.model_config = args.model
        config = AutoConfig.from_pretrained(
            args.model, attn_implementation=args.attn_implementation
        )

        self.tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False,legacy=False,trust_remote_code=True)
        # self.model = AutoModelForCausalLM.from_pretrained(args.model, config=config, device_map='cpu',torch_dtype=config.torch_dtype)
        self.model = AutoModelForCausalLM.from_pretrained(args.model, config=config, device_map='cpu',torch_dtype=torch.float16,trust_remote_code=True)
        # bias_list = ['model.layers.0.self_attn.o_proj.bias', 'model.layers.1.self_attn.o_proj.bias', 'model.layers.10.self_attn.o_proj.bias', 'model.layers.11.self_attn.o_proj.bias', 'model.layers.12.self_attn.o_proj.bias', 'model.layers.13.self_attn.o_proj.bias', 'model.layers.14.self_attn.o_proj.bias', 'model.layers.15.self_attn.o_proj.bias', 'model.layers.16.self_attn.o_proj.bias', 'model.layers.17.self_attn.o_proj.bias', 'model.layers.18.self_attn.o_proj.bias', 'model.layers.19.self_attn.o_proj.bias', 'model.layers.2.self_attn.o_proj.bias', 'model.layers.20.self_attn.o_proj.bias', 'model.layers.21.self_attn.o_proj.bias', 'model.layers.22.self_attn.o_proj.bias', 'model.layers.23.self_attn.o_proj.bias', 'model.layers.24.self_attn.o_proj.bias', 'model.layers.25.self_attn.o_proj.bias', 'model.layers.26.self_attn.o_proj.bias', 'model.layers.27.self_attn.o_proj.bias', 'model.layers.28.self_attn.o_proj.bias', 'model.layers.29.self_attn.o_proj.bias', 'model.layers.3.self_attn.o_proj.bias', 'model.layers.30.self_attn.o_proj.bias', 'model.layers.31.self_attn.o_proj.bias', 'model.layers.4.self_attn.o_proj.bias', 'model.layers.5.self_attn.o_proj.bias', 'model.layers.6.self_attn.o_proj.bias', 'model.layers.7.self_attn.o_proj.bias', 'model.layers.8.self_attn.o_proj.bias', 'model.layers.9.self_attn.o_proj.bias']
        # val = torch.load("/auto/regrt/sw/titash/qwen_int4/int4weights.dat", map_location="cpu")
        # print(self.model.state_dict().keys())
        
        # for k in self.model.state_dict().keys():
        #     if 'down_proj' not in k:
        #         val[k] = self.model.state_dict()[k]
        # self.model.load_state_dict(val)
        self.seqlen = self.model.config.max_position_embeddings
        self.model.eval()
        self.vocab_size = self.tokenizer.vocab_size
        print("vocab size: ", self.vocab_size)

    @property
    def eot_token(self) -> str:
        return self.tokenizer.eos_token

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        try:
            return self.gpt2.config.n_ctx
        except AttributeError:
            # gptneoconfig doesn't have n_ctx apparently
            return self.model.config.max_position_embeddings

    @property
    def max_gen_toks(self):
        print("max_gen_toks fn")
        return 256

    @property
    def batch_size(self):
        # TODO: fix multi-gpu
        return self.batch_size_per_gpu  # * gpus

    @property
    def device(self):
        # TODO: fix multi-gpu
        return self._device

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_encode_batch(self, strings):
        return self.tokenizer(
            strings,
            padding=True,
            add_special_tokens=False,
            return_tensors="pt",
        )

    def tok_decode(self, tokens):
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call
        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        with torch.no_grad():

            return self.model(inps)["logits"]

    def model_batched_set(self, inps):
        dataset_logits = []
        for batch in inps:
            multi_logits = F.log_softmax(
                self._model_call(batch), dim=-1
            ).cpu()  # [batch, padding_length, vocab]
            dataset_logits.append(multi_logits)
        return dataset_logits

    def _model_generate(self, context, max_length, eos_token_id):
        return self.model.generate(
            context, max_length=max_length, eos_token_id=eos_token_id, do_sample=False
        )
