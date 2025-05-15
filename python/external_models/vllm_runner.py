import torch
import numpy as np
from loguru import logger
from typing import List, Tuple
from transformers import AutoTokenizer

try:
    from vllm import LLM, SamplingParams
except ImportError as e:
    print("Cannot import vllm")
    pass
from .external_parser import *


class VLLMTacticGenerator(Generator, Transformer):
    def __init__(self, **args) -> None:
        self.name = args["model"]
        self.llm = LLM(
            model=self.name,
            tokenizer=self.name,
            tensor_parallel_size=args["tensor_parallel_size"],
            enforce_eager=True,
            max_model_len=4096,
            disable_custom_all_reduce=False,
            trust_remote_code=True,
        )
        self.sampling_params = SamplingParams(
            n=args["n"],
            max_tokens=args["max_tokens"],
            temperature=args["temperature"],
            top_p=args["top_p"],
            frequency_penalty=0,
            presence_penalty=0,
            use_beam_search=args.get("use_beam_search", False),
        )
        
        self.skip_exp = args.get("skip_exp", False)
        self.max_output = args.get("max_output", -1)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.name, trust_remote_code=True
        )
        device = args["device"]
        if device == "auto":
            device = get_cuda_if_available()
        else:
            device = torch.device(device)
        logger.info(f"Loading {self.name} on {device}")

    def generate(self, input: str, target_prefix: str = "") -> List[Tuple[str, float]]:
        prompt = input + target_prefix
        '''prompt= 'Here is a theorom you need to prove in Lean:\n'+prompt+'\nNow you should suggest one line tactic in lean code:'
        prompt = f"""<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"""
        '''
        prompt = pre_process_input(self.name, prompt)

        vllm_outputs = self.llm.generate(prompt, self.sampling_params)
        result = []
        for output in vllm_outputs[0].outputs:  # bsz=1 for now
            out = output.text.split("<|im_end|>")[0]
            result.append(
                (
                    post_process_output(self.name, out), 
                    output.cumulative_logprob if self.skip_exp else np.exp(output.cumulative_logprob)
                )
            )

        result = choices_dedup(result)
        if self.max_output > 0:
            result = sample_outputs_by_logprob(result, self.max_output)
        return result


if __name__ == "__main__":
    generation_kwargs = {
        "model": "bytedance-research/BFS-Prover",
        "tensor_parallel_size": 2,
        "temperature": 1.1,
        "max_tokens": 64,
        "top_p": 1,
        "length_penalty": 0,
        "n": 8,
        "do_sample": True,
        "output_scores": True,
        "output_logits": False,
        "return_dict_in_generate": True,
        "device": "auto",
        "skip_exp": True,
        "max_output": 4,
        # use_beam_search=True,
    }
    model = VLLMTacticGenerator(**generation_kwargs)
    print(model.generate("n : ℕ\n⊢ gcd n n = n"))
