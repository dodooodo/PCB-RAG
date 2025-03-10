from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import torch


class LLM:
    def __init__(self, model_name, system_prompt='', cache_dir=None, use_streamer=False):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            # torch_dtype=torch.float16,
            torch_dtype=torch.bfloat16,
            # load_in_8bit=True,
            device_map="auto",
            max_memory={
                0: torch.cuda.mem_get_info(0)[0], 
                1: torch.cuda.mem_get_info(1)[0], 
                2: torch.cuda.mem_get_info(2)[0],
                3: torch.cuda.mem_get_info(3)[0],
                4: torch.cuda.mem_get_info(4)[0],
                },
            # offload_cpu=True,
            # use_flash_attn_2=False,
            # use_flash_attn=False,
            # _attn_implementation='eager',
            cache_dir=cache_dir,
            trust_remote_code=True,
        ).eval()
        
        self.system_prompt = system_prompt
        
        if use_streamer:
            self.streamer = TextStreamer(self.tokenizer)
        else:
            self.streamer = None
            
    
    @property
    def device(self):
        return self.model.device
    
    @torch.inference_mode()
    def __call__(self, user_prompt: str) -> str:
        if self.tokenizer.use_default_system_prompt:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        else:
            messages = [
                {"role": "user", "content": self.system_prompt + user_prompt},
            ]

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)

        # terminators = [
        #     tokenizer.eos_token_id,
        #     tokenizer.convert_tokens_to_ids("<|eot_id|>")
        # ]L

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=1024,
            # eos_token_id=terminators,
            # do_sample=False,
            # top_p=0.9,
            # length_penalty=0,
            num_beams=6,
            low_memory=True,
            streamer=self.streamer,
        )
        response = outputs[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(response, skip_special_tokens=True)