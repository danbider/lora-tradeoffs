# LoRA Learns Less and Forgets Less

This minimal repo contains information for the paper ["LoRA Learns Less and Forgets Less"](https://arxiv.org/abs/2405.09673) (Biderman et al. TMLR, 2024).

Model checkpoints and LoRA adapters can be found here: [https://huggingface.co/LoRA-TMLR-2024](https://huggingface.co/LoRA-TMLR-2024)


| Setting | Dataset | Link |
| --------| ------| ------ |
| Continued Pretraining - Code | StarCoder-Python| [LoRA-TMLR-2024/continued-pretraining-code-starcoder-python](https://huggingface.co/collections/LoRA-TMLR-2024/continued-pretraining-code-starcoder-python-66f22ce3b26f416f21f58142) |
| Continued Pretraing - Math | [OpenWebMath](https://huggingface.co/datasets/open-web-math/open-web-math) | TBD |
| Instruction Finetuning - Code | [Magicoder-Evol-Instruct-110K](https://huggingface.co/datasets/ise-uiuc/Magicoder-Evol-Instruct-110K)| [LoRA-TMLR-2024/instruction-finetuning-code-magicoder-evol-instruct-110k](https://huggingface.co/collections/LoRA-TMLR-2024/instruction-finetuning-code-magicoder-evol-instruct-110k-66f224a800152f31e4942a3b) |
| Instruction Finetuning - Math | [MetaMathQA](https://huggingface.co/datasets/meta-math/MetaMathQA) | TBD |

All training was done using the Databricks MosaicML
[composer](https://github.com/mosaicml/composer), [streaming](https://github.com/mosaicml/streaming), and [llm-foundry](https://github.com/mosaicml/llm-foundry) repositories, as well as the HuggingFace peft library


----

5/15/2024 - v1 of the paper shared on arXiv

8/13/2024 - Paper [accepted to TMLR](https://openreview.net/forum?id=aloEru2qCG)

9/23/2024 - arXiv v2 updated (same as TMLR camera ready version)

9/24/2024 - Model checkpoints uploaded to HuggingFace (WIP)

-----


In all four scenarios below, we use the Llama-2-7B base model [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf). For
the CPT runs, we use the [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf) tokenizer, while for the IFT runs we use the
[meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) tokenizer.

### Code CPT

| Parameter                    | Value                                                                                   |
|------------------------------|-----------------------------------------------------------------------------------------|
| seq_len                      | 4096                                                                                    |
| optimizer                    | decoupled_lionw (betas=[0.9, 0.95])                                                     |
| learning_rate                | 1.0e-05 for LoRA and Full Finetuning                                                    |
| scheduler                    | inv_sqrt_with_warmup (t_scale=1000ba, t_warmup=1000ba, t_cooldown=5086ba, alpha_f_decay=1, alpha_f_cooldown=0) |
| weight_decay                 | 1.0e-06                                                                                 |
| precision                    | amp_bf16                                                                                |
| global_train_batch_size      | 192                                                                                     |
| device_train_microbatch_size | 6                                                                                       |
| gradient_clipping            | norm (threshold=1)                                                                      |
| num_gpus                     | 32                                                                                      |

### Math CPT

| Parameter                    | Value                                                                                   |
|------------------------------|-----------------------------------------------------------------------------------------|
| max_seq_len                  | 4096                                                                                    |
| optimizer                    | decoupled_lionw (betas=[0.9, 0.95])                                                     |
| learning_rate                | 1.0e-05 for full finetuning, 4.0e-05 for LoRA                                           |
| scheduler                    | inv_sqrt_with_warmup (t_scale=1000ba, t_warmup=1000ba, t_cooldown=5086ba, alpha_f_decay=1, alpha_f_cooldown=0) |
| weight_decay                 | 0                                                                                       |
| precision                    | amp_bf16                                                                                |
| global_train_batch_size      | 192                                                                                     |
| device_train_microbatch_size | 6                                                                                       |
| gradient_clipping            | norm (threshold=1)                                                                      |
| num_gpus                     | 32                                                                                      |

### Code IFT

| Parameter                    | Value                                                                                   |
|------------------------------|-----------------------------------------------------------------------------------------|
| max_seq_len                  | 4096                                                                                    |
| optimizer                    | decoupled_lionw (betas=[0.9, 0.95])                                                     |
| learning_rate                | 2e-4 for rank r = 16, 64 and 1e-4 for r = 256 α = 2r = 512 (due to instabilities/loss spikes at 2e-4) |
| scheduler                    | cosine_with_warmup (alpha_f=0.01, t_warmup=0.1dur)                                      |
| weight_decay                 | 0                                                                                       |
| precision                    | amp_bf16                                                                                |
| global_train_batch_size      | 192                                                                                     |
| device_train_microbatch_size | 6                                                                                       |
| gradient_clipping            | norm (threshold=1)                                                                      |
| num_gpus                     | 32                                                                                      |

### Math IFT

| Parameter                    | Value                                                                                   |
|------------------------------|-----------------------------------------------------------------------------------------|
| seq_len                      | 1024                                                                                    |
| optimizer                    | decoupled_lionw (betas=[0.9, 0.95])                                                     |
| learning_rate                | Full finetuning: 1e-5, LoRA: 1e-4 for r = 16, 64, 5e-5 for r = 256 due to instabilities |
| scheduler                    | cosine_with_warmup (alpha_f=0.01, t_warmup=0.1dur)                                      |
| weight_decay                 | 0                                                                                       |
| precision                    | amp_bf16                                                                                |
| global_train_batch_size      | 768                                                                                     |
| device_train_microbatch_size | 24                                                                                      |
| gradient_clipping            | norm (threshold=1)                                                                      |
| num_gpus                     | 32                                                                                      |
```


