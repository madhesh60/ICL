==================================================
 CAIRN EVALUATION STARTED
==================================================
Tue Nov 17 04:29:55 PM UTC 2026
Tasks: 1 3 4 2 5 6 7 8
Limit: 60
Split: dev

==================================================
 1/7 GPU CHECK
==================================================
index, name, memory.total [MiB], memory.used [MiB]
0, Tesla T4, 15360 MiB, 0 MiB
1, Tesla T4, 15360 MiB, 0 MiB
Detected GPUs: 2

==================================================
 2/7 CUDA DRIVER LIBRARY CHECK
==================================================
CUDA driver: /usr/local/nvidia/lib64/libcuda.so.1
CUDA linker directory: /kaggle/working/cuda-link
lrwxrwxrwx 1 root root 36 Nov 17 16:29 /kaggle/working/cuda-link/libcuda.so -> /usr/local/nvidia/lib64/libcuda.so.1
CUDA linker check passed.

==================================================
 3/7 GET PROJECT CODE
==================================================
Cloning into '/kaggle/working/cairn'...
Repository downloaded successfully.
f13d95a Act on the second real run: measured escalation policy, unambiguous induction, task-8 measurement

==================================================
 4/7 INSTALL DEPENDENCIES
==================================================
Installing CAIRN and vLLM...
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
gradio-client 2.7.2 requires huggingface-hub<3.0,>=1.16.0, but you have huggingface-hub 0.36.2 which is incompatible.
gradio 6.30.0 requires huggingface-hub<3.0,>=1.16.0, but you have huggingface-hub 0.36.2 which is incompatible.
gradio 6.30.0 requires pillow<13.0,>=8.0, but you have pillow 13.0.0 which is incompatible.
diffusers 0.42.0 requires huggingface-hub<3.0,>=1.32.0, but you have huggingface-hub 0.36.2 which is incompatible.
vLLM version: 0.12.0
CAIRN import: successful

==================================================
 5/7 MODEL SETUP
==================================================

Fetching 13 files:   0%|          | 0/13 [00:00<?, ?it/s]Downloading 'merges.txt' to '/tmp/Qwen3-4B/.cache/huggingface/download/PtHk0z_I45atnj23IIRhTExwT3w=.31349551d90c7606f325fe0f11bbb8bd5fa0d7c7.incomplete'
Download complete. Moving file to /tmp/Qwen3-4B/merges.txt
Downloading 'generation_config.json' to '/tmp/Qwen3-4B/.cache/huggingface/download/3EVKVggOldJcKSsGjSdoUCN1AyQ=.20a8a9156fc8c3f25295ca067f61fdf120d517c5.incomplete'
Downloading 'LICENSE' to '/tmp/Qwen3-4B/.cache/huggingface/download/DhCjcNQuMpl4FL346qr3tvNUCgY=.6634c8cc3133b3848ec74b9f275acaaa1ea618ab.incomplete'
Download complete. Moving file to /tmp/Qwen3-4B/generation_config.json
Download complete. Moving file to /tmp/Qwen3-4B/LICENSE
Downloading 'README.md' to '/tmp/Qwen3-4B/.cache/huggingface/download/Xn7B-BWUGOee2Y6hCZtEhtFu4BE=.2de5ee7eee214bb55ea33ec7505c5838a7adf7f6.incomplete'
Download complete. Moving file to /tmp/Qwen3-4B/README.md
Downloading 'config.json' to '/tmp/Qwen3-4B/.cache/huggingface/download/8_PA_wEVGiVa2goH2H4KQOQpvVY=.e49eccdc32f36da9c09cfa0e737084f9e0105e5e.incomplete'
Download complete. Moving file to /tmp/Qwen3-4B/config.json
Downloading 'model-00002-of-00003.safetensors' to '/tmp/Qwen3-4B/.cache/huggingface/download/D5-QTaCbl9VUFv9Wr0C28eFewRc=.6cd087b316306a68c562436b5492edbcf6e16c6dba3a1308279caa5a58e21ca5.incomplete'
Downloading 'model-00001-of-00003.safetensors' to '/tmp/Qwen3-4B/.cache/huggingface/download/Y6g195DXbsCiN_Ka1NwH6jyLkvc=.328a91d3122359d5547f9d79521205bc0a46e1f79a792dfe650e99fc2d651223.incomplete'
Downloading '.gitattributes' to '/tmp/Qwen3-4B/.cache/huggingface/download/wPaCkH-WbT7GsmxMKKrNZTV4nSM=.52373fe24473b1aa44333d318f578ae6bf04b49b.incomplete'
Download complete. Moving file to /tmp/Qwen3-4B/.gitattributes

Fetching 13 files:   8%|▊         | 1/13 [00:00<00:05,  2.04it/s]Downloading 'model-00003-of-00003.safetensors' to '/tmp/Qwen3-4B/.cache/huggingface/download/NmVQgThqhsRkG2FlXLh-M51MilA=.e4bf436957184f4eeb86a80e9db394503f1f56446b2e6b7edeac5b81470f4ca1.incomplete'
Downloading 'model.safetensors.index.json' to '/tmp/Qwen3-4B/.cache/huggingface/download/yVzAsSxRSINSz-tQbpx-TLpfkLU=.95c0a0059df040d75dc6c396b174382cf61d2f91.incomplete'
Downloading 'tokenizer.json' to '/tmp/Qwen3-4B/.cache/huggingface/download/HgM_lKo9sdSCfRtVg7MMFS7EKqo=.aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4.incomplete'
Download complete. Moving file to /tmp/Qwen3-4B/model.safetensors.index.json
Downloading 'tokenizer_config.json' to '/tmp/Qwen3-4B/.cache/huggingface/download/vzaExXFZNBay89bvlQv-ZcI6BTg=.417d038a63fa3de29cfde265caedae14d1a58d92.incomplete'
Download complete. Moving file to /tmp/Qwen3-4B/tokenizer_config.json
Downloading 'vocab.json' to '/tmp/Qwen3-4B/.cache/huggingface/download/j3m-Hy6QvBddw8RXA1uSWl1AJ0c=.4783fe10ac3adce15ac8f358ef5462739852c569.incomplete'
Download complete. Moving file to /tmp/Qwen3-4B/vocab.json
Download complete. Moving file to /tmp/Qwen3-4B/tokenizer.json
Download complete. Moving file to /tmp/Qwen3-4B/model-00003-of-00003.safetensors
Download complete. Moving file to /tmp/Qwen3-4B/model-00002-of-00003.safetensors
Download complete. Moving file to /tmp/Qwen3-4B/model-00001-of-00003.safetensors

Fetching 13 files:  54%|█████▍    | 7/13 [00:31<00:28,  4.73s/it]
Fetching 13 files: 100%|██████████| 13/13 [00:31<00:00,  2.45s/it]
/tmp/Qwen3-4B
RoPE configuration: {'rope_type': 'yarn', 'factor': 4.0, 'original_max_position_embeddings': 32768}
Model configuration ready.

==================================================
 6/7 START MODEL SERVER
==================================================
Starting attempt: A
Arguments: --tensor-parallel-size 2 --disable-custom-all-reduce --gpu-memory-utilization 0.85 --max-num-seqs 16
Server PID: 203
Server log: /kaggle/working/vllm_A.log
Server loading: 1 minutes
Server loading: 2 minutes
Server loading: 3 minutes
vLLM server is ready.
Testing CAIRN environment...
checking http://127.0.0.1:2026/v1 (model Qwen3-4B)

[PASS] server reachable: models: Qwen3-4B
[PASS] plain answer (thinking off): got 'OK'
[PASS] token probabilities (logprobs): 5 candidates returned
[PASS] constrained answers (structured outputs): works with structured_api: structured_outputs
[PASS] thinking mode: 1323 characters of reasoning, answer '17 + 26 = 43\n\n**Step'
[PASS] prefix cache: second call: 4016 cached tokens, 1.10s -> 0.16s

==================================================
 7/7 RUN EVALUATION
==================================================
Split: dev
Tasks: 1 3 4 2 5 6 7 8
Questions per task: 60

------------------------------------------
STARTING TASK 1
------------------------------------------
2026-11-17 16:39:51,763 INFO cairn.tokens: token counter: tiktoken with the bundled Qwen vocabulary
2026-11-17 16:43:59,918 INFO cairn.induction.program: program round 0: token budget ran out before any code
2026-11-17 16:43:59,918 INFO cairn.induction.program: program round 0: token budget ran out before any code
2026-11-17 16:43:59,918 INFO cairn.induction.program: program round 0 gave no code; retrying without thinking
2026-11-17 16:44:04,941 INFO cairn.induction.program: program round 1: 100.0% right on 150 verify examples
2026-11-17 16:44:05,003 INFO cairn.induction.program: program accepted in round 1: 150/150 verify, 150/150 confirm
2026-11-17 16:44:05,003 INFO cairn.pipeline.runner: task 1: program accepted after 3 attempt(s)
task 1 [dev] accuracy 100.0% (60/60, 95% range 94.0%..100.0%), ECE 0.000
    program          100% of inputs, accuracy 100.0%
    program: ACCEPTED after 3 attempt(s); verified on 150 + 150 held-out examples

run folder: /kaggle/working/runs/dev
Task 1 completed.

------------------------------------------
STARTING TASK 3
------------------------------------------
2026-11-17 16:44:06,471 INFO cairn.tokens: token counter: tiktoken with the bundled Qwen vocabulary
2026-11-17 16:48:13,398 INFO cairn.induction.program: program round 0: token budget ran out before any code
2026-11-17 16:48:13,398 INFO cairn.induction.program: program round 0: token budget ran out before any code
2026-11-17 16:48:13,398 INFO cairn.induction.program: program round 0 gave no code; retrying without thinking
2026-11-17 16:48:17,898 INFO cairn.induction.program: program round 1: 100.0% right on 150 verify examples
2026-11-17 16:48:17,933 INFO cairn.induction.program: program accepted in round 1: 150/150 verify, 150/150 confirm
2026-11-17 16:48:17,933 INFO cairn.pipeline.runner: task 3: program accepted after 3 attempt(s)
task 3 [dev] accuracy 100.0% (60/60, 95% range 94.0%..100.0%), ECE 0.000
    program          100% of inputs, accuracy 100.0%
    program: ACCEPTED after 3 attempt(s); verified on 150 + 150 held-out examples

run folder: /kaggle/working/runs/dev
Task 3 completed.

------------------------------------------
STARTING TASK 4
------------------------------------------
2026-11-17 16:48:19,363 INFO cairn.tokens: token counter: tiktoken with the bundled Qwen vocabulary
2026-11-17 16:50:33,431 INFO cairn.induction.program: program round 0: 100.0% right on 150 verify examples
2026-11-17 16:50:33,487 INFO cairn.induction.program: program accepted in round 0: 150/150 verify, 150/150 confirm
2026-11-17 16:50:33,487 INFO cairn.pipeline.runner: task 4: program accepted after 1 attempt(s)
task 4 [dev] accuracy 100.0% (60/60, 95% range 94.0%..100.0%), ECE 0.000
    program          100% of inputs, accuracy 100.0%
    program: ACCEPTED after 1 attempt(s); verified on 150 + 150 held-out examples

run folder: /kaggle/working/runs/dev
Task 4 completed.

------------------------------------------
STARTING TASK 2
------------------------------------------
2026-11-17 16:50:34,791 INFO cairn.tokens: token counter: tiktoken with the bundled Qwen vocabulary
2026-11-17 17:21:48,835 INFO cairn.pipeline.runner: task 2: escalation probe on 40 held-out inputs: cheap 50%, thinking 60% -> cheap_only
task 2 [dev] accuracy 43.3% (26/60, 95% range 31.6%..55.9%), ECE 0.533
    generate         100% of inputs, accuracy 43.3%
    policy: CHEAP_ONLY from a probe on 40 held-out inputs: cheap 50%, thinking 60%
        sure   n= 38 cheap 53% vs thinking 58% (only-cheap 4, only-thinking 6, p=0.754)
        unsure n=  2 cheap 0% vs thinking 100% (only-cheap 0, only-thinking 2, p=0.500)

run folder: /kaggle/working/runs/dev
Task 2 completed.

------------------------------------------
STARTING TASK 5
------------------------------------------
2026-11-17 17:22:40,706 INFO cairn.tokens: token counter: tiktoken with the bundled Qwen vocabulary
2026-11-17 17:44:59,834 INFO cairn.pipeline.runner: task 5: escalation probe on 40 held-out inputs: cheap 75%, thinking 80% -> cheap_only
task 5 [dev] accuracy 85.0% (51/60, 95% range 73.9%..91.9%), ECE 0.100
    score            100% of inputs, accuracy 85.0%
    policy: CHEAP_ONLY from a probe on 40 held-out inputs: cheap 75%, thinking 80%
        sure   n= 33 cheap 79% vs thinking 79% (only-cheap 2, only-thinking 2, p=1.000)
        unsure n=  7 cheap 57% vs thinking 86% (only-cheap 0, only-thinking 2, p=0.500)
    calibration: stack on 124 examples, held-out accuracy 78.4% -> 80.0%; sure on 76% of held-out inputs, right on 88% of those

run folder: /kaggle/working/runs/dev
Task 5 completed.

------------------------------------------
STARTING TASK 6
------------------------------------------
2026-11-17 17:45:24,087 INFO cairn.tokens: token counter: tiktoken with the bundled Qwen vocabulary
2026-11-17 18:20:32,168 INFO cairn.pipeline.runner: task 6: escalation probe on 40 held-out inputs: cheap 95%, thinking 78% -> cheap_only
task 6 [dev] accuracy 96.7% (58/60, 95% range 88.6%..99.1%), ECE 0.026
    score            100% of inputs, accuracy 96.7%
    policy: CHEAP_ONLY from a probe on 40 held-out inputs: cheap 95%, thinking 78%
        sure   n= 38 cheap 95% vs thinking 76% (only-cheap 8, only-thinking 1, p=0.039)
        unsure n=  2 cheap 100% vs thinking 100% (only-cheap 0, only-thinking 0, p=1.000)
    calibration: stack on 150 examples, held-out accuracy 74.0% -> 94.7%; sure on 95% of held-out inputs, right on 95% of those

run folder: /kaggle/working/runs/dev
Task 6 completed.

------------------------------------------
STARTING TASK 7
------------------------------------------
2026-11-17 18:21:10,823 INFO cairn.tokens: token counter: tiktoken with the bundled Qwen vocabulary
2026-11-17 18:54:00,695 INFO cairn.pipeline.runner: task 7: escalation probe on 40 held-out inputs: cheap 20%, thinking 38% -> cheap_only
task 7 [dev] accuracy 16.7% (10/60, 95% range 9.3%..28.0%), ECE 0.629
    fallback_nearest 2% of inputs, accuracy 0.0%
    generate         98% of inputs, accuracy 17.0%
    policy: CHEAP_ONLY from a probe on 40 held-out inputs: cheap 20%, thinking 38%
        sure   n= 29 cheap 28% vs thinking 45% (only-cheap 2, only-thinking 7, p=0.180)
        unsure n= 11 cheap 0% vs thinking 18% (only-cheap 0, only-thinking 2, p=0.500)

run folder: /kaggle/working/runs/dev
Task 7 completed.

------------------------------------------
STARTING TASK 8
------------------------------------------
2026-11-17 18:55:25,769 INFO cairn.tokens: token counter: tiktoken with the bundled Qwen vocabulary
task 8 [dev] static check OK 96%, runs 0%, matches PyTorch 0%, consistency score 0.000 (n=24)
    execution outcomes: {'not_run': 24}

run folder: /kaggle/working/runs/dev
Task 8 completed.

==================================================
 FINAL EVALUATION SUMMARY
==================================================

Task 1: Accuracy 100.0% (n=60, 95% CI 94%..100%) | 253s
Program: accepted=True attempts=3 best_verify=1.0

Task 2: Accuracy 43.3% (n=60, 95% CI 32%..56%) | 1925s

Task 3: Accuracy 100.0% (n=60, 95% CI 94%..100%) | 252s
Program: accepted=True attempts=3 best_verify=1.0

Task 4: Accuracy 100.0% (n=60, 95% CI 94%..100%) | 134s
Program: accepted=True attempts=1 best_verify=1.0

Task 5: Accuracy 85.0% (n=60, 95% CI 74%..92%) | 1362s
Calibration: singleton_rate=0.76 singleton_accuracy=0.8842 trusted=True

Task 6: Accuracy 96.7% (n=60, 95% CI 89%..99%) | 2146s
Calibration: singleton_rate=0.9533 singleton_accuracy=0.951 trusted=True

Task 7: Accuracy 16.7% (n=60, 95% CI 9%..28%) | 2054s

Task 8: Consistency 0.000 | Callable 0% | PyTorch Match 0% | 1289s
Outcomes: {'not_run': 24}

Reports directory: /kaggle/working/runs/dev
Summary completed.

==================================================
 EVALUATION FINISHED
==================================================
Main logs: /kaggle/working/cairn_run.log
vLLM attempt A logs: /kaggle/working/vllm_A.log
vLLM attempt B logs: /kaggle/working/vllm_B.log
Reports: /kaggle/working/runs/dev
Tue Nov 17 07:16:55 PM UTC 2026