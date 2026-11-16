# 05 — Kaggle GPU Setup Guide: From Zero to Running Qwen3-4B

This guide walks you through setting up your ICL annotation solution on **Kaggle's free GPU**, downloading Qwen3-4B, and running your first task.

---

## **STEP 1: Choose Your Platform**

### **Best for you: Kaggle (Free + Easiest)**
- ✅ 2× T4 GPUs (16 GB each)
- ✅ 30 hours/week free compute
- ✅ Comes with Jupyter, internet access
- ✅ Can mount datasets and download models
- ✅ Sessions save outputs to disk automatically

**Alternative: Google Colab**
- ✅ Free + Optional paid ($10/month for L4 24GB, better for long context)
- ✅ Pre-configured with Python/PyTorch
- ⚠️ Limited free GPU hours (12 hours/session, may get disconnected)

---

## **STEP 2: Set Up on Kaggle (Do This First)**

### 2.1 Create Account and Notebook

1. Go to https://www.kaggle.com/ → Sign up or Login
2. Click "Create" → "Notebook"
3. Select **Python** environment
4. Name it: `icl-qwen3-annotation`

### 2.2 Check GPU Availability

In the first cell, run:

```python
!nvidia-smi
```

**Expected output:**
```
NVIDIA-SMI 12.2
GPU 0: Tesla T4 (16GB)
GPU 1: Tesla T4 (16GB)
```

If you don't see GPUs, click **⚙️ Settings** → Verify "GPU T4 x2" is selected.

### 2.3 Install Dependencies

```bash
!pip install transformers torch vllm pydantic python-dotenv openai -q
!pip install rank_bm25 scikit-learn tqdm -q
```

### 2.4 Download Qwen3-4B (First Time = 5–10 mins)

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

print("Downloading Qwen3-4B tokenizer...")
model_name = "Qwen/Qwen3-4B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
print(f"✓ Tokenizer loaded. Vocab size: {len(tokenizer)}")

print("Downloading Qwen3-4B model (~8 GB)...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # Use fp16 to fit on T4
    device_map="auto",
    trust_remote_code=True
)
print(f"✓ Model loaded. Device: {next(model.parameters()).device}")
```

---

## **STEP 3: Start vLLM Server (Critical!)**

Instead of loading the model directly, run it as an **OpenAI-compatible API server** so your code stays clean and reusable.

### 3.1 Install vLLM

```bash
!pip install vllm -q
```

### 3.2 Launch the Server

In a new notebook cell:

```python
import subprocess
import time

print("Starting vLLM server with prefix caching...")
# Start vLLM with 2 GPUs, prefix caching, 40K context
process = subprocess.Popen(
    [
        "vllm", "serve", "Qwen/Qwen3-4B",
        "--tensor-parallel-size", "2",
        "--gpu-memory-utilization", "0.95",
        "--max-model-len", "40000",
        "--port", "8000",
        "--enable-prefix-caching",
        "--dtype", "float16"
    ],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)

print("Waiting for server to start (30 seconds)...")
time.sleep(30)

# Check if server is running
import requests
for i in range(5):
    try:
        response = requests.get("http://localhost:8000/health")
        if response.status_code == 200:
            print("✓ vLLM server is running!")
            break
    except:
        if i < 4:
            print(f"  Attempt {i+1}/5 - server not ready yet, waiting...")
            time.sleep(5)
        else:
            print("✗ Server failed to start. Check logs above.")
```

### 3.3 Test the Server

```python
import requests
import json

url = "http://localhost:8000/v1/completions"
payload = {
    "model": "Qwen3-4B",
    "prompt": "Hello, my name is",
    "max_tokens": 20,
    "temperature": 0.1
}

print("Testing vLLM API...")
response = requests.post(url, json=payload)
result = response.json()

if "choices" in result:
    print("✓ Server is working!")
    print(f"Response: {result['choices'][0]['text']}")
else:
    print("✗ Server error:")
    print(result)
```

---

## **STEP 4: Run Your First Annotation Task**

Create a new notebook cell to test the annotation pipeline:

```python
from openai import OpenAI

# Connect to local vLLM server
client = OpenAI(
    api_key="dummy",  # Dummy key for local server
    base_url="http://localhost:8000/v1"
)

# Minimal test: Task 1 (closest integers)
prompt = """You are an annotation assistant. Given demonstrations and a query, follow the same pattern.

DEMONSTRATIONS:
Example 1: Input: [1, 2, 5] → Output: [1, 2]
Example 2: Input: [10, 20, 15] → Output: [10, 15]
Example 3: Input: [3, 7, 5] → Output: [3, 5]

QUERY: Input: [4, 8, 6]
Output:"""

print("Running annotation on Task 1 sample...")
response = client.completions.create(
    model="Qwen3-4B",
    prompt=prompt,
    max_tokens=50,
    temperature=0.1
)

answer = response.choices[0].text.strip()
print(f"Model answer: {answer}")
```

---

## **STEP 5: Build Your Solution Structure**

Create these Python files locally on your laptop (or in Kaggle, then download).

### 5.1 Create Directory Structure

```bash
# On your laptop:
cd C:\Users\RAJ\Desktop\PROJs\ICL
mkdir -p solution\{data,llm,tasks,eval,runs}
cd solution
```

### 5.2 Create `solution/llm/client.py`

```python
"""OpenAI-compatible client for Qwen3-4B inference."""

from openai import OpenAI
from typing import List, Dict, Any
import time

class QwenClient:
    """Wrapper around vLLM OpenAI API."""
    
    def __init__(self, base_url: str = "http://localhost:8000/v1", api_key: str = "dummy"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = "Qwen3-4B"
    
    def complete(self, prompt: str, max_tokens: int = 100, temperature: float = 0.1) -> str:
        """Generate completion from prompt."""
        try:
            response = self.client.completions.create(
                model=self.model_name,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].text
        except Exception as e:
            print(f"Error in completion: {e}")
            return ""
    
    def chat(self, messages: List[Dict[str, str]], max_tokens: int = 100) -> str:
        """Generate chat response."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.1
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in chat: {e}")
            return ""
    
    def logprobs(self, prompt: str, max_tokens: int = 1) -> Dict[str, Any]:
        """Get log probabilities for next token."""
        try:
            response = self.client.completions.create(
                model=self.model_name,
                prompt=prompt,
                max_tokens=max_tokens,
                logprobs=5
            )
            return response.choices[0].logprobs
        except Exception as e:
            print(f"Error getting logprobs: {e}")
            return {}
```

### 5.3 Create `solution/data/load_tasks.py`

```python
"""Load task data from OpenSeek repository."""

import json
import os
from typing import Tuple, List, Dict, Any

TASK_NAMES = {
    1: "closest_integers",
    2: "count_nouns_verbs",
    3: "collatz",
    4: "concat_strings",
    5: "sadness_classification",
    6: "same_genre",
    7: "jeopardy_qa",
    8: "kernel_generation"
}

class TaskDataLoader:
    """Load and manage task data."""
    
    def __init__(self, data_dir: str = "../OpenSeek/openseek/competition/LongContext-ICL-Annotation/data"):
        self.data_dir = data_dir
    
    def load_task(self, task_num: int) -> Tuple[List[Dict], List[Dict]]:
        """Load demonstrations and test samples for a task."""
        task_name = TASK_NAMES[task_num]
        task_dir = os.path.join(self.data_dir, f"task{task_num}")
        
        # Load demonstrations
        demo_file = os.path.join(task_dir, "demo.json")
        with open(demo_file, "r") as f:
            demos = json.load(f)
        
        # Load test samples
        test_file = os.path.join(task_dir, "test.json")
        with open(test_file, "r") as f:
            test_data = json.load(f)
        
        print(f"✓ Task {task_num} ({task_name}): {len(demos)} demos, {len(test_data)} test samples")
        return demos, test_data
    
    def load_all_tasks(self) -> Dict[int, Tuple[List[Dict], List[Dict]]]:
        """Load all 8 tasks."""
        tasks = {}
        for task_num in range(1, 9):
            try:
                demos, test_data = self.load_task(task_num)
                tasks[task_num] = (demos, test_data)
            except FileNotFoundError as e:
                print(f"✗ Task {task_num} not found: {e}")
        return tasks
```

### 5.4 Create `solution/run.py` (Minimal Baseline)

```python
"""Minimal baseline: run all 8 tasks."""

import sys
import json
import os
from tqdm import tqdm

# Add paths
sys.path.insert(0, os.path.dirname(__file__))

from data.load_tasks import TaskDataLoader
from llm.client import QwenClient

def format_demo_baseline(demos: list, num_shots: int = 5) -> str:
    """Format first N demos as context."""
    demo_text = ""
    for i, demo in enumerate(demos[:num_shots]):
        demo_text += f"Example {i+1}: Input: {demo.get('input', 'N/A')} → Output: {demo.get('output', 'N/A')}\n"
    return demo_text

def run_task(task_num: int, num_shots: int = 5, max_samples: int = None):
    """Run a single task on test set."""
    
    loader = TaskDataLoader()
    demos, test_data = loader.load_task(task_num)
    
    if max_samples:
        test_data = test_data[:max_samples]
    
    client = QwenClient()
    
    results = []
    print(f"\nRunning Task {task_num} baseline ({len(test_data)} samples)...")
    
    for i, sample in enumerate(tqdm(test_data, desc=f"Task {task_num}")):
        # Build prompt
        demo_context = format_demo_baseline(demos, num_shots)
        prompt = f"""Follow the pattern from these examples:

{demo_context}

Now annotate this input:
Input: {sample.get('input', 'N/A')}
Output:"""
        
        # Get response
        answer = client.complete(prompt, max_tokens=50, temperature=0.1)
        
        results.append({
            "id": i,
            "input": sample.get('input'),
            "prediction": answer.strip(),
            "gold": sample.get('output')
        })
    
    return results

def main():
    """Run all tasks on small dev set."""
    all_results = {}
    
    for task_num in range(1, 9):
        results = run_task(task_num, num_shots=10, max_samples=5)
        all_results[task_num] = results
        
        # Print first result
        if results:
            print(f"  Sample 0: {results[0]['prediction']}")
    
    # Save results
    os.makedirs("runs", exist_ok=True)
    with open("runs/baseline_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    print("\n✓ Results saved to runs/baseline_results.json")

if __name__ == "__main__":
    main()
```

### 5.5 Create `solution/requirements.txt`

```
transformers==4.42.0
torch==2.2.0
vllm==0.8.0
openai==1.3.0
pydantic==2.5.0
python-dotenv==1.0.0
rank-bm25==0.2.2
scikit-learn==1.3.2
tqdm==4.66.1
```

---

## **STEP 6: Download OpenSeek Data**

Download the official datasets:

```python
import subprocess
import os

# Clone the repo (one-time)
if not os.path.exists("OpenSeek"):
    print("Downloading OpenSeek repository...")
    subprocess.run(
        ["git", "clone", "https://github.com/FlagAI-Open/OpenSeek.git"],
        capture_output=True
    )
    print("✓ Repository cloned")
else:
    print("✓ OpenSeek repo already present")

# Verify data exists
data_path = "OpenSeek/openseek/competition/LongContext-ICL-Annotation/data"
if os.path.exists(data_path):
    import os
    tasks = len([d for d in os.listdir(data_path) if d.startswith('task')])
    print(f"✓ Found {tasks} tasks in {data_path}")
```

---

## **STEP 7: Copy Solution to Kaggle**

### Option A: Upload as Dataset

1. Create folder locally: `solution/`
2. Zip it: `solution.zip`
3. Upload to Kaggle Datasets
4. In notebook:
   ```python
   !cp /kaggle/input/icl-solution/* .
   ```

### Option B: Direct Copy-Paste

In Kaggle notebook, create cells for each file above.

---

## **STEP 8: Run Baseline (First Test)**

```python
import os
os.chdir("solution")

# Test data loading
from data.load_tasks import TaskDataLoader
loader = TaskDataLoader(data_dir="../OpenSeek/openseek/competition/LongContext-ICL-Annotation/data")
demos, test_data = loader.load_task(1)
print(f"Task 1 loaded: {len(demos)} demos, {len(test_data)} tests")

# Run baseline on 5 samples
python run.py  # Or import and call directly
```

---

## **STEP 9: Check Results**

```python
import json

with open("solution/runs/baseline_results.json") as f:
    results = json.load(f)

# Show task 1 results
print("Task 1 results (first 3):")
for r in results["1"][:3]:
    print(f"  Pred: {r['prediction']}")
    print(f"  Gold: {r['gold']}")
    print()
```

---

## **Troubleshooting**

### vLLM Server Won't Start

```bash
# Check if port 8000 is in use
!lsof -i :8000

# Try different port
!vllm serve Qwen/Qwen3-4B --port 8001
```

### Out of Memory

```bash
# Reduce context or memory usage
!vllm serve Qwen/Qwen3-4B \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.8 \
  --max-model-len 32000  # Reduce from 40K
```

### Model Not Found

```bash
# Manually download first
from transformers import AutoModel
AutoModel.from_pretrained("Qwen/Qwen3-4B", cache_dir="/tmp/models")

# Then serve from cache
!vllm serve /tmp/models/Qwen3-4B ...
```

---

## **What to Do Next (In Order)**

- [ ] Create Kaggle account + start notebook
- [ ] Run vLLM server (test with simple completion)
- [ ] Clone OpenSeek repo to get task data
- [ ] Build `solution/` folder with `data/load.py` + `llm/client.py`
- [ ] Reproduce baseline on Task 1 (just 5 samples to verify setup works)
- [ ] Add your first optimization (e.g., structured output parsing)
- [ ] Run all 8 tasks on dev split
- [ ] Log results in `docs/04_EXPERIMENT_LOG.md`
- [ ] Iterate per Phase 1 of plan

---

## **Key Files to Keep**

After running:
- ✅ `runs/baseline_results.json` — your first results
- ✅ `runs/experiment_XYZ.json` — each experiment
- ✅ Kaggle notebook link — for reproducibility

---

**Ready to start? Let me know when:**
1. ✓ Kaggle account created
2. ✓ vLLM server starts without errors
3. ✓ Task 1 data loads successfully

Then I'll give you the optimized task-specific code!
