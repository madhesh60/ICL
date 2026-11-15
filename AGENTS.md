1. Competition Overview
In the era of large language models (LLMs), high-quality annotated data has become a critical driving force behind rapid advances in model capability. Whether for pre-training, fine-tuning, alignment, or domain adaptation, the upper bound of model performance is largely determined by the scale, diversity, and annotation quality of the training data. However, traditional human annotation is costly, time-consuming, and difficult to scale to complex tasks involving long-range semantics, which severely limits the practical deployment of LLMs in specialized domains and long-text scenarios.

In-context learning (ICL), as a few-shot learning paradigm that requires no parameter updates, has demonstrated remarkable flexibility and plug-and-play usability. By incorporating input–output examples into prompts, LLMs can infer task intent and generalize to unseen samples, achieving performance comparable to or even exceeding that of fine-tuned models in many NLP tasks, while significantly reducing reliance on large-scale annotated data.

Nevertheless, existing ICL research has predominantly focused on short or medium-length contexts. In ultra-long context settings that span tens of thousands to hundreds of thousands of tokens, the effectiveness, robustness, and scalability of ICL remain insufficiently explored. From one perspective, large collections of example texts encode rich task- and domain-specific information, which makes it difficult to efficiently select, order, and organize demonstrations and instructions without introducing interference, ambiguity, or attention dilution. From another perspective, ultra-long contexts often contain abundant structured and unstructured information, thereby imposing substantially higher requirements on a model’s memory capacity, reasoning ability, and key information extraction. As a result, simple single-step in-context inference is no longer adequate. Instead, effective solutions require the integration of multi-step reasoning and analytical strategies to support collaborative annotation frameworks that exhibit perception, planning, and execution capabilities.

This track focuses on complex, real-world ultra-long text scenarios and aims to explore the capabilities and practical approaches of large language models (LLMs) for automated data annotation. Participants are encouraged to develop long-context in-context learning (ICL) frameworks with effective example selection, context organization, and multi-step reasoning, in order to improve annotation accuracy, robustness, and scalability toward human-level performance. The competition seeks to bring together ideas from both academia and industry, promoting reproducible and open solutions that support the development of high-quality, cost-effective, and scalable intelligent data systems.

. Challenge Description: LLM Automatic Data Annotation in Long-Context Scenarios
1.1 Overview
The LLM Automatic Data Annotation Challenge for Long-Context Scenarios focuses on automated data annotation under ultra-long context settings using the Qwen3-4B large language model and the In-Context Learning (ICL) paradigm. Participating teams are required to use the unified datasets provided by the organizing committee, design effective ICL-based annotation strategies tailored for ultra-long contexts, and complete inference and evaluation on standardized evaluation sets.

Based on standardized evaluation results, the organizing committee will conduct a comprehensive assessment of all submitted solutions and determine the final rankings.

1.1.1 Objectives
Driven by large language models (LLMs), this challenge targets automated data annotation under ultra-long context conditions and aims to explore novel technical paradigms that jointly optimize efficiency and accuracy. The competition focuses on the following three key scientific and engineering challenges:

Instruction and Prompt Design:
How can effective model instructions and prompt strategies be designed in ultra-long context scenarios to guide LLMs toward stable and high-quality data annotation?

Ultra-Long Context Construction:
When the number of available annotation examples significantly exceeds the model’s context capacity, how can information-dense and structurally coherent ultra-long context inputs be constructed for target data annotation?

Multi-Turn and Continuous Annotation:
In automated multi-round dialogue or continuous interaction settings, how can ultra-long contexts be efficiently leveraged to achieve both consistency and scalability in data annotation?

1.2 Task Details
Teams are required to independently design a complete LLM-based automatic data annotation solution and conduct experimental validation under unified datasets and evaluation settings. Evaluation scores and rankings will be publicly released through a standardized leaderboard.

In addition to evaluation results, teams must submit a technical report and reproducible source code in accordance with the competition requirements. The organizing committee will perform reproducibility verification and technical review. Final scores will be determined by a weighted combination of prediction performance scores and technical solution scores, as specified below.

Participants are required to submit their technical report and complete source code to the official OpenSeek repository in the form of a Pull Request (PR), and to provide the corresponding PR link on the relevant competition page of the FlagOS platform.

1.2.1 Task Requirements
Model and Task Requirements

Teams must design an ICL-based solution for automatic data annotation using the Qwen3-4B model.

All inference and validation must be conducted on the unified datasets provided by the organizing committee.

Evaluation results will be summarized and ranked on a standardized leaderboard using a unified evaluation pipeline.

Model Links:

HuggingFace: https://huggingface.co/Qwen/Qwen3-4B

ModelScope: https://modelscope.cn/models/Qwen/Qwen3-4B

Modelers: https://modelers.cn/models/Qwen-AI/Qwen3-4B

Model Usage Restrictions

The use of any model other than Qwen3-4B is strictly prohibited.

Violations will result in disqualification, and corresponding results will be deemed invalid.

Data Usage Restrictions

Participants are only permitted to use the datasets officially provided by the competition organizers. The use of any external, third-party, or self-collected data is strictly prohibited.

Any violation of this requirement will result in disqualification, and the corresponding competition results will be deemed invalid.

Experimental Completeness Requirement

During the evaluation stage, teams must perform inference on all datasets and submit complete prediction results.

Teams that fail to submit results or submit only partial results will be considered as “not having completed the competition”, and their scores will not be included in evaluation or ranking.

Development Framework Requirement

All teams must use FlagScale as the development framework.

Only teams that pass the code compliance review conducted by the organizing committee will be eligible for final ranking.

Reproducibility and Validity

During the technical review stage, the organizing committee will reproduce submitted solutions.

Solutions that cannot be successfully reproduced will be deemed invalid.

Requirements for the Technical Report and Code Open-Source

To promote technical exchange and knowledge sharing, all teams are required to publicly release their technical report and complete source code under the official OpenSeek open-source project within 3 calendar days after the conclusion of the competition stage. Teams that fail to complete the open-source submission within the specified timeframe, or whose submissions are incomplete or inconsistent, will be deemed to have not fulfilled the required competition procedures.

Score Composition and Ranking Rules
Final scores consist of two components:

Prediction Performance Score: 60%

Technical Solution Score: 40%

Rankings and awards will be determined based on the final score.

1.2.2 Data and Benchmark Description
The organizing committee provides a unified standard dataset to evaluate the effectiveness of submitted solutions. A complete set of baseline code is also provided to demonstrate the end-to-end automatic data annotation workflow.

A total of 8 datasets are included in this challenge. Teams must run their proposed annotation solution on all eight datasets and submit the final annotation results.

Each development dataset contains the following components:

Task definition and description

ICL demonstration set

Test sample set

Minimum ICL context length constraint

All datasets and baseline code will be released via GitHub at the following repository:
https://github.com/FlagAI-Open/OpenSeek/tree/main/openseek/competition/LongContext-ICL-Annotation

The project's directory structure is as follows:

复制代码
LongContext-ICL-Annotation/
|
|---- data             dataset dir
|---- src              baseline dir
|---- README.md        English Readme
|---- READMD_zh_cn.md  Chinese Readme


1.2.4 Evaluation Protocol
Prediction Performance Score (60 %)
After prediction submission, the organizing committee will evaluate consistency between predicted labels and ground-truth annotations. Accuracy is used as the evaluation metric:

Average Accuracy: Mean annotation accuracy over test datasets 1–7:
S
A
c
c
u
r
a
c
y
=
N
r
i
g
h
t
N
a
l
l
S 
Accuracy
​
 = 
N 
all
​
 
N 
right
​
 
​
 

where, 
N
r
i
g
h
t
N 
right
​
  denotes the number of correctly annotated samples, and 
N
a
l
l
N 
all
​
  denotes the total number of test samples.

Consistency-Aware Metric：For Test dataset 8, we adopt a consistency-aware metric based on code generation to assess the logical consistency of model outputs. This metric focuses on the alignment between executability and computational correctness of the generated code. Specifically, the generated code is required not only to execute successfully but also to produce correct computational results. The score is defined as follows:
S
C
o
n
s
i
s
t
e
n
c
y
=
S
E
x
e
c
u
t
i
o
n
(
1
−
S
C
a
l
l
−
S
E
x
e
c
u
t
i
o
n
S
C
a
l
l
)
S 
Consistency
​
 =S 
Execution
​
 (1− 
S 
Call
​
 
S 
Call
​
 −S 
Execution
​
 
​
 )

where, 
S
C
a
l
l
S 
Call
​
  denotes the proportion of generated code that can be successfully invoked and executed at the functional level, and 
S
E
x
e
c
u
t
i
o
n
S 
Execution
​
  denotes the proportion of generated code that correctly completes the computation and produces the expected results.

Technical Solution Score (40 %)
The organizing committee will reproduce submitted code and evaluate the technical solution in conjunction with the technical report, focusing on the depth and effectiveness of solutions to the following core challenges:

Prompt and instruction design for stable, high-quality annotation in ultra-long contexts (20%)

How to construct information-dense and structurally coherent ultra-long contexts when the number of available annotated examples significantly exceeds the model’s context capacity (40%)

Efficient utilization of ultra-long contexts in multi-turn or continuous annotation settings, balancing consistency and scalability (40%)

Notes:

Solutions that cannot be successfully reproduced will be disqualified from final ranking.

During the reproducibility phase, the organizing committee will deploy the Qwen3-4B in a unified hardware environment and conduct reproduction experiments on the solutions submitted by the participating teams.