# Awesome-LLM-Disease-Diagnosis


# Introduction 
This repository contains resources for the review paper "[Large language models for disease diagnosis: A scoping review](https://www.nature.com/articles/s44387-025-00011-z)" (NPJ Artificial Intelligence 2025).



---

### Figures

---

#### **Figure 1: Overview of the Investigated Scope**

<img src="Fig_1_scope.jpg" alt="Figure: Overview of the investigated scope" width="700px" />

*Overview of the investigated scope. It illustrated disease types and the associated clinical specialties, clinical
data types, modalities of the utilized data, the applied LLM techniques, and evaluation methods. We only presented
part of the clinical specialties, some representative diseases, and partial LLM techniques.*

---

#### **Figure 2: Metadata of Information**

<img src="Fig_2_meta.png" alt="Figure: Future meta" width="700px" />

*Metadata of information extracted from LLM-based diagnostic studies in the scoping review.*

---

#### **Figure 3: Evaluation Approaches**

<img src="Fig_3_eval.jpg" alt="Figure: Evaluation" width="700px" />

*A summary of the evaluation approaches used for diagnostic tasks.*

---

#### **Figure 4: Limitations and Future Directions**

<img src="Fig_future_direction.jpg" alt="Figure: Future direction" width="700px" />

*A summary of the limitations and future directions for LLM-based disease diagnosis.*

---









# RAG


## Disease diagnosis


| Title                                                                                                                 | Published Year | Journal                                 | Task                     | Input Data Modality               | LLM Technique Type                            |
|-----------------------------------------------------------------------------------------------------------------------|----------------|-----------------------------------------|--------------------------|------------------------------------|-----------------------------------------------|
| CUPCase: Clinically Uncommon Patient Cases and Diagnoses Dataset                                                     | 2025           | AAAI Conference on Artificial Intelligence | Disease diagnosis        | Text                               | RAG (corpus)                                  |
| Optimization of hepatological clinical guidelines interpretation by large language models: a retrieval augmented generation-based framework | 2024           | npj Digital Medicine                   | Disease diagnosis        | Text                               | RAG (corpus)                                  |
| Explanatory argumentation in natural language for correct and incorrect medical diagnoses                             | 2024           | Journal of Biomedical Semantics        | Disease diagnosis        | Text, Image                        | RAG (database)                                |
| ECG Semantic Integrator (ESI): A Foundation ECG Model Pretrained with LLM-Enhanced Cardiological Text                 | 2024           | arXiv                                  | Disease diagnosis        | Time series, Text                  | RAG (corpus)                                  |
| Beyond Direct Diagnosis: LLM-based Multi-Specialist Agent Consultation for Automatic Diagnosis                        | 2024           | arXiv                                  | Disease diagnosis        | Text                               | RAG (database)                                |
| Diagnosis Assistant for Liver Cancer Utilizing a Large Language Model with Three Types of Knowledge                   | 2024           | arXiv                                  | Disease diagnosis        | Image, Text                        | RAG (database), Prompt (CoT)                 |
| Guiding clinical reasoning with large language models via knowledge seeds                                             | 2024           | arXiv                                  | Disease diagnosis        | Text                               | RAG (knowledge graph)                         |
| Large Language Model-informed ECG Dual Attention Network for Heart Failure Risk Prediction                            | 2024           | arXiv                                  | Disease diagnosis        | Time series, Text                  | RAG (database)                                |
| Dermacen Analytica: A Novel Methodology Integrating Multi-Modal Large Language Models with Machine Learning in tele-dermatology | 2024           | arXiv                                  | Disease diagnosis        | Image, Text                        | RAG (database)                                |
| Autonomous artificial intelligence agents for clinical decision making in oncology                                   | 2024           | arXiv                                  | Disease diagnosis        | Text                               | RAG (database)                                |
| medIKAL: Integrating Knowledge Graphs as Assistants of LLMs for Enhanced Clinical Diagnosis on EMRs                   | 2024           | arXiv                                  | Disease diagnosis        | Graph, Text                        | RAG (knowledge graph)                         |
| A 360º View for Large Language Models: Early Detection of Amblyopia in Children using Multi-View Eye Movement Recordings. | 2024           | Artificial Intelligence in Medicine    | Disease diagnosis        | Image                              | RAG (corpus), Prompt (few-shot)              |
| Empowering PET Imaging Reporting with Retrieval-Augmented Large Language Models and Reading Reports Database: A Pilot Single Center Study | 2024           | medRxiv                                | Disease diagnosis        | Text                               | RAG (database)                                |
| EMERGE: Integrating RAG for Improved Multimodal EHR Predictive Modeling                                               | 2024           | arXiv                                  | Disease diagnosis        | Text, Time series, Graph           | RAG (corpus), RAG (knowledge graph)          |
| Large Language Models and Medical Knowledge Grounding for Diagnosis Prediction                                        | 2023           | medRxiv                                | Differential diagnosis    | Text, Graph                        | RAG (knowledge graph)                         |
| Generative Large Language Models are autonomous practitioners of evidence-based medicine                              | 2024           | arXiv                                  | Disease diagnosis        | Text                               | RAG (corpus)                                  |
| Enhancing Large Language Models for Clinical Decision Support by Incorporating Clinical Practice Guidelines           | 2024           | arXiv                                  | Disease diagnosis        | Text                               | RAG (corpus)                                  |




## Conversational diagnosis

| Title                                                                                                               | Published Year | Journal             | Task                 | Input Data Modality | LLM Technique Type |
|---------------------------------------------------------------------------------------------------------------------|----------------|---------------------|----------------------|---------------------|--------------------|
| Heterogeneous Knowledge Grounding for Medical Question Answering with Retrieval Augmented Large Language Model     | 2024           | WWW                 | Text-based Med QA    | Text                | RAG (database)     |
| Development of a Liver Disease-Specific Large Language Model Chat Interface using Retrieval Augmented Generation    | 2024           | Hepatology          | Text-based Med QA    | Text                | RAG (corpus)       |
| GPT-agents based on medical guidelines can improve the responsiveness and explainability of outcomes for traumatic brain injury rehabilitation | 2024           | Scientific Reports  | Text-based Med QA    | Text                | RAG (corpus)       |
| Mindmap: Knowledge graph prompting sparks graph of thoughts in large language models                                | 2023           | arXiv               | Text-based Med QA    | Text                | RAG (knowledge graph) |
| Improving accuracy of GPT-3/4 results on biomedical data using a retrieval-augmented language model                 | 2023           | arXiv               | Text-based QA        | Text                | RAG (database)     |
| Development of a Liver Disease-Specific Large Language Model Chat Interface using Retrieval Augmented Generation    | 2023           | Hepatology          | Text-based QA        | Text                | RAG (corpus)       |
| A Context-based Chatbot Surpasses Trained Radiologists and Generic ChatGPT in Following the ACR Appropriateness Guidelines | 2023           | Radiology           | Text-based Med QA    | Text                | RAG (database)     |








# Fine-tuning




## Disease diagnosis



| Title                                                                                                       | Published Year | Journal                                   | Task                            | Input Data Modality            | LLM Technique Type                       |
|-------------------------------------------------------------------------------------------------------------|----------------|-------------------------------------------|---------------------------------|---------------------------------|------------------------------------------|
| Large language model for providing patient-focused guidance following radical prostatectomy                | 2024           | Urologic Oncology                         | Disease diagnosis              | Text                            | Fine-tune (supervised FT)               |
| PneumoLLM: Harnessing the power of large language model for pneumoconiosis diagnosis                       | 2024           | Medical Image Analysis                    | Disease diagnosis              | Image                           | Fine-tune (supervised FT)               |
| Knowledge-enhanced visual-language pre-training on chest radiology images                                 | 2023           | Nature Communications                     | Disease diagnosis              | Image, Text                    | Fine-tune (supervised FT)               |
| From Classification to Clinical Insights: Towards Analyzing and Reasoning About Mobile and Behavioral Health Data With Large Language Models | 2023           | arXiv                                     | Multi-modal disease diagnosis  | Time series                    | Fine-tune (supervised FT), Prompt (CoT) |
| Enhancing automatic placenta analysis through distributional feature recomposition in vision-language contrastive learning | 2023           | MICCAI                                   | Multi-modal disease diagnosis  | Image, Text                    | Fine-tune (supervised FT)               |
| Evaluating the efficacy of supervised learning vs large language models for identifying cognitive distortions and suicidal risks in chinese social media | 2023           | arXiv                                     | Disease diagnosis              | Text                            | Fine-tune (supervised FT)               |
| Distilling the Knowledge from Large-language Model for Health Event Prediction                            | 2024           | medRxiv                                   | Disease diagnosis              | Text                            | Fine-tune (supervised FT)               |
| Development and Testing of a Novel Large Language Model-Based Clinical Decision Support Systems for Medication Safety in 12 Clinical Specialties | 2024           | arXiv                                     | Disease diagnosis              | Text                            | Fine-tune (supervised FT)               |
| Evaluating machine learning approaches for multi-label classification of unstructured electronic health records with a generative large language model | 2024           | medRxiv                                   | Disease diagnosis              | Text                            | Fine-tune (supervised FT)               |
| MEDFuse: Multimodal EHR Data Fusion with Masked Lab-Test Modeling and Large Language Models                | 2024           | arXiv                                     | Disease diagnosis              | Text, Tabular                  | Fine-tune (supervised FT)               |
| Towards conversational diagnostic AI                                                                      | 2024           | NEJM-AI                                   | Disease diagnosis              | Text                            | Fine-tune (supervised FT)               |
| Navigating Complexity: Enhancing Pediatric Diagnostics With Large Language Models                         | 2024           | Pediatric Critical Care Medicine          | Disease diagnosis              | Text                            | Fine-tune (supervised FT)               |
| Reasoning Like a Doctor: Improving Medical Dialogue Systems via Diagnostic Reasoning Process Alignment     | 2024           | ACL 2024 Findings                         | Disease diagnosis              | Text                            | Fine-tune (supervised FT)               |
| Integrating Physician Diagnostic Logic into Large Language Models: Preference Learning from Process Feedback | 2024           | ACL 2024 Findings                         | Disease diagnosis              | Text                            | Fine-tune (supervised FT), Fine-tune (RLHF) |
| Ophglm: Training an ophthalmology large language-and-vision assistant based on instructions and dialogue  | 2023           | arXiv                                     | Disease diagnosis              | Text, Image                    | Fine-tune (supervised FT)               |
| ClinicalGPT: large language models finetuned with diverse medical data and comprehensive evaluation        | 2023           | arXiv                                     | Disease diagnosis              | Text                            | Fine-tune (supervised FT), Fine-tune (RLHF) |
| EyeGPT: Ophthalmic Assistant with Large Language Models                                                   | 2024           | arXiv                                     | Domain-specific general model  | Text                            | Fine-tune (supervised FT)               |
| LLMs for Doctors: Leveraging Medical LLMs to Assist Doctors, Not Replace Them                             | 2024           | arXiv                                     | Disease diagnosis              | Text                            | Fine-tune (supervised FT)               |
| Advancing multimodal medical capabilities of Gemini                                                      | 2024           | arXiv                                     | Domain-specific general model  | Image, Omics, Text             | Fine-tune (supervised FT)               |
| Qilin-med: Multi-stage knowledge injection advanced medical large language model                          | 2023           | arXiv                                     | Domain-specific general model  | Text                            | Fine-tune (RLHF)                        |
| Moelora: An moe-based parameter efficient fine-tuning method for multi-task medical applications          | 2023           | arXiv                                     | Domain-specific general model  | Text                            | Fine-tune (supervised FT)               |
| MiniGPT-Med: Large Language Model as a General Interface for Radiology Diagnosis                         | 2024           | arXiv                                     | Disease diagnosis              | Image, Text                    | Fine-tune (supervised FT)               |
| MedDr: Diagnosis-Guided Bootstrapping for Large-Scale Medical Vision-Language Learning                    | 2024           | arXiv                                     | Disease diagnosis              | Image, Text                    | Fine-tune (supervised FT)               |
| Pre-trained multimodal large language model enhances dermatological diagnosis using SkinGPT-4            | 2024           | Nature Communications                     | Multi-modal disease diagnosis  | Image, Text                    | Fine-tune (supervised FT)               |
| Enhancing Human-Computer Interaction in Chest X-ray Analysis using Vision and Language Model with Eye Gaze Patterns | 2024           | arXiv                                     | Differential Diagnosis         | Image, Text                    | Fine-tune (supervised FT)               |
| A Multimodal Generative AI Copilot for Human Pathology                                                   | 2024           | Nature                                    | Disease diagnosis              | Text, Image                    | Fine-tune (supervised FT)               |
| SkinGPT-4: an interactive dermatology diagnostic system with visual large language model                 | 2023           | arXiv                                     | Disease diagnosis              | Image, Text                    | Fine-tune (supervised FT)               |
| Cxr-llava: Multimodal large language model for interpreting chest x-ray images                           | 2023           | arXiv                                     | Disease diagnosis              | Image, Text                    | Fine-tune (supervised FT)               |
| Skin disease diagnosis using deep neural network and large language model                               | 2023           | International Conference on AI in Medicine | Disease diagnosis             | Image, Text                    | Fine-tune (supervised FT)               |
| DictLLM: Harnessing Key-Value Data Structures with Large Language Models for Enhanced Medical Diagnostics | 2024           | arXiv                                     | Disease diagnosis              | Text, Tabular                  | Fine-tune (supervised FT)               |
| EyeFound: A Multimodal Generalist Foundation Model for Ophthalmic Imaging                               | 2024           | arXiv                                     | Disease diagnosis              | Text, Image                    | Prompt (zero-shot), Fine-tune (supervised FT), Fine-tune (parameter efficient FT) |
| Zero-shot ECG diagnosis with large language models and retrieval-augmented generation                   | 2023           | Machine Learning for Health (ML4H)        | Disease diagnosis             | Time series                    | RAG (database), Prompt (CoT), Fine-tune (supervised FT) |
| Human-AI collectives produce the most accurate differential diagnoses                                   | 2024           | arXiv                                     | Disease diagnosis              | Text                            | Fine-tune (supervised FT)               |
| When LLMs meets acoustic landmarks: An efficient approach to integrate speech into large language models for depression detection | 2024           | arXiv                                     | Disease diagnosis              | Audio, Text                    | Fine-tune (supervised FT)               |
| Learning the natural history of human disease with generative transformers                              | 2024           | medRxiv                                   | Disease diagnosis              | Text                            | Fine-tune (supervised FT)               |
| AI-Generated Content Enhanced Computer-Aided Diagnosis Model for Thyroid Nodules: A ChatGPT-Style Assistant | 2024           | arXiv                                     | Disease diagnosis              | Text, Image                    | Fine-tune (supervised FT)               |
| Merlin: A Vision Language Foundation Model for 3D Computed Tomography                                   | 2024           | Res Sq                                    | Disease diagnosis              | Tabular, Image, Text           | Fine-tune (supervised FT)               |
| RULE: Reliable Multimodal RAG for Factuality in Medical Vision Language Models                          | 2024           | arXiv                                     | Disease diagnosis              | Text, Image                    | RAG (corpus), Fine-tune (RLHF)          |
| Supervised Learning and Large Language Model Benchmarks on Mental Health Datasets: Cognitive Distortions and Suicidal Risks in Chinese Social Media | 2024           | arXiv                                     | Disease diagnosis              | Text                            | Prompt (zero-shot), Prompt (few-shot), Fine-tune (supervised FT) |
| ChatASD: LLM-Based AI Therapist for ASD                                                                | 2024           | Digital Multimedia Communications         | Medical QA, Disease Diagnosis | Audio, Text                    | Fine-tune (supervised FT)               |
| Large Language Models Are Clinical Reasoners: Reasoning-Aware Diagnosis Framework with Prompt-Generated Rationales | 2024           | AAAI                                      | Disease diagnosis              | Text, Image                    | Fine-tune (supervised FT)               |
| DRG-LLaMA: Tuning LLaMA Model to Predict Diagnosis-related Group for Hospitalized Patients             | 2024           | NPJ Digital Medicine                      | Differential diagnosis         | Text                            | Fine-tune (parameter efficient FT)     |
| CancerLLM: A Large Language Model in Cancer Domain                                                    | 2024           | arXiv                                     | Disease diagnosis              | Text                            | Fine-tune (parameter efficient FT)     |
| CPLLM: Clinical Prediction with Large Language Models                                                  | 2024           | PLOS Digital Health                      | Disease diagnosis              | Text                            | Fine-tune (supervised FT)               |




## Conversational diagnosis

| Title                                                                                                   | Published Year | Journal                                                     | Task                | Input Data Modality | LLM Technique Type                                   |
|---------------------------------------------------------------------------------------------------------|----------------|-------------------------------------------------------------|---------------------|---------------------|----------------------------------------------------|
| Clinical camel: An open expert-level medical language model with dialogue-based knowledge encoding      | 2023           | arXiv                                                       | Text-based Med QA   | Text                | Fine-tune (supervised FT)                          |
| Qibo: A Large Language Model for Traditional Chinese Medicine                                           | 2024           | arXiv                                                       | Text-based Med QA   | Text                | Fine-tune (supervised FT)                          |
| PediatricsGPT: Large Language Models as Chinese Medical Assistants for Pediatric Applications           | 2024           | arXiv                                                       | Text-based Med QA   | Text                | Fine-tune (supervised FT), Fine-tune (RLHF)        |
| WundtGPT: Shaping Large Language Models To Be An Empathetic, Proactive Psychologist                     | 2024           | arXiv                                                       | Text-based Med QA   | Text                | Fine-tune (supervised FT), Fine-tune (RLHF)        |
| Huatuogpt, towards taming language model to be a doctor                                                 | 2023           | Findings of the Association for Computational Linguistics: EMNLP 2023 | Text-based Med QA   | Text                | Fine-tune (supervised FT), Fine-tune (RLHF)        |
| MedKP: Medical Dialogue with Knowledge Enhancement and Clinical Pathway Encoding                        | 2024           | arXiv                                                       | Text-based Med QA   | Text                | Fine-tune (supervised FT)                          |
| BP4ER: Bootstrap Prompting for Explicit Reasoning in Medical Dialogue Generation                        | 2024           | arXiv                                                       | Text-based Med QA   | Text                | Fine-tune (supervised FT)                          |
| A Generalist Learner for Multifaceted Medical Image Interpretation                                      | 2024           | arXiv                                                       | Multi-modal Med QA  | Image, Text         | Fine-tune (supervised FT)                          |
| PathGen-1.6 M: 1.6 Million Pathology Image-text Pairs Generation through Multi-agent Collaboration      | 2024           | arXiv                                                       | VQA                 | Image, Text         | Fine-tune (supervised FT)                          |
| Assessing and Optimizing Large Language Models on Spondyloarthritis Multi-Choice Question Answering     | 2024           | JMIR Res Protoc                                             | Med QA              | Text                | Fine-tune (supervised FT)                          |
| Xraygpt: Chest radiographs summarization using medical vision-language models                           | 2023           | arXiv                                                       | Med VQA             | Image, Text         | Fine-tune (supervised FT)                          |
| Benchmarking large language models on cmexam-a comprehensive chinese medical exam dataset               | 2023           | NIPS                                                        | Med QA              | Text                | Fine-tune (supervised FT)                          |
| Radonc-gpt: A large language model for radiation oncology                                               | 2023           | arXiv                                                       | Med QA              | Text                | Fine-tune (supervised FT)                          |
| MedAide: Leveraging Large Language Models for On-Premise Medical Assistance on Edge Devices             | 2024           | arXiv                                                       | Text-based Med QA   | Text                | Fine-tune (supervised FT), Fine-tune (RLHF), Fine-tune (parameter efficient FT) |
| CoD, Towards an Interpretable Medical Agent using Chain of Diagnosis                                    | 2024           | arXiv                                                       | Text-based Med QA   | Text                | Fine-tune (supervised FT)                          |



## Mental health disorder detection

| Title                                                                                                   | Published Year | Journal                                       | Task                        | Input Data Modality | LLM Technique Type                               |
|---------------------------------------------------------------------------------------------------------|----------------|-----------------------------------------------|-----------------------------|---------------------|--------------------------------------------------|
| MentaLLaMA: Interpretable Mental Health Analysis on Social Media with Large Language Models            | 2024           | WWW                                           | Depression Detection        | Text                | Prompt (few-shot), Fine-tune (supervised FT)    |
| Spontaneous Speech-Based Suicide Risk Detection Using Whisper and Large Language Models                | 2024           | Interspeech                                   | Suicide Risk Detection       | Audio, Text         | Fine-tune (supervised FT)                       |
| Mental-LLM: Leveraging Large Language Models for Mental Health Prediction via Online Text Data         | 2024           | arXiv                                         | Mental Health Prediction    | Text                | Fine-tune (supervised FT), Prompt (zero-shot), Prompt (few-shot) |
| Explainable Depression Symptom Detection in Social Media                                               | 2023           | Health Information Science and Systems       | Depression Detection        | Text                | Fine-tune (supervised FT), Prompt (few-shot)    |




# Pre-training


## Disease diagnosis

| Title                                                                                                         | Published Year | Journal                                   | Task                               | Input Data Modality                     | LLM Technique Type                     |
|---------------------------------------------------------------------------------------------------------------|----------------|-------------------------------------------|------------------------------------|------------------------------------------|----------------------------------------|
| Human-Algorithmic Interaction Using a Large Language Model-Augmented Artificial Intelligence Clinical Decision Support System | 2024           | ACM CHI                                  | Disease diagnosis                 | Text                                     | Pre-training                           |
| A medical multimodal large language model for future pandemics                                               | 2023           | npj Digital Medicine                     | Disease diagnosis                 | Image, Text                              | Pre-training, Fine-tune (supervised FT) |
| A large language model for electronic health records                                                         | 2022           | npj Digital Medicine                     | Clinical natural language processing | Text                                     | Pre-training, Fine-tune (supervised FT) |
| Towards accurate differential diagnosis with large language models                                           | 2023           | arXiv                                    | Differential diagnosis             | Text                                     | Fine-tune (supervised FT), Pre-training |
| Stone needle: A general multimodal large-scale model framework towards healthcare                            | 2023           | arXiv                                    | Multimodal healthcare integration | Text, Image, Video, Audio                | Pre-training, Fine-tune (supervised FT) |
| Towards generalist foundation model for radiology                                                            | 2023           | arXiv                                    | Disease diagnosis                 | Image, Text                              | Pre-training                           |
| A generalist vision–language foundation model for diverse biomedical tasks                                   | 2023           | Nature Medicine                          | Multi-modal disease diagnosis      | Image, Text                              | Pre-training, Fine-tune (supervised FT) |
| Clinicalmamba: A generative clinical language model on longitudinal clinical notes                           | 2024           | arXiv                                    | Disease diagnosis                 | Text                                     | Pre-training                           |
| Decomposing Disease Descriptions for Enhanced Pathology Detection: A Multi-Aspect Vision-Language Pre-training Framework | 2024           | Proceedings of the …                     | Disease diagnosis                 | Image, Text                              | Pre-training                           |
| Mica: Towards explainable skin lesion diagnosis via multi-level image-concept alignment                      | 2024           | Proceedings of the AAAI Conference on Artificial … | Disease diagnosis                 | Image, Text                              | Fine-tune (supervised FT), Pre-training |
| A visual–language foundation model for pathology image analysis using medical Twitter                        | 2023           | Nature Medicine                          | Multi-modal disease diagnosis      | Image, Text                              | Pre-training, Fine-tune (supervised FT) |
| A visual-language foundation model for computational pathology                                               | 2024           | Nature Medicine                          | Disease diagnosis                 | Text, Image                              | Pre-training, Fine-tune (supervised FT) |



## Conversational diagnosis

| Title                                                                                      | Published Year | Journal | Task             | Input Data Modality | LLM Technique Type                   |
|--------------------------------------------------------------------------------------------|----------------|---------|------------------|---------------------|--------------------------------------|
| Biomistral: A collection of open-source pretrained large language models for medical domains | 2024           | arXiv   | Text-based Med QA | Text                | Pre-training, Fine-tune (supervised FT) |




## Risk prediction

| Title                                                                                                      | Published Year | Journal | Task                                      | Input Data Modality       | LLM Technique Type                     |
|------------------------------------------------------------------------------------------------------------|----------------|---------|------------------------------------------|----------------------------|----------------------------------------|
| Large Language Multimodal Models for New-Onset Type 2 Diabetes Prediction using Five-Year Cohort Electronic Health Records | 2024           | arXiv   | Chronic Disease Prediction              | Text, Tabular              | Pre-training                           |
| From Supervised to Generative: A Novel Paradigm for Tabular Deep Learning with Large Language Models      | 2024           | KDD     | Disease Classification, Risk Prediction | Tabular                   | Fine-tune (supervised FT), Pre-training |
| Large Language Multimodal Models for 5-Year Chronic Disease Cohort Prediction Using EHR Data              | 2024           | arXiv   | Risk Prediction                         | Text, Tabular              | Pre-training                           |






# Citation

Please kindly cite the paper if it benefits your research:

Zhou, Shuang, et al. "Large language models for disease diagnosis: A scoping review." npj Artificial Intelligence 1.1 (2025): 1-17.

or

```bib
@article{zhou2025LLM_Dx,
  title={Large language models for disease diagnosis: A scoping review},
  author={Zhou, Shuang and Xu, Zidu and Zhang, Mian and Xu, Chunpu and Guo, Yawen and Zhan, Zaifu and Fang, Yi and Ding, Sirui and Wang, Jiashuo and Xu, Kaishuai and others},
  journal={npj Artificial Intelligence},
  volume={1},
  number={1},
  pages={1--17},
  year={2025},
  publisher={Nature Publishing Group}
}
```

