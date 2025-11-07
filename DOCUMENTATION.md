# DOCUMENTATION 

This document aims to explain the high level workflow of our project by a diagram approach then breaks down each supporting module/script purpose

## Simple Agentic Architecture Diagram
```mermaid
flowchart TD
   A[model_setting]<--> B[<b>User Input</b>] 
   B --> C[<b>baseline_sentiment_json</b><br/><small><i>raw llm call</i></small>] 
   B --> D[<b>metamind_sentiment_json</b><br/><small><i>agentic llm call</i></small>]
   E[synthesize_with_llm<br/><small><i>Agent 1</i></small>] --> D 
   F[extract_aspects_with_llm<br/><small><i>Agent 2</i></small>] --> D
   G[generate_recommendation_with_llm<br/><small><i>Agent 3</i></small>] --> D
   H[ToMAgent + DomainAgent] <--> E
   C --> I[<b>UI Ouput</b>] 
   D --> I
```
## Scripts/Local Dependencies description 

### UI Layer
- `interface_V2.py` (runnable)

Streamlit user interface that load data, dependencies, triggers analysis workflow and display the results.

### Configuration
- `config.py` (module)

Set API credentials, model settings(like temperature, max_tokens) and agents parameters

### LLM Layer 
- `base_llm.py` (module)
- `openai_llm.py` (module)

Initializes the language model interface used by all agents in the pipeline.<br>
The agents communicate with the model using the OpenAI-style API payloads and parameters.

### Agent Layer 

- `base_agent.py` (module)
- `tom_agent.py` (module)
- `domain_agent.py` (module)

ToM and Domain refinements agent from the original MetaMind framework

### Memory Layer

- `social_memory.py` (module)

Provides user-context summaries consumed by agents during MetaMind processing.

### Prompt Library 

- `prompt_templates.py` (module)

Prompt templates injected into LLM calls across different stages of the pipeline

### Baseline Sentiment 

- `raw_sentiment.py`(module)

One-shot sentiment analysis call to the LLM that ouput a JSON - invoked directly from the UI for comparison purpose-
### MetaMind Analysis

- `sentiment.py` (module)
- `recommandation_text.py` (module)

Orchestrates the agentic pipeline with the 3 agent (+TOM and Domain) making hypothesis enrichment, synthesis, aspects extraction and recommandations. It also ensure the JSONS format of the ouput and human readable recommandations from the resulted JSON. 

### Utilites 

- `helpers.py` (helper)

A JSON parsing helper used everywhere in the pipeline 



