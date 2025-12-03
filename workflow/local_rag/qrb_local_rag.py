#!/usr/bin/env python
# coding: utf-8

# YOUR MODEL INFERENCE / EMBEDDINGS

# Reference : https://github.com/The-Pocket/PocketFlow/blob/main/cookbook/pocketflow-agent/nodes.py
from pocketflow import Node, Flow
import numpy as np
from operator import itemgetter
import yaml
import sys

# RAG Node and functions
import os
import requests
CHUNK_SIZE = 4096
OVERLAP = 512

# use your embedding models as you wish
model = "qwen3-embedding:0.6b"

def get_embedding_ollama(text: str, timeout: int = 10) -> np.ndarray:
    payload = {"model": model, "prompt": text}
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return np.array(data.get("embedding", []), dtype=float)

def download_file(url:str):
    response = requests.get(url)
    response.raise_for_status()

    with open("test.txt", "wb") as f:
        f.write(response.content)

    return True

def chunk_text_file(file_path: str):
    """
    load ./test.txt and cut into chunks
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"file not existed: {file_path}")

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    # change \r into \n
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    
    # keep some overlap of chunks
    step = CHUNK_SIZE - OVERLAP
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + CHUNK_SIZE
        chunk = text[start:end]
        if not chunk:
            break
        chunks.append(chunk)
        start += step
        
    return chunks

class LocalRAG(Node):
    def prep(self, shared):
        """prepare local RAG results : encode query text and compute similarities"""
        query = shared["question"]
        
        # solute to https://github.com/The-Pocket/PocketFlow-Tutorial-Video-Generator and its author
        url = "https://raw.githubusercontent.com/The-Pocket/PocketFlow-Tutorial-Video-Generator/refs/heads/main/docs/llm/transformer.md"
        if download_file(url):
            chunk_list = chunk_text_file("test.txt")
        else:
            print("test txt file download failed.")

        print(f"chunked txt num : {len(chunk_list)} ")

        # get embedding of each chunk
        vector_list = []
        for chunk in chunk_list:
            v = get_embedding_ollama(chunk)
            vector_list.append(v)

        # build vector store, each item assembled with chunk,vector,similarity
        # NOTE : this assembling process contains duplicated computation
        query_embeddings = get_embedding_ollama(query)
        vector_store = [(chunk, None, None) for chunk in chunk_list]
        for i, (chunk, _, _) in enumerate(vector_store):
            score = np.dot(query_embeddings, vector_list[i])
            vector_store[i] = (chunk, vector_list[i], score)

        # rank top5 with score
        top5 = sorted(vector_store, key=itemgetter(2), reverse=True)[:5]
        top5_context = "\n".join(
            f"\n\n``````CHUNK PIECE #{i+1}:\n{row[0]}\n``````END\n\n"
            for i, row in enumerate(top5)
        )
        shared["top5_context"] = top5_context

        return shared["top5_context"], shared["question"]
        
    def exec(self, inputs):
        """Assemble RAG Node Prompt"""
        top5_context, search_query = inputs
 
        prompt = f"""
### CONTEXT
You are a data assistant.
You have 5 data pieces sorted with dot product similarities. 
You need to analyze the confidence interval with given data pieces and user queries. 

### USER QUERY
{search_query}

### DATA PIECES
{top5_context}

## NEXT ACTION
Decide the next action based on the context and available actions.
Return your response in this format:

```yaml
thinking: |
    <your step-by-step reasoning process>
confidence_interval: <confidence interval based on your reasoning>
```

IMPORTANT: Make sure to:
1.  Use proper indentation (4 spaces) for all multi-line fields
2.  Use the | character for multi-line text fields
3.  Keep single-line fields without the | character

### END
"""
        # Call the LLM to make a decision
        response = call_llm(prompt)
        
        # Response should be yaml string already
        yaml_str = response.split("```yaml")[1].split("```")[0].strip()
        data_assistant_dict = yaml.safe_load(yaml_str)

        return data_assistant_dict["confidence_interval"]
    
    def post(self, shared, prep_res, exec_res):
        """Save the RAG results and go back to the decision node."""
        # Add the local search file to the context in the shared store
        confidence_interval = exec_res
        
        shared["confidence_interval"] = exec_res

        shared["context"] = "\n\nData Assistant found 5 pieces of local database: " + shared["top5_context"] + f"\n``````\nData Assistant Confidence Interval: \n{shared['confidence_interval']}\n``````" + "\n\n"
        
        print(f"📚 RAG Node job done")
        # print(shared["context"])
        
        # Always go back to the decision node after searching
        return "decide"

# Decide Node
class DecideAction(Node):
    def prep(self, shared):
        """Prepare the context and question for the decision-making process."""
        # Get the current context (default to "No local RAG results" if none exists)
        context = shared.get("context", "No local RAG results")
        # Get the question from the shared store
        question = shared["question"]
        # Return both for the exec step
        return question, context
        
    def exec(self, inputs):
        """Call the LLM to decide whether to search or answer."""
        question, context = inputs
        
        #fetch timestamp
        ts = __import__('datetime').datetime.now().isoformat()
        
        print(f"🤔 Agent deciding what to do next...")
        
        # Create a prompt to help the LLM decide what to do next with proper yaml formatting
        # be careful to construct your prompt to achieve best efficiency
        prompt = f"""
### CONTEXT
You are a research assistant that can search the web.
Current Time:{ts}
Question: {question}
Local RAG Results: {context}

### ACTION SPACE
[1] data_assistant
  Description: Look up more information with local database
  Parameters:
    - query (str): What to query with local data assistant

[2] answer
  Description: Answer the question with current knowledge
  Parameters:
    - answer (str): Final answer to the question

## NEXT ACTION
Decide the next action based on the context and available actions.
Return your response in this format:

```yaml
thinking: |
    <your step-by-step reasoning process>
action: data_assistant OR answer
reason: <why you chose this action>
answer: <if action is answer>
query: <specific query if action is data_assistant>
```
IMPORTANT: Make sure to:
1. Use proper indentation (4 spaces) for all multi-line fields
2. Use the | character for multi-line text fields
3. Keep single-line fields without the | character

### END

"""
        # Call the LLM to make a decision
        response = call_llm(prompt)
        
        # Parse the response to get the decision
        yaml_str = response.split("```yaml")[1].split("```")[0].strip()
        decision = yaml.safe_load(yaml_str)
        
        return decision
    
    def post(self, shared, prep_res, exec_res):
        """Save the decision and determine the next step in the flow."""
        # If LLM decided to data_assistant, save the data_assistant query
        if exec_res["action"] == "data_assistant":
            shared["query"] = exec_res["query"]
            print(f"🔍 Agent decided to query for: {exec_res['query']}")
        else:
            shared["context"] = exec_res["answer"]
            print(f"💡 Agent decided to answer the question")
        
        # Return the action to determine the next node in the flow
        return exec_res["action"]

# Answer Node

class AnswerQuestion(Node):
    def prep(self, shared):
        """Get the question and context for answering."""
        return shared["question"], shared.get("context", "")
        
    def exec(self, inputs):
        """Call the LLM to generate a final answer."""
        question, context = inputs
        
        print(f"✍️ Crafting final answer...")
        
        # Create a prompt for the LLM to answer the question
        prompt = f"""
### CONTEXT
Based on the following information, answer the question.
Question: {question}
Research: {context}

## YOUR ANSWER:
Provide a comprehensive answer using the research results.
"""
        # Call the LLM to generate an answer
        answer = call_llm(prompt)
        return answer
    
    def post(self, shared, prep_res, exec_res):
        """Save the final answer and complete the flow."""
        # Save the answer in the shared store
        shared["answer"] = exec_res
        
        print(f"✅ Answer generated successfully")
        
        # We're done - no need to continue the flow
        return "done" 

from pocketflow import Flow

def create_agent_flow():
    """
    Create and connect the nodes to form a complete agent flow.
    
    The flow works like this:
    1. DecideAction node decides whether to search or use data_assistant
    2. If data_assistant, go to LocalRAG node, After LocalRAG completes, go back to DecideAction
    3. If answer, go to AnswerQuestion node
    
    Returns:
        Flow: A complete research agent flow
    """
    # Create instances of each node
    decide = DecideAction()
    rag = LocalRAG()
    answer = AnswerQuestion()

    # Connect the nodes
    # If DecideAction returns "search", go to SearchWeb
    decide - "data_assistant" >> rag
    
    # If DecideAction returns "answer", go to AnswerQuestion
    decide - "answer" >> answer
    
    # After SearchWeb completes and returns "decide", go back to DecideAction
    rag - "decide" >> decide

    # Create and return the flow, starting with the DecideAction node
    return Flow(start=decide)

# Reference : https://github.com/The-Pocket/PocketFlow/blob/main/cookbook/pocketflow-agent/flow.py
# Reference : https://github.com/The-Pocket/PocketFlow/blob/main/cookbook/pocketflow-agent/main.py
import sys

def main():
    """Simple function to process a question."""

    # this example may heavily rely on LLM model capabilities
    # try also : 
    # default_question = "based on local data, how to create a learnable representation and give me some intuitive examples"
    default_question = "how to create a learnable representation and give me some intuitive examples, also need to show data assistant analysis in final answers"

    # Get question from command line if provided with --
    question = default_question
    for arg in sys.argv[1:]:
        if arg.startswith("--"):
            question = arg[2:]
            break

    # Create the agent flow
    agent_flow = create_agent_flow()

    # Process the question
    shared = {"question": question}
    print(f"🤔 Processing question: {question}")
    agent_flow.run(shared)
    print("\n🎯 Final Answer:")
    res = shared.get("answer", "No answer found")
    print(res)

main()