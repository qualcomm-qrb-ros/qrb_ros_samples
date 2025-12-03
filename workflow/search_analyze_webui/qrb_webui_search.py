#!/usr/bin/env python
# coding: utf-8

# use basic redis to communicate with nicegui
import redis
r = redis.Redis(host='localhost', port=6379, decode_responses=True)
pubsub = r.pubsub()
pubsub.subscribe('input_task')

# <place_holder> 
# use ollama or any other AI API provider

# Tavily Search Tool hand crafted
# https://github.com/tavily-ai/tavily-python?tab=readme-ov-file#getting-and-printing-the-full-search-api-response
from tavily import TavilyClient

tavily_client = TavilyClient(api_key=<your_key>)

def search_web(query_str):
    response = tavily_client.search(query_str)
    results = response["results"]
    results_str = "\n\n".join([f"URL: {r['url']}\nTitle: {r['title']}\nContent: {r['content']}" for r in results])
    return results_str

# Reference : https://github.com/The-Pocket/PocketFlow/blob/main/cookbook/pocketflow-agent/nodes.py
# changed DuckDuckGo with Tavily
from pocketflow import Node, Flow
import yaml
import sys

class SearchWeb(Node):
    def prep(self, shared):
        """Get the search query from the shared store."""
        return shared["search_query"]
        
    def exec(self, search_query):
        """Search the web for the given query."""
        # Call the search utility function
        print(f"🌐 Searching the web for: {search_query}")
        results = search_web(search_query)
        r.publish("exec_tool_result", results)
        return results
    
    def post(self, shared, prep_res, exec_res):
        """Save the search results and go back to the decision node."""
        # Add the search results to the context in the shared store
        previous = shared.get("context", "")
        shared["context"] = previous + "\n\nSEARCH: " + shared["search_query"] + "\nRESULTS: " + exec_res
        
        print(f"📚 Found information, analyzing results...")
        
        # Always go back to the decision node after searching
        return "decide"

class DecideAction(Node):
    def prep(self, shared):
        """Prepare the context and question for the decision-making process."""
        # Get the current context (default to "No previous search" if none exists)
        context = shared.get("context", "No previous search")
        # Get the question from the shared store
        question = shared["question"]
        # Return both for the exec step
        return question, context
        
    def exec(self, inputs):
        """Call the LLM to decide whether to search or answer."""
        question, context = inputs
        
        print(f"🤔 Agent deciding what to do next...")
        
        # Create a prompt to help the LLM decide what to do next with proper yaml formatting
        # be careful to construct your prompt to achieve best efficiency
        prompt = f"""
### CONTEXT
You are a research assistant that can search the web.
Question: {question}
Previous Research: {context}

### ACTION SPACE
[1] search
  Description: Look up more information on the web
  Parameters:
    - query (str): What to search for

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
action: search OR answer
reason: <why you chose this action>
answer: <if action is answer>
search_query: <specific search query if action is search>
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
        r.publish("yaml_decide_node",str(decision))
        
        return decision
    
    def post(self, shared, prep_res, exec_res):
        """Save the decision and determine the next step in the flow."""
        # If LLM decided to search, save the search query
        if exec_res["action"] == "search":
            shared["search_query"] = exec_res["search_query"]
            print(f"🔍 Agent decided to search for: {exec_res['search_query']}")
            r.publish("action_decide_node",f"🔍 Agent decided to search for: {exec_res['search_query']}")
        else:
            shared["context"] = exec_res["answer"] #save the context if LLM gives the answer without searching.
            print(f"💡 Agent decided to answer the question")
        
        # Return the action to determine the next node in the flow
        return exec_res["action"]

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
    1. DecideAction node decides whether to search or answer
    2. If search, go to SearchWeb node
    3. If answer, go to AnswerQuestion node
    4. After SearchWeb completes, go back to DecideAction
    
    Returns:
        Flow: A complete research agent flow
    """
    # Create instances of each node
    decide = DecideAction()
    search = SearchWeb()
    answer = AnswerQuestion()

    # Connect the nodes
    # If DecideAction returns "search", go to SearchWeb
    decide - "search" >> search
    
    # If DecideAction returns "answer", go to AnswerQuestion
    decide - "answer" >> answer
    
    # After SearchWeb completes and returns "decide", go back to DecideAction
    search - "decide" >> decide

    # Create and return the flow, starting with the DecideAction node
    return Flow(start=decide)

# Reference : https://github.com/The-Pocket/PocketFlow/blob/main/cookbook/pocketflow-agent/flow.py
# Reference : https://github.com/The-Pocket/PocketFlow/blob/main/cookbook/pocketflow-agent/main.py
import sys

def main():
    """Simple function to process a question."""
    # Default question
    while True:
        default_question = "Tell one liner joke and make it short and funny. Stop until user approved the final answer."
        
        print("Waiting for webui messages...")
        
        for message in pubsub.listen():
            if message['type'] == 'message':
                data = message['data']
                print(f"Received: {data}")
                break

        print("================================")
        print("clicked button, res is :")
        print(data)
        print("================================")
        default_question = data

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
        r.publish("result", res)

main()