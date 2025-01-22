

// import { TavilySearchResults } from "@langchain/community/tools/tavily_search";
// import { ChatOpenAI } from "@langchain/openai";
// import { MemorySaver } from "@langchain/langgraph";
// import { HumanMessage } from "@langchain/core/messages";
// import { createReactAgent } from "@langchain/langgraph/prebuilt";
//import 'dotenv/config';
import * as dotenv from 'dotenv';
dotenv.config();


// // Define the tools for the agent to use
// const agentTools = [new TavilySearchResults({ maxResults: 3 })];
// const agentModel = new ChatOpenAI({ temperature: 0, modelName: "gpt-4o-mini" });

// // Initialize memory to persist state between graph runs
// const agentCheckpointer = new MemorySaver();
// const agent = createReactAgent({
//   llm: agentModel,
//   tools: agentTools,
//   checkpointSaver: agentCheckpointer, //saves the state of the agent's memory at each step
// });

// // Now it's time to use!
// (async()=>{const agentFinalState = await agent.invoke(
//   { messages: [new HumanMessage("what is the current weather in New Jersey")] },
//   { configurable: { thread_id: "42" } },
// );

// console.log(
//   agentFinalState.messages[agentFinalState.messages.length - 1].content,
// );

// const agentNextState = await agent.invoke(
//   { messages: [new HumanMessage("what about ny")] },
//   { configurable: { thread_id: "42" } },
// );

// console.log(
//   agentNextState.messages[agentNextState.messages.length - 1].content,
// );})();

// agent.ts

// IMPORTANT - Add your API keys here. Be careful not to publish them.

import { TavilySearchResults } from "@langchain/community/tools/tavily_search";
import { ChatOpenAI } from "@langchain/openai"
import { HumanMessage } from "@langchain/core/messages";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { StateGraph, MessagesAnnotation } from "@langchain/langgraph";

// Define the tools for the agent to use
const tools = [new TavilySearchResults({ maxResults: 3 })];
const toolNode = new ToolNode(tools);

// Create a model and give it access to the tools
const model = new ChatOpenAI({
  model: "gpt-4o-mini",
  temperature: 0,
}).bindTools(tools);

// Define the function that determines whether to continue or not
function shouldContinue({ messages }: typeof MessagesAnnotation.State) {
  const lastMessage = messages[messages.length - 1];

  // If the LLM makes a tool call, then we route to the "tools" node
  if (lastMessage.additional_kwargs.tool_calls) {
    return "tools";
  }
  // Otherwise, we stop (reply to the user) using the special "__end__" node
  return "__end__";
}

// Define the function that calls the model
async function callModel(state: typeof MessagesAnnotation.State) {
  const response = await model.invoke(state.messages);

  // We return a list, because this will get added to the existing list
  return { messages: [response] };
}

// Define a new graph
const workflow = new StateGraph(MessagesAnnotation)
  .addNode("agent", callModel)
  .addEdge("__start__", "agent") // __start__ is a special name for the entrypoint
  .addNode("tools", toolNode)
  .addEdge("tools", "agent")
  .addConditionalEdges("agent", shouldContinue);

// Finally, we compile it into a LangChain Runnable.
const app = workflow.compile(); //compiles the state graph into a "Runnable," which is an executable agent that can be invoked.

// Use the agent
(async()=>{const finalState = await app.invoke({
  messages: [new HumanMessage("what is the weather in New Delhi")],
});
console.log(finalState.messages[finalState.messages.length - 1].content);

const nextState = await app.invoke({
  // Including the messages from the previous run gives the LLM context.
  // This way it knows we're asking about the weather in NY
  messages: [...finalState.messages, new HumanMessage("what about Uttarkashi")],
});
console.log(nextState.messages[nextState.messages.length - 1].content);})();

