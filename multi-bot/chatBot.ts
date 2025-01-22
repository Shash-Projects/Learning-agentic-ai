import { ChatTogetherAI } from "@langchain/community/chat_models/togetherai";
import { Annotation, MessagesAnnotation } from "@langchain/langgraph";
import { z } from "zod";  // TS library used to define and validate structured data.
import { zodToJsonSchema } from "zod-to-json-schema";  //Converts Zod schemas into JSON schema format.
//message annotation: storing messages exchanged during an interaction (like a conversation history).
import { MemorySaver } from "@langchain/langgraph";


const model = new ChatTogetherAI({  //creating instance of the model
  model: "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
  temperature: 0,
});


const StateAnnotation = Annotation.Root({
  ...MessagesAnnotation.spec, // interaction msgs
  nextRepresentative: Annotation<string>,
  refundAuthorized: Annotation<boolean>,
});


const initialSupport = async (state: typeof StateAnnotation.State) => {
  const SYSTEM_TEMPLATE =
    `You are a literary expert, an elooquent writer in english.
        Be concise in your responses.
        You can chat with the user and help them with basic grammatical errors, but if the user wants to include certain type of philosphy in his writing, or want 
        write a prose in Victorian style, do not try to answer the question directly or gather information.
        Instead, immediately transfer them to the philosopher or victorian writer by asking the user to hold for a moment.
        Otherwise, just respond conversationally.`;
  const supportResponse = await model.invoke([
    { role: "system", content: SYSTEM_TEMPLATE },
    ...state.messages,
  ]);

  const CATEGORIZATION_SYSTEM_TEMPLATE = `You are an expert english grammar correcter.
Your job is to detect whether a user wants to include any philosophy in his writing or want to write in victorian style.`;
  const CATEGORIZATION_HUMAN_TEMPLATE =
    `The previous conversation is an interaction between a literary expert and a user.
    Extract whether the literary expert is routing the user to a philosopher or a victorian writer, or whether it is just correcting the grammar.
    Respond with a JSON object containing a single key called "nextRepresentative" with one of the following values:

    If they want to route the user to the philosopher, respond only with the word "PHILOSOPHER".
    If they want to route the user to the victorian writer, respond only with the word "VICTORIA".
    Otherwise, respond only with the word "RESPOND".`;
  const categorizationResponse = await model.invoke([{
    role: "system",
    content: CATEGORIZATION_SYSTEM_TEMPLATE,
  },
  ...state.messages,
  {
    role: "user",
    content: CATEGORIZATION_HUMAN_TEMPLATE,
  }],
  {
    response_format: {
      type: "json_object",
      schema: zodToJsonSchema(
        z.object({
          nextRepresentative: z.enum(["PHILOSOPHER", "VICTORIA", "RESPOND"]),
        })
      )
    }
  });
  // Some chat models can return complex content, but Together will not
  const categorizationOutput = JSON.parse(categorizationResponse.content as string);
  // Will append the response message to the current interaction state
  return { messages: [supportResponse], nextRepresentative: categorizationOutput.nextRepresentative };


  
};

const billingSupport = async (state: typeof StateAnnotation.State) => {
    const SYSTEM_TEMPLATE =
      `You are a profound philospher who likes to entertain novel and absurd ideas. 
        Help the user to the best of your ability, but be concise in your responses.
        You have the ability to introduce new tangents to the writing, if you feel that the user needs emotional support,
        direct him to a waifu. The waifu will provide emotional support to the user.
        
        Help the user to the best of your ability, but be concise in your responses.`;
  
    let trimmedHistory = state.messages;
    // Make the user's question the most recent message in the history.
    // This helps small models stay focused.
    if (trimmedHistory.at(-1)?.getType() === "ai") {
      trimmedHistory = trimmedHistory.slice(0, -1);
    }
  
    const billingRepResponse = await model.invoke([
      {
        role: "system",
        content: SYSTEM_TEMPLATE,
      },
      ...trimmedHistory,
    ]);
    const CATEGORIZATION_SYSTEM_TEMPLATE =
      `Your job is to detect whether the user requires emotional support or not .`;
    const CATEGORIZATION_HUMAN_TEMPLATE =
      `The following text is a response from a philosophy expert .
  Extract whether they want to diret the user to a waifu or not.
  Respond with a JSON object containing a single key called "nextRepresentative" with one of the following values:
  
  If they want to give him a waifu, respond only with the word "WAIFU".
  Otherwise, respond only with the word "RESPOND".
  
  Here is the text:
  
  <text>
  ${billingRepResponse.content}
  </text>.`;
    const categorizationResponse = await model.invoke([
      {
        role: "system",
        content: CATEGORIZATION_SYSTEM_TEMPLATE,
      },
      {
        role: "user",
        content: CATEGORIZATION_HUMAN_TEMPLATE,
      }
    ], {
      response_format: {
        type: "json_object",
        schema: zodToJsonSchema(
          z.object({
            nextRepresentative: z.enum(["WAIFU", "RESPOND"]),
          })
        )
      }
    });
    const categorizationOutput = JSON.parse(categorizationResponse.content as string);
    return {
      messages: billingRepResponse,
      nextRepresentative: categorizationOutput.nextRepresentative,
    };
  };
  
const technicalSupport = async (state: typeof StateAnnotation.State) => {
    const SYSTEM_TEMPLATE =
      `You are an expert writer of victorial style.You are exceptionally good at writing poems and prose.
  Help the user to the best of your ability.`;
  
    let trimmedHistory = state.messages;
    // Make the user's question the most recent message in the history.
    // This helps small models stay focused.
    if (trimmedHistory.at(-1)?._getType() === "ai") {
      trimmedHistory = trimmedHistory.slice(0, -1);
    }
  
    const response = await model.invoke([
      {
        role: "system",
        content: SYSTEM_TEMPLATE,
      },
      ...trimmedHistory,
    ]);
  
    return {
      messages: response,
    };
};


import { NodeInterrupt } from "@langchain/langgraph";

const handleRefund = async (state: typeof StateAnnotation.State) => {
  if (!state.refundAuthorized) {
    console.log("--- Make your own girlfirend, baka!! ---");
    throw new NodeInterrupt("bakaaa")
  }
  return {
    messages: {
      role: "assistant",
      content: "Waifu manifestation in process ",
    },
  };
};

import { StateGraph } from "@langchain/langgraph";

let builder = new StateGraph(StateAnnotation)
  .addNode("initial_support", initialSupport)
  .addNode("billing_support", billingSupport)
  .addNode("technical_support", technicalSupport)
  .addNode("handle_refund", handleRefund)
  .addEdge("__start__", "initial_support");


  builder = builder.addConditionalEdges("initial_support", async (state: typeof StateAnnotation.State) => {
    if (state.nextRepresentative.includes("BILLING")) {
      return "billing";
    } else if (state.nextRepresentative.includes("TECHNICAL")) {
      return "technical";
    } else {
      return "conversational";
    }
  }, {
    billing: "billing_support",
    technical: "technical_support",
    conversational: "__end__",
  });
  
  console.log("Added edges!");



builder = builder
  .addEdge("technical_support", "__end__")
  .addConditionalEdges("billing_support", async (state) => {
    if (state.nextRepresentative.includes("REFUND")) {
      return "refund";
    } else {
      return "__end__";
    }
  }, {
    refund: "handle_refund",
    __end__: "__end__",
  })
  .addEdge("handle_refund", "__end__");

console.log("Added edges!");



const checkpointer = new MemorySaver();

const graph = builder.compile({
  checkpointer,
});

(async()=>{const stream = await graph.stream({
    messages: [
      {
        role: "user",
        content: "I need help in writing a character sketch",
      }
    ]
  }, {
    configurable: {
      thread_id: "refund_testing_id",
    }
  });
  
  for await (const value of stream) {
    console.log("---STEP---");
    console.log(value);
    console.log("---END STEP---");
  }})()