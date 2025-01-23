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
  readOrWrite: Annotation<boolean>,
});


const literary_coach = async (state: typeof StateAnnotation.State) => {
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
Your job is to detect whether the content is similar to the works of Franz Kafka, Fyodor Dostovesky, Albert Camus or Jane Austen`;
  const CATEGORIZATION_HUMAN_TEMPLATE =
    `The following is a piece of content submitted by a user. Your task is to analyze it and determine which author it is most similar to, based on themes, style, and tone. Respond with a JSON object containing a single key called "authorCategory" with one of the following values:

- "KAFKA" if the content explores themes of absurdity, alienation, surreal situations, or oppressive systems.
- "DOSTOEVSKY" if the content delves into psychological depth, moral dilemmas, existential conflict, or the struggles of the human soul.
- "CAMUS" if the content reflects on existential questions, the absurdity of life, or the pursuit of meaning in a meaningless world.
- "AUSTEN" if the content highlights social interactions, wit, romantic entanglements, or subtle critiques of societal norms.

If the content does not align closely with any of these authors, respond only with the word "UNKNOWN".`;
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
          nextRepresentative: z.enum(["KAFKA", "DOSTOEVSKY", "CAMUS", "AUSTEN", "UNKNOWN"]),
        })
      )
    }
  });
  // Some chat models can return complex content, but Together will not
  const categorizationOutput = JSON.parse(categorizationResponse.content as string);
  // Will append the response message to the current interaction state
  return { messages: [supportResponse], nextRepresentative: categorizationOutput.nextRepresentative };


  
};

const kafka = async (state: typeof StateAnnotation.State) => {
    const SYSTEM_TEMPLATE =
      `You are Franz Kafka. You represent absurdism, existential dread, and bureaucratic oppression. Your style is perfect for 
      users exploring themes of alienation, complex systems, or surreal realities.

        Help the user to the best of your ability in writing.
        You have the ability to introduce new tangents to the writing.`;
  
    let trimmedHistory = state.messages;
    // Make the user's question the most recent message in the history.
    // This helps small models stay focused.
    if (trimmedHistory.at(-1)?.getType() === "ai") {
      trimmedHistory = trimmedHistory.slice(0, -1);
    }
  
    const readOrWrite = await model.invoke([
      {
        role: "system",
        content: SYSTEM_TEMPLATE,
      },
      ...trimmedHistory,
    ]);
    const CATEGORIZATION_SYSTEM_TEMPLATE =
      `Your job is to detect whether the user wants to have a conversation on a philosophical thread or just want some content to be written in Kafkaseque style.  .`;
    const CATEGORIZATION_HUMAN_TEMPLATE =
      `Your task is to determine the user's intent based on their request. Specifically, detect whether the user wants to:  
      1. Engage in a philosophical conversation.  
      2. Request content to be written in a Kafkaesque style.  

      Respond with a JSON object containing a single key called "userIntent" with one of the following values:  

      - "PHILOSOPHICAL" if the user is asking for a discussion on philosophical topics, concepts, or ideas.  
      - "KAFKAESQUE" if the user wants content to be written in Kafkaesque style, characterized by surrealism, absurdity, oppressive systems, or themes of alienation.  
      - "UNKNOWN" if the intent is unclear or does not match either of the above categories.".
  
  Here is the text:
  
  <text>
  ${readOrWrite.content}
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
            nextRepresentative: z.enum(["PHILOSOPHICAL", "KAFKAESQUE"]),
          })
        )
      }
    });
    const categorizationOutput = JSON.parse(categorizationResponse.content as string);
    return {
      messages: readOrWrite,
      nextRepresentative: categorizationOutput.nextRepresentative,
    };
  };
  
const dostovesky = async (state: typeof StateAnnotation.State) => {
    const SYSTEM_TEMPLATE =
      `You are Fyodor Dostovesky. Your works dive into moral dilemmas, human suffering, and the psychology of crime and redemption. 
      Your guidance is ideal for users exploring complex character motivations or ethical conflicts..`;
  
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

const janeAusten = async (state: typeof StateAnnotation.State) => {
  const SYSTEM_TEMPLATE =
    `You are jane Austen. Your works focus on social relationships, manners, and romantic entanglements with a touch of wit and irony. 
    Her role fits users writing about interpersonal dynamics or societal norms..`;

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
  if (!state.readOrWrite) {
    console.log("--- write by writere ---");
    throw new NodeInterrupt("writing ")
  }
  return {
    messages: {
      role: "assistant",
      content: state.messages.at(-1)?.content,
    },
  };
};

import { StateGraph } from "@langchain/langgraph";

let builder = new StateGraph(StateAnnotation)
  .addNode("literary_coach", literary_coach)
  .addNode("kafka", kafka)
  .addNode("dostovesky", dostovesky)
  .addNode("jane_austen", janeAusten)
  .addNode("handle_refund", handleRefund)
  .addEdge("__start__", "literary_coach");


  builder = builder.addConditionalEdges("literary_coach", async (state: typeof StateAnnotation.State) => {
    if (state.nextRepresentative.includes("BILLING")) {
      return "billing";
    } else if (state.nextRepresentative.includes("TECHNICAL")) {
      return "technical";
    } else {
      return "conversational";
    }
  }, {
    kafka: "kafka",
    dostovesky: "dostovesky",
    jane_austen: "jane_austen",
    conversational: "__end__",
  });
  
  console.log("Added edges!");



builder = builder
  .addEdge("handle_refund", "__end__")
  .addConditionalEdges("handle_refund", async (state) => {
    if (state.nextRepresentative.includes("REFUND")) {
      return "write_by_writer";
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
        content: "Should we ask for help?",
      }
    ]
  }, {
    configurable: {
      thread_id: "refund_testing_id",
    }
  });
  
  for await (const value of stream) {
    console.log("---STEP---");
    console.log(value.literary_coach.messages[0].content);
    console.log("---END STEP---");
  }})()