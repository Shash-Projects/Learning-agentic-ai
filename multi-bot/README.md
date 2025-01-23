Different "roles" or "agents" are simulated by switching system prompts.

The SYSTEM_TEMPLATE in each function (initialSupport, billingSupport, technicalSupport) defines how the AI behaves.
MessagesAnnotation.spec: Stores conversation history.
nextRepresentative: Tracks the next agent/role (PHILOSOPHER, VICTORIA, etc.).
refundAuthorized: Determines whether you are worthy for a waifu or not. (boolean).