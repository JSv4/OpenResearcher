import { ChatPromptTemplate, MessagesPlaceholder } from "@langchain/core/prompts";
import { ChatOpenAI } from "@langchain/openai";
import { z } from "zod";
import { JsonOutputToolsParser } from "@langchain/core/output_parsers/openai_tools";
import { Tools } from "../tools";


/**
 * Example Brainstormer agent that proposes queries for the user_topic.
 * This should produce an array of queries in a structured format,
 * stored in the 'brainstormed_queries' annotation.
 */
export async function setupBrainstormer(tools: Tools) {
  const llm = new ChatOpenAI({ modelName: "gpt-4o" });
  // We can define a simple tool or just parse JSON from the LLM output
  const routeTool = {
    name: "brainstorm",
    description: "Generate a list of potential research queries based on the user topic.",
    schema: z.object({
      queries: z.array(z.string()),
    }),
  };

  let prompt = ChatPromptTemplate.fromMessages([
    ["system", "You are a research expert who helps break down complex topics into " +
        "meaningful sub-queries. For each iteration, generate unique, insightful questions " +
        "that explore different aspects of the main topic.\n\n" +
        "Guidelines:\n" +
        "- Each query should explore a distinct aspect of the topic\n" +
        "- Include both broad and specific questions\n" +
        "- Consider different perspectives and angles\n" +
        "- Ensure queries build upon each other\n" +
        "- Maintain relevance to the original question\n" +
        "- IMPORTANT: Do not use semicolons within individual queries\n\n" +
        "Generate between 3-10 queries per iteration, depending on the topic's complexity.\n" +
        "Continue generating new iterations until you feel all important aspects have been covered."],
    new MessagesPlaceholder("messages"),
    [
      "system",
      "Output JSON with the field 'queries' containing an array of possible search queries.",
    ],
  ]);

  const brainstormer = prompt
    .pipe(
      llm.bindTools([routeTool], {
        tool_choice: "brainstorm",
      })
    )
    .pipe(new JsonOutputToolsParser())
    .pipe((x: {args: { queries: string[] } }[] ) => ({
      queries: x[0].args.queries
    }));

  return brainstormer;
}