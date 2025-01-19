import { ChatOpenAI } from "@langchain/openai";
import { Tools } from "../tools";
import { z } from "zod";
import { ChatPromptTemplate, MessagesPlaceholder } from "@langchain/core/prompts";
import { JsonOutputToolsParser } from "@langchain/core/output_parsers/openai_tools";

/**
 * OutlineBuilder agent that takes the research results (saved in the workspace or in memory)
 * and produces an outline containing references to relevant files. 
 */
export async function setupOutlineBuilder(tools: Tools) {
    
    const llm = new ChatOpenAI({ modelName: "gpt-4o" });

    const routeTool = {
      name: "outline_builder",
      description: "Create an outline referencing the relevant files collected by the research team.",
      schema: z.object({
        outline: z.string().describe("The structured markdown or textual outline."),
      }),
    };
  
    let prompt = ChatPromptTemplate.fromMessages([
      [
        "system",
        "You are the OutlineBuilder. Based on the recorded sources or files from the research step, " +
          "create a detailed outline. Include references to the relevant files in each section.",
      ],
      new MessagesPlaceholder("messages"),
      [
        "system",
        "Output JSON with a single field: 'outline'. The field should contain the entire outline.",
      ],
    ]);
  
    const outlineBuilder = prompt
      .pipe(
        llm.bindTools([routeTool], {
          tool_choice: "route",
        })
      )
      .pipe(new JsonOutputToolsParser())
      .pipe((x: { args: { next: string; instructions: string } }[]) => ({
        next: x[0].args.next,
        instructions: x[0].args.instructions,
      }));
  
    return outlineBuilder;
  }