import { BaseMessage, HumanMessage, SystemMessage } from "@langchain/core/messages";
import { ChatPromptTemplate, MessagesPlaceholder } from "@langchain/core/prompts";
import { ChatOpenAI } from "@langchain/openai";
import { Runnable } from "@langchain/core/runnables";
import { JsonOutputToolsParser } from "@langchain/core/output_parsers/openai_tools";
import { Annotation, END, START, StateGraph } from "@langchain/langgraph";
import { createReactAgent } from "@langchain/langgraph/prebuilt";
import type { Tools } from "../tools";
import { z } from "zod";


// Define the research team state
export const ResearchTeamState = Annotation.Root({
  messages: Annotation<BaseMessage[]>({
    reducer: (x: BaseMessage[], y: BaseMessage[]) => x.concat(y),
  }),
  team_members: Annotation<string[]>({
    reducer: (x: string[], y: string[]) => x.concat(y),
  }),
  next: Annotation<string>({
    reducer: (x: string, y: string | null) => y ?? x,
    default: () => "supervisor",
  }),
  instructions: Annotation<string>({
    reducer: (x: string, y: string | null) => y ?? x,
    default: () => "Thoroughly research the current topic.",
  }),
  topic: Annotation<string>({
    reducer: (x: string, y: string | null) => y ?? x,
    default: () => "",
  }),

});

// Helper function to modify agent state
function agentStateModifier(
  systemPrompt: string,
  tools: any[],
  teamMembers: string[],
): (state: { messages: BaseMessage[] }, config: any) => BaseMessage[] {
  const toolNames = tools.map((t) => t.name).join(", ");
  const systemMsgStart = new SystemMessage(
    systemPrompt +
    "\nWork autonomously according to your specialty, using the tools available to you." +
    " Do not ask for clarification." +
    " Your other team members (and other teams) will collaborate with you with their own specialties." +
    ` You are chosen for a reason! You are one of the following team members: ${teamMembers.join(", ")}.`
  );
  const systemMsgEnd = new SystemMessage(
    `Supervisor instructions: ${systemPrompt}\n` +
    `Remember, you individually can only use these tools: ${toolNames}` +
    "\n\nEnd if you have already completed the requested task. Communicate the work completed."
  );

  return (state: { messages: BaseMessage[] }, _config: any): BaseMessage[] => 
    [systemMsgStart, ...state.messages, systemMsgEnd];
}

// Helper function to run agent node
async function runAgentNode(params: {
  state: any;
  agent: Runnable;
  name: string;
}) {
  const { state, agent, name } = params;
  const result = await agent.invoke({
    messages: state.messages,
  });
  const lastMessage = result.messages[result.messages.length - 1];
  return {
    messages: [new HumanMessage({ content: lastMessage.content, name })],
  };
}

export async function setupResearchTeam(tools: Tools) {
  const llm = new ChatOpenAI({ modelName: "gpt-4o" });

  // Create search node
  const searchNode = (state: typeof ResearchTeamState.State) => {
    const stateModifier = agentStateModifier(
      "You are a research assistant who can search for up-to-date info using the tavily search engine.",
      [tools.tavilyTool],
      state.team_members ?? ["Search"],
    );
    const searchAgent = createReactAgent({
      llm,
      tools: [tools.tavilyTool],
      stateModifier,
    });
    return runAgentNode({ state, agent: searchAgent, name: "Search" });
  };

  // Create research node
  const researchNode = (state: typeof ResearchTeamState.State) => {
    const stateModifier = agentStateModifier(
      "You are a research assistant who can scrape specified urls for more detailed information using the scrapeWebpage function.",
      [tools.stealthScrapeWebpage],
      state.team_members ?? ["WebScraper"],
    );
    const researchAgent = createReactAgent({
      llm,
      tools: [tools.stealthScrapeWebpage],
      stateModifier,
    }); 
    return runAgentNode({ state, agent: researchAgent, name: "WebScraper" });
  };

  // Create supervisor agent
  const supervisorAgent = await createTeamSupervisor(
    llm,
    "You are a supervisor tasked with managing a conversation between the" +
    " following workers:\n\n{team_members}.\n\nYour job is to thoroughly research {topic}." +
    " Respond with the worker to act next. Each worker will perform a" +
    " task and respond with their results and status. When finished," +
    " respond with FINISH.\n\n" +
    " Select strategically to minimize the number of steps taken.",
    ["Search", "WebScraper"],
  );

  // Create and return the research graph
  const researchGraph = new StateGraph(ResearchTeamState)
    .addNode("Search", searchNode)
    .addNode("supervisor", supervisorAgent)
    .addNode("WebScraper", researchNode)
    .addEdge("Search", "supervisor")
    .addEdge("WebScraper", "supervisor")
    .addConditionalEdges("supervisor", (x) => x.next, {
      Search: "Search",
      WebScraper: "WebScraper",
      FINISH: END,
    })
    .addEdge(START, "supervisor");

  return researchGraph.compile();
}

// Helper function to create team supervisor
async function createTeamSupervisor(
  llm: ChatOpenAI,
  systemPrompt: string,
  members: string[],
): Promise<Runnable> {
  const options = ["FINISH", ...members];
  const routeTool = {
    name: "route",
    description: "Select the next role.",
    schema: z.object({
      reasoning: z.string(),
      next: z.enum(["FINISH", ...members]),
      instructions: z.string().describe("The specific instructions of the sub-task the next role should accomplish."),
    })
  }

  let prompt = ChatPromptTemplate.fromMessages([
    ["system", systemPrompt],
    new MessagesPlaceholder("messages"),
    [
      "system",
      "Given the conversation above, who should act next? Or should we FINISH? Select one of: {options}",
    ],
  ]);
  
  prompt = await prompt.partial({
    options: options.join(", "),
    team_members: members.join(", "),
  });

  return prompt
    .pipe(
      llm.bindTools([routeTool], {
        tool_choice: "route",
      }),
    )
    .pipe(new JsonOutputToolsParser())
    .pipe((x: { args: { next: string; instructions: string } }[]) => ({
      next: x[0].args.next,
      instructions: x[0].args.instructions,
    }));
} 