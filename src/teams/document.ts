import { BaseMessage, HumanMessage, SystemMessage } from "@langchain/core/messages";
import { ChatPromptTemplate, MessagesPlaceholder } from "@langchain/core/prompts";
import { ChatOpenAI } from "@langchain/openai";
import { Runnable, RunnableLambda } from "@langchain/core/runnables";
import { JsonOutputToolsParser } from "@langchain/core/output_parsers/openai_tools";
import { Annotation, END, START, StateGraph } from "@langchain/langgraph";
import { createReactAgent } from "@langchain/langgraph/prebuilt";
import * as fs from "fs/promises";
import { z } from "zod";
import { Tools } from "../tools";


const WORKING_DIRECTORY = process.cwd() + "/workspace";

// Define the document writing team state
export const DocWritingState = Annotation.Root({
  messages: Annotation<BaseMessage[]>({
    reducer: (x, y) => x.concat(y),
  }),
  team_members: Annotation<string[]>({
    reducer: (x, y) => x.concat(y),
  }),
  next: Annotation<string>({
    reducer: (x, y) => y ?? x,
    default: () => "supervisor",
  }),
  current_files: Annotation<string>({
    reducer: (x, y) => (y ? `${x}\n${y}` : x),
    default: () => "No files written.",
  }),
  instructions: Annotation<string>({
    reducer: (x, y) => y ?? x,
    default: () => "Solve the human's question.",
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
    " Your other team members will collaborate with you with their own specialties." +
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

// Prelude function to check workspace state
const prelude = new RunnableLambda({
  func: async (state: {
    messages: BaseMessage[];
    next: string;
    instructions: string;
  }) => {
    let writtenFiles: string[] = [];
    if (
      !(await fs
        .stat(WORKING_DIRECTORY)
        .then(() => true)
        .catch(() => false))
    ) {
      await fs.mkdir(WORKING_DIRECTORY, { recursive: true });
    }
    try {
      const files = await fs.readdir(WORKING_DIRECTORY);
      for (const file of files) {
        writtenFiles.push(file);
      }
    } catch (error) {
      console.error(error);
    }
    const filesList = writtenFiles.length > 0
      ? "\nBelow are files your team has written to the directory:\n" +
        writtenFiles.map((f) => ` - ${f}`).join("\n")
      : "No files written.";
    return { ...state, current_files: filesList };
  },
});

export async function setupDocWritingTeam(tools: Tools) {
  const docWritingLlm = new ChatOpenAI({ modelName: "gpt-4o" });

  const { writeDocument, editDocument, readDocument, createOutline } = tools;

  // Create doc writing node
  const docWritingNode = (state: typeof DocWritingState.State) => {
    const stateModifier = agentStateModifier(
      `You are an expert writing a research document.\nBelow are files currently in your directory:\n${state.current_files}`,
      [writeDocument, editDocument, readDocument],
      state.team_members ?? [],
    );
    const docWriterAgent = createReactAgent({
      llm: docWritingLlm,
      tools: [writeDocument, editDocument, readDocument],
      stateModifier,
    });
    const contextAwareDocWriterAgent = prelude.pipe(docWriterAgent);
    return runAgentNode({ state, agent: contextAwareDocWriterAgent, name: "DocWriter" });
  };

  // Create note taking node
  const noteTakingNode = (state: typeof DocWritingState.State) => {
    const stateModifier = agentStateModifier(
      "You are an expert senior researcher tasked with writing a paper outline and" +
      ` taking notes to craft a perfect paper. ${state.current_files}`,
      [createOutline, readDocument],
      state.team_members ?? [],
    );
    const noteTakingAgent = createReactAgent({
      llm: docWritingLlm,
      tools: [createOutline, readDocument],
      stateModifier,
    });
    const contextAwareNoteTakingAgent = prelude.pipe(noteTakingAgent);
    return runAgentNode({ state, agent: contextAwareNoteTakingAgent, name: "NoteTaker" });
  };

  // Create chart generating node
  const chartGeneratingNode = (state: typeof DocWritingState.State) => {
    const stateModifier = agentStateModifier(
      "You are a data viz expert tasked with generating charts for a research project." +
      `${state.current_files}`,
      [readDocument],
      state.team_members ?? [],
    );
    const chartGeneratingAgent = createReactAgent({
      llm: docWritingLlm,
      tools: [readDocument],
      stateModifier,
    });
    const contextAwareChartGeneratingAgent = prelude.pipe(chartGeneratingAgent);
    return runAgentNode({ state, agent: contextAwareChartGeneratingAgent, name: "ChartGenerator" });
  };

  // Create supervisor agent
  const docTeamMembers = ["DocWriter", "NoteTaker", "ChartGenerator"];
  const docWritingSupervisor = await createTeamSupervisor(
    docWritingLlm,
    "You are a supervisor tasked with managing a conversation between the" +
      " following workers:  {team_members}. Given the following user request," +
      " respond with the worker to act next. Each worker will perform a" +
      " task and respond with their results and status. When finished," +
      " respond with FINISH.\n\n" +
      " Select strategically to minimize the number of steps taken.",
    docTeamMembers,
  );

  // Create and return the document writing graph
  const authoringGraph = new StateGraph(DocWritingState)
    .addNode("DocWriter", docWritingNode)
    .addNode("NoteTaker", noteTakingNode)
    .addNode("ChartGenerator", chartGeneratingNode)
    .addNode("supervisor", docWritingSupervisor)
    .addEdge("DocWriter", "supervisor")
    .addEdge("NoteTaker", "supervisor")
    .addEdge("ChartGenerator", "supervisor")
    .addConditionalEdges("supervisor", (x) => x.next, {
      DocWriter: "DocWriter",
      NoteTaker: "NoteTaker",
      ChartGenerator: "ChartGenerator",
      FINISH: END,
    })
    .addEdge(START, "supervisor");

  const enterAuthoringChain = RunnableLambda.from(
    ({ messages }: { messages: BaseMessage[] }) => {
      return {
        messages: messages,
        team_members: ["DocWriter", "NoteTaker", "ChartGenerator"],
      };
    },
  );

  return enterAuthoringChain.pipe(authoringGraph.compile());
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
  };

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