import "dotenv/config";
import { setupTools } from "./tools";
import { setupResearchTeam } from "./teams/research";
import { BaseMessage, HumanMessage } from "@langchain/core/messages";
import { setupDocWritingTeam } from "./teams/document";
import { Annotation, END, START, StateGraph } from "@langchain/langgraph";
import { RunnableLambda } from "@langchain/core/runnables";
import { createTeamSupervisor } from "./supervisor";
import { ChatOpenAI } from "@langchain/openai";

const State = Annotation.Root({
    messages: Annotation<BaseMessage[]>({
        reducer: (x, y) => x.concat(y),
    }),
    next: Annotation<string>({
        reducer: (x, y) => y ?? x,
        default: () => "ResearchTeam",
    }),
    instructions: Annotation<string>({
        reducer: (x, y) => y ?? x,
        default: () => "Resolve the user's request.",
    }),
});

const getMessages = RunnableLambda.from((state: typeof State.State) => {
    return { messages: state.messages };
});

const joinGraph = RunnableLambda.from((response: any) => {
    return {
        messages: [response.messages[response.messages.length - 1]],
    };
});

async function main() {
    // Initialize tools
    const tools = await setupTools();

    const llm = new ChatOpenAI({ modelName: "gpt-4o" });

    //Setup research team
    const researchTeam = await setupResearchTeam(tools);

    // Test document team
    const documentTeam = await setupDocWritingTeam(tools);

    const supervisorNode = await createTeamSupervisor(
        llm,
        "You are a supervisor tasked with managing a conversation between the" +
        " following teams: {team_members}. Given the following user request," +
        " respond with the worker to act next. Each worker will perform a" +
        " task and respond with their results and status. When finished," +
        " respond with FINISH.\n\n" +
        " Select strategically to minimize the number of steps taken.",
        ["ResearchTeam", "PaperWritingTeam"],
    );

    const superGraph = new StateGraph(State)
        .addNode("ResearchTeam", async (input) => {
            const getMessagesResult = await getMessages.invoke(input);
            const researchChainResult = await researchTeam.invoke({
                messages: getMessagesResult.messages,
            });
            const joinGraphResult = await joinGraph.invoke({
                messages: researchChainResult.messages,
            });
            return joinGraphResult;
        })
        .addNode("PaperWritingTeam", getMessages.pipe(documentTeam).pipe(joinGraph))
        .addNode("supervisor", supervisorNode)
        .addEdge("ResearchTeam", "supervisor")
        .addEdge("PaperWritingTeam", "supervisor")
        .addConditionalEdges("supervisor", (x) => x.next, {
            PaperWritingTeam: "PaperWritingTeam",
            ResearchTeam: "ResearchTeam",
            FINISH: END,
        })
        .addEdge(START, "supervisor");

    const compiledSuperGraph = superGraph.compile();

    const resultStream = compiledSuperGraph.stream(
        {
          messages: [
            new HumanMessage(
              "Create a comprehensive overview of what tax sales are in Tarrant county TX and what the process is to buy property.",
            ),
          ],
        },
        { recursionLimit: 150 },
      );
      
      for await (const step of await resultStream) {
        if (!step.__end__) {
          console.log(step);
          console.log("---");
        }
      }

}

main().catch(console.error); 