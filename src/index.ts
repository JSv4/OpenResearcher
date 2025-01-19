import "dotenv/config";
import { setupTools } from "./tools";
import { setupResearchTeam } from "./teams/research";
import { setupBrainstormer } from "./teams/brainstorm";
import { BaseMessage, HumanMessage } from "@langchain/core/messages";
import { setupDocWritingTeam } from "./teams/document";
import { Annotation, END, START, StateGraph } from "@langchain/langgraph";
import { RunnableLambda } from "@langchain/core/runnables";
import { createTeamTopicSupervisor } from "./supervisor";
import { ChatOpenAI } from "@langchain/openai";

const State = Annotation.Root({
    user_request: Annotation<string>({
        reducer: (x, y) => y ?? x,
        default: () => "Do nothing!",
    }),
    messages: Annotation<BaseMessage[]>({
        reducer: (x, y) => x.concat(y),
    }),
    next: Annotation<string>({
        reducer: (x, y) => y ?? x,
        default: () => "Brainstormer",
    }),
    instructions: Annotation<string>({
        reducer: (x, y) => y ?? x,
        default: () => "Brainstorm how to answer the user's question, then complete identified research, and, finally, write a clear, concise, and comprehensive report to answer the user.",
    }),
    user_topic: Annotation<string>({
        reducer: (x, y) => y ?? x,
        default: () => "",
    }),
    brainstormed_queries: Annotation<string[]>({
        reducer: (x, y) => x.concat(y),
        default: () => [],
    }),
    active_topic: Annotation<string>({
        reducer: (x, y) => y ?? x,
        default: () => "",
    }),
    outline_plan: Annotation<string>({
        reducer: (x, y) => y ?? x,
        default: () => "No outline yet.",
    }),
});

const getMessages = RunnableLambda.from((state: typeof State.State) => {
    return { messages: state.messages };
});

const getBrainstormedQueries = RunnableLambda.from((state: typeof State.State) => {
    return { queries: state.brainstormed_queries };
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

    // Setup brainstormer
    const brainstormerAgent = await setupBrainstormer(tools);

    const supervisorNode = await createTeamTopicSupervisor(
        llm,
        "You are a supervisor tasked with managing a conversation between the" +
        " following teams: {team_members}. Given this user request:\n\n{user_request}\n\n," +
        " and a list of related research topics, you are responsible for a) ensuring all " +
        " identified topics are researched comprehensively and accuratly and then b) each topics has a comprehensive " +
        " section written about it (with citations to source). You will respond with " +
        " the name of the next worker to call and the relevant topic to work on. Each worker will perform a" +
        " task and respond with their results and status. When finished," +
        " respond with FINISH.\n\n" +
        " Select strategically to minimize the number of steps taken.",
        ["ResearchTeam", "PaperWritingTeam", "Brainstormer"],
    );

    const superGraph = new StateGraph(State)
        .addNode("Brainstormer", async (input) => {
            const getMessagesResult = await getMessages.invoke(input);
            const brainstormerResult = await brainstormerAgent.invoke({
                messages: getMessagesResult.messages,
            });
            const joinGraphResult = await joinGraph.invoke({
                brainstormerResult: brainstormerResult.queries,
                next: "ResearchTeam",
            });
            return joinGraphResult;
        })
        .addNode("ResearchTeam", async (input) => {
            const getMessagesResult = await getMessages.invoke(input);
            const researchChainResult = await researchTeam.invoke({
                messages: getMessagesResult.messages,
                topic: input.active_topic,
            });
            const joinGraphResult = await joinGraph.invoke({
                messages: researchChainResult.messages,
            });
            return joinGraphResult;
        })
        .addNode("PaperWritingTeam", getMessages.pipe(documentTeam).pipe(joinGraph))
        .addNode("supervisor", async (state) => {
            return supervisorNode.invoke({
                messages: state.messages,
                brainstormed_queries: state.brainstormed_queries
            });
        })
        .addEdge("supervisor", "Brainstormer")
        .addEdge("ResearchTeam", "supervisor")
        .addEdge("PaperWritingTeam", "supervisor")
        .addConditionalEdges("supervisor", (x) => x.next, {
            PaperWritingTeam: "PaperWritingTeam",
            ResearchTeam: "ResearchTeam",
            FINISH: END,
        })
        .addEdge(START, "Brainstormer");

    const compiledSuperGraph = superGraph.compile();

    const resultStream = compiledSuperGraph.stream(
        {
            user_request: "Create a comprehensive overview of what tax sales are in Tarrant county TX and what the process is to buy property.",
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