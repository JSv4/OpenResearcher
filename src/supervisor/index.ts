import { BaseMessage } from "@langchain/core/messages";
import { JsonOutputToolsParser } from "@langchain/core/output_parsers/openai_tools";
import { ChatPromptTemplate, MessagesPlaceholder } from "@langchain/core/prompts";
import { Runnable, RunnableLambda } from "@langchain/core/runnables";
import { ChatOpenAI } from "@langchain/openai";
import { z } from "zod";

type SupervisorOutput = {
    next: string;
    instructions: string;
    topic: string;
};

export async function createTeamTopicSupervisor(
    llm: ChatOpenAI,
    systemPrompt: string,
    members: string[],
): Promise<Runnable<{ 
    messages: BaseMessage[], 
    brainstormed_queries: string[] 
}, {
    next: string,
    instructions: string,
    topic: string
}>> {
    const options = ["FINISH", ...members] as const;
    return RunnableLambda.from(async ({ messages, brainstormed_queries }: { 
        messages: BaseMessage[], 
        brainstormed_queries: string[] 
    }) => {
        const routeTool = {
            name: "route",
            description: "Select the next role and topic to work on.",
            schema: z.object({
                reasoning: z.string(),
                next: z.enum(options).describe("The specific actor to act next."),
                instructions: z.string().describe(
                    "The specific instructions of the sub-task the next role should accomplish."
                ),
                topic: z.enum(["NONE", ...brainstormed_queries] as [string, ...string[]]).describe(
                    "The specific topic the next role should work on."
                ),
            })
        };

        let prompt = ChatPromptTemplate.fromMessages([
            ["system", systemPrompt],
            ["human", "{user_request}"],
            new MessagesPlaceholder("messages"),
            [
                "system",
                "Given the conversation above, who should act next, and on which topic? Or should we FINISH (empty topic)? Select from one of these actors:\n\n{options}\n\nand one of these topics:\n\n{brainstormed_queries}",
            ],
        ]);

        prompt = await prompt.partial({
            options: options.join(", "),
            team_members: members.join(", "),
            brainstormed_queries: brainstormed_queries.join(", "),
        });

        const chain = prompt
            .pipe(
                llm.bindTools([routeTool], {
                    tool_choice: "route",
                }),
            )
            .pipe(new JsonOutputToolsParser())
            .pipe((x: { args: SupervisorOutput }[]) => ({
                next: x[0].args.next,
                instructions: x[0].args.instructions,
                topic: x[0].args.topic,
            }));

        return chain.invoke({ messages, user_request: messages[0].content });
    });
}