import { JsonOutputToolsParser } from "@langchain/core/output_parsers/openai_tools";
import { ChatPromptTemplate, MessagesPlaceholder } from "@langchain/core/prompts";
import { Runnable } from "@langchain/core/runnables";
import { ChatOpenAI } from "@langchain/openai";
import { z } from "zod";

type SupervisorOutput = {
    next: string;
    instructions: string;
};

export async function createTeamSupervisor(
    llm: ChatOpenAI,
    systemPrompt: string,
    members: string[],
): Promise<Runnable<any, SupervisorOutput>> {
    const options = ["FINISH", ...members] as const;
    const routeTool = {
        name: "route",
        description: "Select the next role.",
        schema: z.object({
            reasoning: z.string(),
            next: z.enum(options),
            instructions: z.string().describe(
                "The specific instructions of the sub-task the next role should accomplish."
            ),
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

    const supervisor = prompt
        .pipe(
            llm.bindTools([routeTool], {
                tool_choice: "route",
            }),
        )
        .pipe(new JsonOutputToolsParser())
        .pipe((x: { args: SupervisorOutput }[]) => ({
            next: x[0].args.next,
            instructions: x[0].args.instructions,
        }));

    return supervisor;
}