import "dotenv/config";
import { setupTools } from "./tools";
import { setupResearchTeam } from "./teams/research";
import { HumanMessage } from "@langchain/core/messages";

async function main() {
  // Initialize tools
  const tools = await setupTools();
  
  // Setup research team
  const researchTeam = await setupResearchTeam(tools);
  
  // Test the research team
  const streamResults = researchTeam.stream(
    {
      messages: [new HumanMessage("What's the price of a big mac in Argentina?")],
    },
    { recursionLimit: 100 },
  );

  for await (const output of await streamResults) {
    if (!output?.__end__) {
      console.log(output);
      console.log("----");
    }
  }
}

main().catch(console.error); 