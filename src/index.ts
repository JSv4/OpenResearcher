import "dotenv/config";
import { setupTools } from "./tools";
import { setupResearchTeam } from "./teams/research";
import { HumanMessage } from "@langchain/core/messages";
import { setupDocWritingTeam } from "./teams/document";

async function main() {
  // Initialize tools
  const tools = await setupTools();
  
  // Setup research team
//   const researchTeam = await setupResearchTeam(tools);
  
//   // Test the research team
//   const streamResults = researchTeam.stream(
//     {
//       messages: [new HumanMessage("Find me three interesting stories about SpaceX and then please fetch me short summaries of each.")],
//     },
//     { recursionLimit: 100 },
//   );

//   for await (const output of await streamResults) {
//     if (!output?.__end__) {
//       console.log(output);
//       console.log("----");
//     }
//   }

    // Test document team
    const documentTeam = await setupDocWritingTeam(tools);

    const docStreamResults = documentTeam.stream(
        {
          messages: [new HumanMessage("Write me an exemplary product specification template for a law firm.")],
        },
        { recursionLimit: 100 },
      );
    
      for await (const output of await docStreamResults) {
        if (!output?.__end__) {
          console.log(output);
          console.log("----");
        }
      }
}

main().catch(console.error); 