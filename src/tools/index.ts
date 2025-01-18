import { DynamicStructuredTool } from "@langchain/core/tools";
import { tavilyTool } from "./search/SearchEngines";
import { stealthScrapeWebpage } from "./webloaders/PuppeteerStealth";
import { ZodObject, ZodString, ZodTypeAny } from "zod";
import { 
  writeDocumentTool,
  createOutlineTool,
  editDocumentTool,
  readDocumentTool
} from "./document";


export interface Tools {
  tavilyTool: typeof tavilyTool;
  stealthScrapeWebpage: DynamicStructuredTool<ZodObject<{
      url: ZodString;
  }, "strip", ZodTypeAny, {
      url: string;
  }, {
      url: string;
  }>>;
  writeDocument: typeof writeDocumentTool;
  editDocument: typeof editDocumentTool;
  readDocument: typeof readDocumentTool;
  createOutline: typeof createOutlineTool;
}

export function setupTools(): Tools {
  return {
    tavilyTool,
    stealthScrapeWebpage,
    writeDocument: writeDocumentTool,
    editDocument: editDocumentTool,
    readDocument: readDocumentTool,
    createOutline: createOutlineTool,
  };
} 