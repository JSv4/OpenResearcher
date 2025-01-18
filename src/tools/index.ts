import { DynamicStructuredTool } from "@langchain/core/tools";
import { tavilyTool } from "./search/SearchEngines";
import { stealthScrapeWebpage } from "./webloaders/PuppeteerStealth";
import { ZodObject, ZodString, ZodTypeAny } from "zod";


export interface Tools {
  tavilyTool: typeof tavilyTool;
  stealthScrapeWebpage: DynamicStructuredTool<ZodObject<{
      url: ZodString;
  }, "strip", ZodTypeAny, {
      url: string;
  }, {
      url: string;
  }>>
}

export function setupTools(): Tools {
  return {
    tavilyTool,
    stealthScrapeWebpage,
  };
} 