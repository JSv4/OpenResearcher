import { Document } from "@langchain/core/documents";
import { AsyncCaller } from "@langchain/core/utils/async_caller";
import { BaseDocumentLoader } from "@langchain/core/document_loaders/base";
import puppeteer from "puppeteer-extra";
import StealthPlugin from "puppeteer-extra-plugin-stealth";
import { parseDocument } from "htmlparser2";
import * as cheerio from "cheerio";
import type { CheerioAPI, SelectorType } from "cheerio";
import type { WebBaseLoader } from "./html";
import { DynamicStructuredTool } from "@langchain/core/tools";
import {z} from "zod";

/**
 * Parameters for configuring the PuppeteerStealthLoader
 */
export interface PuppeteerStealthLoaderParams {
  /** Timeout in milliseconds for the page load */
  timeout?: number;
  /** Selector to extract text from the document */
  selector?: SelectorType;
  /** Additional headers to send with the request */
  headers?: HeadersInit;
}

/**
 * A document loader that uses Puppeteer with stealth mode to load web pages
 * while avoiding detection, then processes the content using Cheerio.
 */
export class PuppeteerStealthLoader
  extends BaseDocumentLoader
  implements WebBaseLoader
{
  public timeout: number;
  public caller: AsyncCaller;
  public selector: SelectorType;
  public headers?: HeadersInit;

  constructor(
    public webPath: string,
    fields?: PuppeteerStealthLoaderParams
  ) {
    super();
    const { timeout, selector, headers } = fields ?? {};
    this.timeout = timeout ?? 10000;
    this.caller = new AsyncCaller({});
    this.selector = selector ?? "body";
    this.headers = headers;
  }

  /**
   * Scrapes the webpage using Puppeteer in stealth mode
   */
  async scrape(): Promise<CheerioAPI> {
    puppeteer.use(StealthPlugin());
    const browser = await puppeteer.launch({ headless: true });

    try {
      const page = await browser.newPage();
      if (this.headers) {
        await page.setExtraHTTPHeaders(this.headers);
      }
      
      await page.goto(this.webPath, { 
        waitUntil: "networkidle2",
        timeout: this.timeout 
      });

      const rawHtml = await page.evaluate(() => document.documentElement.outerHTML);
      const doc = parseDocument(rawHtml);
      const $ = cheerio.load(doc);

      // Remove non-content elements
      $("script").remove();
      $("style").remove();
      $("link[rel='stylesheet']").remove();
      $("meta").remove();
      $("noscript").remove();
      $("iframe").remove();

      return $;
    } finally {
      await browser.close();
    }
  }

  /**
   * Loads the webpage and returns it as a Document
   */
  async load(): Promise<Document[]> {
    const $ = await this.scrape();
    const title = $("title").text();
    const text = $(this.selector).text();
    
    return [
      new Document({
        pageContent: text,
        metadata: {
          source: this.webPath,
          title,
        },
      }),
    ];
  }
}

/**
 * A tool that uses PuppeteerStealth to scrape a webpage while avoiding detection
 */
export const stealthScrapeWebpage = new DynamicStructuredTool({
  name: "stealthScrapeWebpage",
  description: "Scrapes a webpage using stealth mode to avoid detection",
  schema: z.object({
    url: z.string().describe("The URL of the webpage to scrape"),
  }),
  async func({ url }: { url: string }): Promise<string> {
    const loader = new PuppeteerStealthLoader(url);
    const docs = await loader.load();
    return docs[0].pageContent;
  },
});