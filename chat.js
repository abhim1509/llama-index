import "dotenv/config";
const keys = process.env.OPENAI_API_KEY; // read API key from .env

import { Document, VectorStoreIndex, SimpleDirectoryReader } from "llamaindex";
import * as llamaIndex from "llamaindex";

let customLLM = new llamaIndex.OpenAI();
let customEmbedding = new llamaIndex.OpenAIEmbedding();

//Read all of the documents in a directory.
// * By default, supports the list of file types
//txt,pdf,csv,md,docx,htm,html,jpg,jpeg,png,gif
const documents = await new SimpleDirectoryReader().loadData({
  directoryPath: "./data",
});

//* High level API: split documents, get keywords, and build index.
const index = await VectorStoreIndex.fromDocuments(
  documents,
  (show_progress = true)
);

let customServiceContext = new llamaIndex.serviceContextFromDefaults({
  llm: customLLM,
  embedModel: customEmbedding,
});

let customQaPrompt = function ({ context = "", query = "" }) {
  return `Context information is below.
        ---------------------
        ${context}
        ---------------------
        Given the context information, answer the query. 
        If you don't have query or answer from context, mention it is not present
        Query: ${query}
        Answer:`;
};

let customResponseBuilder = new llamaIndex.SimpleResponseBuilder(
  customServiceContext,
  customQaPrompt
);

let customSynthesizer = new llamaIndex.ResponseSynthesizer({
  responseBuilder: customResponseBuilder,
  serviceContext: customServiceContext,
});

let customRetriever = new llamaIndex.VectorIndexRetriever({
  index,
});

let customQueryEngine = new llamaIndex.RetrieverQueryEngine(
  customRetriever,
  customSynthesizer
);

let response2 = await customQueryEngine.query({
  query: "How can I track my order?",
});

console.log(response2.response);
