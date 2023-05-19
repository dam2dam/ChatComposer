export type ChatGPTPromptType = { role: string; content: string }[];
export interface ChatGPTApiResponseBodyType {
  prompt: ChatGPTPromptType;
  noteInfo: any[];
}
export interface ChatGPTApiRequestBodyType {
  prevData?: ChatCompletionRequestMessage[];
  message: string;
  genre: GenreType;
  tags: TagType[];
  instrument: InstrumentType;
}
