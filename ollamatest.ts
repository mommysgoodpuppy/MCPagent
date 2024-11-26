import { Ollama, ChatResponse, ToolCall, ChatRequest, Message } from 'npm:ollama'
import { StdioServerTransport } from "npm:@modelcontextprotocol/sdk/server/stdio.js";
import { Readable, Writable } from "node:stream";
import { JSONRPCMessage } from "npm:@modelcontextprotocol/sdk/types.js";
import { spawn } from "node:child_process";
import readline from "node:readline";

export const config = {
    model: "qwen2.5-coder:7b",
    host: "http://localhost:11434/",
    scriptPath: "index.ts",
    allowedDirectories: ["C:/Users/me/Desktop"]
};

export class MCPStdioClient {
    private transport: StdioServerTransport;
    private process: ReturnType<typeof spawn>;

    constructor(scriptPath: string, allowedDirectories: string[]) {
        // Spawn the MCP server process
        this.process = spawn("deno", ["run", "-A", scriptPath, ...allowedDirectories], {
            stdio: ["pipe", "pipe", "inherit"], // Connect stdin and stdout for communication
        });

        // Set up the StdioServerTransport with the process streams
        this.transport = new StdioServerTransport(
            this.process.stdout as Readable,
            this.process.stdin as Writable
        );

        // Log errors from the MCP process
        this.process.on("error", (error) => {
            console.error("Failed to spawn MCP server:", error.message);
        });

        this.process.on("exit", (code) => {
            console.log(`MCP server exited with code ${code}`);
        });
    }

    async start() {
        try {
            await this.transport.start();
            console.log("MCP Stdio Client connected to server.");
        } catch (error) {
            throw new Error(`Failed to start MCP transport: ${error.message}`);
        }
    }

    async sendMessage(message: JSONRPCMessage): Promise<any> {
        return new Promise((resolve, reject) => {
            this.transport.onmessage = (response) => {
                if (response.error) {
                    reject(new Error(`MCP Error: ${response.error.message}`));
                } else {
                    resolve(response.result);
                }
            };
            this.transport.onerror = (error) => reject(error);
            this.transport.send(message).catch((error) => reject(error));
        });
    }

    async listTools(): Promise<any[]> {
        return this.sendMessage({
            jsonrpc: "2.0",
            id: 1,
            method: "tools/list",
            params: {},
        });
    }

    async callTool(name: string, args: Record<string, any>): Promise<any> {
        return this.sendMessage({
            jsonrpc: "2.0",
            id: 2,
            method: "tools/call",
            params: {
                name,
                arguments: args,
            },
        });
    }

    async stop() {
        this.process.kill("SIGTERM"); // Terminate the process
        console.log("MCP server stopped.");
    }
}
async function getOllamaCompatibleTools(mcp: MCPStdioClient) {
    const list = await mcp.listTools();
    const tools = list.tools.map((tool) => {
        // Extract only the necessary properties for Ollama
        const parameters = {
            type: 'object',
            properties: {},
            required: tool.inputSchema.required || []
        };

        // Copy over only the basic property definitions
        if (tool.inputSchema.properties) {
            for (const [key, prop] of Object.entries(tool.inputSchema.properties)) {
                parameters.properties[key] = {
                    type: prop.type,
                    description: prop.description || undefined
                };
            }
        }

        return {
            type: 'function',
            function: {
                name: tool.name,
                description: tool.description,
                parameters: parameters
            }
        };
    });

    return tools;
}
async function main() {
    const mcp = new MCPStdioClient(config.scriptPath, config.allowedDirectories);
    const ollama = new Ollama({ host: config.host });

    try {
        await mcp.start();
        const tools = await getOllamaCompatibleTools(mcp);
        const chatHistory: { role: string; content: string }[] = [];

        const rl = readline.createInterface({
            input: process.stdin,
            output: process.stdout,
        });

        console.log(`CLI Chat App started! Using model: ${config.model}\n`);

        rl.on("line", async (input) => {
            chatHistory.push({ role: "user", content: input });
            let fullContent = '';

            async function getStreamedResponse() {
                let response = await ollama.chat({
                    model: config.model,
                    messages: chatHistory,
                    tools: tools,
                    stream: true
                });

                process.stdout.write('LLM Response: ');
                fullContent = '';

                for await (const part of response) {
                    const content = part.message.content;
                    process.stdout.write(content);
                    fullContent += content;

                    if (fullContent.includes('```json') && fullContent.includes('```')) {
                        const parts = fullContent.split('```json\n');
                        if (parts.length > 1) {
                            const jsonParts = parts[1].split('```');
                            if (jsonParts.length > 1) {
                                try {
                                    const jsonContent = JSON.parse(jsonParts[0]);
                                    // Only process if it looks like a tool call
                                    if (jsonContent.name && jsonContent.arguments) {
                                        console.log(`\nTool Used: ${jsonContent.name}`, jsonContent.arguments);

                                        const result = await mcp.callTool(jsonContent.name, jsonContent.arguments);

                                        if (!result.isError) {
                                            console.log("tool success");
                                            const toolOutput = result.content.map((c) => c.text).join("\n");
                                            chatHistory.push({
                                                role: "user",
                                                content: `Tool ${jsonContent.name} output:\n${toolOutput}`
                                            });
                                            return await getStreamedResponse();
                                        } else {
                                            console.log("tool error", result.content[0].text);
                                            const errorMsg = result.content[0].text;
                                            chatHistory.push({
                                                role: "user",
                                                content: `Tool ${jsonContent.name} error: ${errorMsg}`
                                            });
                                            return await getStreamedResponse();
                                        }
                                    }
                                    // If it's not a tool call JSON, just continue streaming
                                } catch (error) {
                                    console.error("\nError parsing JSON:", error);
                                }
                            }
                        }
                    }
                }

                process.stdout.write('\n\n');
                return fullContent;
            }

            // Start the conversation
            const finalContent = await getStreamedResponse();
            if (finalContent.trim()) {
                chatHistory.push({ role: "assistant", content: finalContent });
            }
        });

        rl.on("close", () => {
            console.log("Exiting CLI Chat App. Goodbye!");
            mcp.stop();
        });
    } catch (error) {
        console.error(`Error: ${error.message}`);
        mcp.stop();
    }
}

main();