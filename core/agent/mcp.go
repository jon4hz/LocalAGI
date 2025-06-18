package agent

import (
	"context"
	"encoding/json"

	"github.com/mark3labs/mcp-go/client"
	"github.com/mark3labs/mcp-go/client/transport"
	"github.com/mark3labs/mcp-go/mcp"
	"github.com/mudler/LocalAGI/core/types"
	"github.com/mudler/LocalAGI/pkg/stdio"
	"github.com/mudler/LocalAGI/pkg/xlog"

	"github.com/sashabaranov/go-openai/jsonschema"
)

var _ types.Action = &mcpAction{}

type MCPServer struct {
	URL   string `json:"url"`
	Token string `json:"token"`
}

type MCPSTDIOServer struct {
	Args []string `json:"args"`
	Env  []string `json:"env"`
	Cmd  string   `json:"cmd"`
}

type mcpAction struct {
	mcpClient       *client.Client
	inputSchema     ToolInputSchema
	toolName        string
	toolDescription string
}

func (a *mcpAction) Plannable() bool {
	return true
}

func (m *mcpAction) Run(ctx context.Context, sharedState *types.AgentSharedState, params types.ActionParams) (types.ActionResult, error) {
	// Convert params to the expected format for the new library
	toolParams := make(map[string]interface{})
	for k, v := range params {
		toolParams[k] = v
	}

	// Create the tool call request
	toolRequest := mcp.CallToolRequest{
		Params: mcp.CallToolParams{
			Name:      m.toolName,
			Arguments: toolParams,
		},
	}

	resp, err := m.mcpClient.CallTool(ctx, toolRequest)
	if err != nil {
		xlog.Error("Failed to call tool", "error", err.Error())
		return types.ActionResult{}, err
	}

	xlog.Debug("MCP response", "response", resp)

	textResult := ""
	for _, c := range resp.Content {
		// Try to cast to TextContent
		if textContent, ok := mcp.AsTextContent(c); ok {
			textResult += textContent.Text + "\n"
		} else {
			xlog.Debug("Non-text content received", "content", c)
		}
	}

	return types.ActionResult{
		Result: textResult,
	}, nil
}

func (m *mcpAction) Definition() types.ActionDefinition {
	props := map[string]jsonschema.Definition{}
	dat, err := json.Marshal(m.inputSchema.Properties)
	if err != nil {
		xlog.Error("Failed to marshal input schema", "error", err.Error())
	}
	json.Unmarshal(dat, &props)

	return types.ActionDefinition{
		Name:        types.ActionDefinitionName(m.toolName),
		Description: m.toolDescription,
		Required:    m.inputSchema.Required,
		//Properties:  ,
		Properties: props,
	}
}

type ToolInputSchema struct {
	Type       string                 `json:"type"`
	Properties map[string]interface{} `json:"properties,omitempty"`
	Required   []string               `json:"required,omitempty"`
}

func (a *Agent) addTools(mcpClient *client.Client) (types.Actions, error) {

	var generatedActions types.Actions
	xlog.Debug("Initializing client")

	// Initialize the client
	initRequest := mcp.InitializeRequest{}
	initRequest.Params.ProtocolVersion = mcp.LATEST_PROTOCOL_VERSION
	initRequest.Params.ClientInfo = mcp.Implementation{
		Name:    "LocalAGI MCP Client",
		Version: "1.0.0",
	}
	initRequest.Params.Capabilities = mcp.ClientCapabilities{}

	response, err := mcpClient.Initialize(a.context, initRequest)
	if err != nil {
		xlog.Error("Failed to initialize client", "error", err.Error())
		return nil, err
	}

	xlog.Debug("Client initialized", "serverInfo", response.ServerInfo)

	// List all tools (the new library handles pagination internally)
	toolsRequest := mcp.ListToolsRequest{}
	toolsResult, err := mcpClient.ListTools(a.context, toolsRequest)
	if err != nil {
		xlog.Error("Failed to list tools", "error", err.Error())
		return nil, err
	}

	for _, t := range toolsResult.Tools {
		desc := ""
		if t.Description != "" {
			desc = t.Description
		}

		xlog.Debug("Tool", "name", t.Name, "description", desc)

		dat, err := json.Marshal(t.InputSchema)
		if err != nil {
			xlog.Error("Failed to marshal input schema", "error", err.Error())
			continue
		}

		xlog.Debug("Input schema", "tool", t.Name, "schema", string(dat))

		// Convert the schema to our internal format
		var inputSchema ToolInputSchema
		err = json.Unmarshal(dat, &inputSchema)
		if err != nil {
			xlog.Error("Failed to unmarshal input schema", "error", err.Error())
			continue
		}

		// Create a new action with Client + tool
		generatedActions = append(generatedActions, &mcpAction{
			mcpClient:       mcpClient,
			toolName:        t.Name,
			inputSchema:     inputSchema,
			toolDescription: desc,
		})
	}

	return generatedActions, nil

}

func (a *Agent) initMCPActions() error {

	a.mcpActions = nil
	var err error

	generatedActions := types.Actions{}

	// MCP HTTP Servers
	for _, mcpServer := range a.options.mcpServers {
		xlog.Debug("Adding tools for MCP HTTP server", "server", mcpServer)

		// Create HTTP transport options
		var options []transport.StreamableHTTPCOption
		if mcpServer.Token != "" {
			headers := map[string]string{
				"Authorization": "Bearer " + mcpServer.Token,
			}
			options = append(options, transport.WithHTTPHeaders(headers))
		}

		// Create the HTTP client using the new library
		mcpClient, err := client.NewStreamableHttpClient(mcpServer.URL, options...)
		if err != nil {
			xlog.Error("Failed to create HTTP client", "server", mcpServer, "error", err.Error())
			continue
		}

		// Start the client
		if err := mcpClient.Start(a.context); err != nil {
			xlog.Error("Failed to start HTTP client", "server", mcpServer, "error", err.Error())
			continue
		}

		actions, err := a.addTools(mcpClient)
		if err != nil {
			xlog.Error("Failed to add tools for MCP server", "server", mcpServer, "error", err.Error())
		}
		generatedActions = append(generatedActions, actions...)
	}

	// MCP STDIO Servers
	a.closeMCPSTDIOServers() // Make sure we stop all previous servers if any is active

	if a.options.mcpPrepareScript != "" {
		xlog.Debug("Preparing MCP box", "script", a.options.mcpPrepareScript)
		client := stdio.NewClient(a.options.mcpBoxURL)
		client.RunProcess(a.context, "/bin/bash", []string{"-c", a.options.mcpPrepareScript}, []string{})
	}

	for _, mcpStdioServer := range a.options.mcpStdioServers {
		xlog.Debug("Adding tools for MCP STDIO server", "server", mcpStdioServer)

		// Create the STDIO client using the new library
		mcpClient, err := client.NewStdioMCPClient(mcpStdioServer.Cmd, mcpStdioServer.Env, mcpStdioServer.Args...)
		if err != nil {
			xlog.Error("Failed to create STDIO client", "server", mcpStdioServer, "error", err.Error())
			continue
		}

		// Start the client
		if err := mcpClient.Start(a.context); err != nil {
			xlog.Error("Failed to start STDIO client", "server", mcpStdioServer, "error", err.Error())
			continue
		}

		actions, err := a.addTools(mcpClient)
		if err != nil {
			xlog.Error("Failed to add tools for MCP server", "server", mcpStdioServer, "error", err.Error())
		}
		generatedActions = append(generatedActions, actions...)
	}

	a.mcpActions = generatedActions

	return err
}

func (a *Agent) closeMCPSTDIOServers() {
	client := stdio.NewClient(a.options.mcpBoxURL)
	client.StopGroup(a.Character.Name)
}
