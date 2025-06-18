package agent

import (
	"context"
	"encoding/json"
	"io"

	"github.com/mark3labs/mcp-go/client"
	"github.com/mark3labs/mcp-go/client/transport"
	"github.com/mark3labs/mcp-go/mcp"
	"github.com/mudler/LocalAGI/core/types"
	"github.com/mudler/LocalAGI/pkg/stdio"
	"github.com/mudler/LocalAGI/pkg/xlog"

	"github.com/sashabaranov/go-openai/jsonschema"
)

// writeCloserWrapper wraps an io.Writer to implement io.WriteCloser
type writeCloserWrapper struct {
	writer io.Writer
}

func (w *writeCloserWrapper) Write(p []byte) (n int, err error) {
	return w.writer.Write(p)
}

func (w *writeCloserWrapper) Close() error {
	// If the underlying writer has a Close method, call it
	if closer, ok := w.writer.(io.Closer); ok {
		return closer.Close()
	}
	// Otherwise, do nothing
	return nil
}

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
	req := mcp.CallToolRequest{}
	req.Params.Name = m.toolName
	req.Params.Arguments = params

	resp, err := m.mcpClient.CallTool(ctx, req)
	if err != nil {
		xlog.Error("Failed to call tool", "tool", m.toolName, "error", err.Error())
		return types.ActionResult{}, err
	}

	xlog.Debug("MCP response", "tool", m.toolName, "response", resp)

	textResult := ""
	for _, c := range resp.Content {
		if textContent, ok := mcp.AsTextContent(c); ok {
			textResult += textContent.Text + "\n"
		} else if _, ok := mcp.AsImageContent(c); ok {
			xlog.Error("Image content not supported yet")
		} else if _, ok := mcp.AsEmbeddedResource(c); ok {
			xlog.Error("Resource content not supported yet")
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
		Properties:  props,
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
	initReq := mcp.InitializeRequest{}
	initReq.Params.ProtocolVersion = mcp.LATEST_PROTOCOL_VERSION
	initReq.Params.ClientInfo = mcp.Implementation{
		Name:    "LocalAGI",
		Version: "1.0.0",
	}
	initReq.Params.Capabilities = mcp.ClientCapabilities{}

	response, err := mcpClient.Initialize(a.context, initReq)
	if err != nil {
		xlog.Error("Failed to initialize client", "error", err.Error())
		return nil, err
	}

	xlog.Debug("Client initialized", "server", response.ServerInfo.Name, "version", response.ServerInfo.Version)

	// List tools
	toolsReq := mcp.ListToolsRequest{}
	toolsResult, err := mcpClient.ListTools(a.context, toolsReq)
	if err != nil {
		xlog.Error("Failed to list tools", "error", err.Error())
		return nil, err
	}

	xlog.Debug("Found tools", "count", len(toolsResult.Tools))

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

		// Convert the input schema
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

	// Close any existing MCP clients first
	a.closeMCPClients()

	var err error

	generatedActions := types.Actions{}

	// MCP HTTP Servers
	for _, mcpServer := range a.options.mcpServers {
		// Create StreamableHTTP client using the convenience method
		var mcpClient *client.Client

		if mcpServer.Token != "" {
			mcpClient, err = client.NewStreamableHttpClient(mcpServer.URL,
				transport.WithHTTPHeaders(map[string]string{
					"Authorization": "Bearer " + mcpServer.Token,
				}))
		} else {
			mcpClient, err = client.NewStreamableHttpClient(mcpServer.URL)
		}

		if err != nil {
			xlog.Error("Failed to create HTTP client", "server", mcpServer, "error", err.Error())
			continue
		}

		// Start the client
		if err := mcpClient.Start(a.context); err != nil {
			xlog.Error("Failed to start HTTP client", "server", mcpServer, "error", err.Error())
			continue
		}

		// Set up notification handler
		mcpClient.OnNotification(func(notification mcp.JSONRPCNotification) {
			xlog.Debug("Received MCP notification", "method", notification.Method, "params", notification.Params)
		})

		xlog.Debug("Adding tools for MCP server", "server", mcpServer)
		actions, err := a.addTools(mcpClient)
		if err != nil {
			xlog.Error("Failed to add tools for MCP server", "server", mcpServer, "error", err.Error())
			// Close the client on error
			mcpClient.Close()
			continue
		}
		generatedActions = append(generatedActions, actions...)
	}

	// MCP STDIO Servers
	a.closeMCPSTDIOServers() // Make sure we stop all previous servers if any is active

	if a.options.mcpPrepareScript != "" {
		xlog.Debug("Preparing MCP box", "script", a.options.mcpPrepareScript)
		stdioClient := stdio.NewClient(a.options.mcpBoxURL)
		stdioClient.RunProcess(a.context, "/bin/bash", []string{"-c", a.options.mcpPrepareScript}, []string{})
	}

	for _, mcpStdioServer := range a.options.mcpStdioServers {
		stdioClient := stdio.NewClient(a.options.mcpBoxURL)
		p, err := stdioClient.CreateProcess(a.context,
			mcpStdioServer.Cmd,
			mcpStdioServer.Args,
			mcpStdioServer.Env,
			a.Character.Name)
		if err != nil {
			xlog.Error("Failed to create process", "error", err.Error())
			continue
		}
		read, writer, err := stdioClient.GetProcessIO(p.ID)
		if err != nil {
			xlog.Error("Failed to get process IO", "error", err.Error())
			continue
		}

		// Create a WriteCloser wrapper for the writer
		writeCloser := &writeCloserWrapper{writer: writer}

		// Create STDIO transport using NewIO
		stdioTransport := transport.NewIO(read, writeCloser, nil)

		// Create a new client using the transport
		mcpClient := client.NewClient(stdioTransport)

		// Start the client
		if err := mcpClient.Start(a.context); err != nil {
			xlog.Error("Failed to start STDIO client", "server", mcpStdioServer, "error", err.Error())
			continue
		}

		// Set up notification handler
		mcpClient.OnNotification(func(notification mcp.JSONRPCNotification) {
			xlog.Debug("Received MCP notification", "method", notification.Method, "params", notification.Params)
		})

		xlog.Debug("Adding tools for MCP server (stdio)", "server", mcpStdioServer)
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
	stdioClient := stdio.NewClient(a.options.mcpBoxURL)
	stdioClient.StopGroup(a.Character.Name)
}

func (a *Agent) closeMCPClients() {
	// Close existing MCP action clients
	if a.mcpActions != nil {
		for _, action := range a.mcpActions {
			if mcpAction, ok := action.(*mcpAction); ok {
				if mcpAction.mcpClient != nil {
					mcpAction.mcpClient.Close()
				}
			}
		}
	}
}
