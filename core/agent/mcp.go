package agent

import (
	"context"
	"encoding/json"
	"io"
	"strings"
	"time"

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
	// Check client health before making the call
	if err := m.checkMCPClientHealth(ctx); err != nil {
		xlog.Error("MCP client health check failed", "tool", m.toolName, "error", err.Error())
		return types.ActionResult{}, err
	}

	req := mcp.CallToolRequest{
		Params: mcp.CallToolParams{
			Name:      m.toolName,
			Arguments: params,
		},
	}

	const maxRetries = 3
	var resp *mcp.CallToolResult
	var err error

	for attempt := 0; attempt < maxRetries; attempt++ {
		resp, err = m.mcpClient.CallTool(ctx, req)
		if err == nil {
			break
		}
		
		// Check if this is a session terminated error that can be retried
		if strings.Contains(err.Error(), "session terminated") || strings.Contains(err.Error(), "404") {
			xlog.Warn("MCP tool call failed, retrying", "tool", m.toolName, "attempt", attempt+1, "error", err)
			if attempt < maxRetries-1 {
				time.Sleep(time.Duration(attempt+1) * time.Second)
			}
		} else {
			// For other errors, don't retry
			break
		}
	}

	if err != nil {
		xlog.Error("Failed to call tool after retries", "tool", m.toolName, "error", err.Error())
		return types.ActionResult{}, err
	}

	xlog.Debug("MCP response", "response", resp)

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

	// Initialize the client with retry logic
	initReq := mcp.InitializeRequest{
		Params: mcp.InitializeParams{
			ProtocolVersion: mcp.LATEST_PROTOCOL_VERSION,
			ClientInfo: mcp.Implementation{
				Name:    "LocalAGI",
				Version: "1.0.0",
			},
			Capabilities: mcp.ClientCapabilities{},
		},
	}

	const maxRetries = 3
	var response *mcp.InitializeResult
	var err error

	for attempt := 0; attempt < maxRetries; attempt++ {
		response, err = mcpClient.Initialize(a.context, initReq)
		if err == nil {
			break
		}
		
		// Check if this is a session terminated error that can be retried
		if strings.Contains(err.Error(), "session terminated") || strings.Contains(err.Error(), "404") {
			xlog.Warn("MCP client initialization failed, retrying", "attempt", attempt+1, "error", err)
			if attempt < maxRetries-1 {
				time.Sleep(time.Duration(attempt+1) * time.Second) // Exponential backoff
			}
		} else {
			// For other errors, don't retry
			break
		}
	}

	if err != nil {
		xlog.Error("Failed to initialize client after retries", "error", err.Error())
		return nil, err
	}

	xlog.Debug("Client initialized", "server", response.ServerInfo.Name)

	// List tools
	toolsReq := mcp.ListToolsRequest{}
	toolsResult, err := mcpClient.ListTools(a.context, toolsReq)
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
		}

		xlog.Debug("Input schema", "tool", t.Name, "schema", string(dat))

		// Convert the input schema
		var inputSchema ToolInputSchema
		err = json.Unmarshal(dat, &inputSchema)
		if err != nil {
			xlog.Error("Failed to unmarshal input schema", "error", err.Error())
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
		// Create StreamableHTTP transport
		httpTransport, err := transport.NewStreamableHTTP(mcpServer.URL)
		if err != nil {
			xlog.Error("Failed to create HTTP transport", "server", mcpServer, "error", err.Error())
			continue
		}

		if mcpServer.Token != "" {
			httpTransport, err = transport.NewStreamableHTTP(mcpServer.URL,
				transport.WithHTTPHeaders(map[string]string{
					"Authorization": "Bearer " + mcpServer.Token,
				}))
			if err != nil {
				xlog.Error("Failed to create HTTP transport with auth", "server", mcpServer, "error", err.Error())
				continue
			}
		}

		// Create a new client
		mcpClient := client.NewClient(httpTransport)

		// Start the client with retry logic
		const maxStartRetries = 3
		var startErr error
		for attempt := 0; attempt < maxStartRetries; attempt++ {
			startErr = mcpClient.Start(a.context)
			if startErr == nil {
				break
			}
			
			if strings.Contains(startErr.Error(), "session terminated") || strings.Contains(startErr.Error(), "404") {
				xlog.Warn("MCP HTTP client start failed, retrying", "attempt", attempt+1, "server", mcpServer, "error", startErr)
				if attempt < maxStartRetries-1 {
					time.Sleep(time.Duration(attempt+1) * time.Second)
				}
			} else {
				break
			}
		}

		if startErr != nil {
			xlog.Error("Failed to start HTTP client after retries", "server", mcpServer, "error", startErr.Error())
			continue
		}

		xlog.Debug("Adding tools for MCP server", "server", mcpServer)
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

		// Create a new client
		mcpClient := client.NewClient(stdioTransport)

		// Start the client with retry logic
		const maxStartRetries = 3
		var startErr error
		for attempt := 0; attempt < maxStartRetries; attempt++ {
			startErr = mcpClient.Start(a.context)
			if startErr == nil {
				break
			}
			
			if strings.Contains(startErr.Error(), "session terminated") || strings.Contains(startErr.Error(), "404") {
				xlog.Warn("MCP STDIO client start failed, retrying", "attempt", attempt+1, "server", mcpStdioServer, "error", startErr)
				if attempt < maxStartRetries-1 {
					time.Sleep(time.Duration(attempt+1) * time.Second)
				}
			} else {
				break
			}
		}

		if startErr != nil {
			xlog.Error("Failed to start STDIO client after retries", "server", mcpStdioServer, "error", startErr.Error())
			continue
		}

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

// checkMCPClientHealth checks if the MCP client connection is still healthy
// and attempts to reinitialize if needed
func (m *mcpAction) checkMCPClientHealth(ctx context.Context) error {
	// Try a simple ListTools call to check if the connection is still active
	toolsReq := mcp.ListToolsRequest{}
	_, err := m.mcpClient.ListTools(ctx, toolsReq)
	
	if err != nil && (strings.Contains(err.Error(), "session terminated") || strings.Contains(err.Error(), "404")) {
		xlog.Warn("MCP client connection unhealthy, reinitializing", "error", err)
		
		// Attempt to reinitialize the client
		initReq := mcp.InitializeRequest{
			Params: mcp.InitializeParams{
				ProtocolVersion: mcp.LATEST_PROTOCOL_VERSION,
				ClientInfo: mcp.Implementation{
					Name:    "LocalAGI",
					Version: "1.0.0",
				},
				Capabilities: mcp.ClientCapabilities{},
			},
		}
		
		const maxRetries = 3
		for attempt := 0; attempt < maxRetries; attempt++ {
			_, reinitErr := m.mcpClient.Initialize(ctx, initReq)
			if reinitErr == nil {
				xlog.Info("MCP client successfully reinitialized")
				return nil
			}
			
			if strings.Contains(reinitErr.Error(), "session terminated") || strings.Contains(reinitErr.Error(), "404") {
				xlog.Warn("MCP client reinitialize failed, retrying", "attempt", attempt+1, "error", reinitErr)
				if attempt < maxRetries-1 {
					time.Sleep(time.Duration(attempt+1) * time.Second)
				}
			} else {
				return reinitErr
			}
		}
		
		return err
	}
	
	return err
}
