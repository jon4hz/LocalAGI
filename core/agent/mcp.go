package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"sync"

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

// MCPClient represents a cached MCP client with its own context
type MCPClient struct {
	id          string
	client      *client.Client
	initialized bool
	processID   string             // For STDIO servers
	ctx         context.Context    // Dedicated context for this client
	cancelFunc  context.CancelFunc // Cancel function for client context
	mutex       sync.RWMutex
}

func (c *MCPClient) GetClient() *client.Client {
	c.mutex.RLock()
	defer c.mutex.RUnlock()
	return c.client
}

func (c *MCPClient) IsInitialized() bool {
	c.mutex.RLock()
	defer c.mutex.RUnlock()
	return c.initialized
}

func (c *MCPClient) SetInitialized(initialized bool) {
	c.mutex.Lock()
	defer c.mutex.Unlock()
	c.initialized = initialized
}

func (c *MCPClient) GetContext() context.Context {
	c.mutex.RLock()
	defer c.mutex.RUnlock()
	return c.ctx
}

func (c *MCPClient) Close() {
	c.mutex.Lock()
	defer c.mutex.Unlock()
	if c.cancelFunc != nil {
		c.cancelFunc()
	}
}

// MCPClientManager manages persistent MCP clients
type MCPClientManager struct {
	clients map[string]*MCPClient
	mutex   sync.RWMutex
	agent   *Agent
}

func NewMCPClientManager(agent *Agent) *MCPClientManager {
	return &MCPClientManager{
		clients: make(map[string]*MCPClient),
		agent:   agent,
	}
}

func (m *MCPClientManager) GetOrCreateClient(ctx context.Context, id string, serverConfig interface{}) (*MCPClient, error) {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	// Check if client already exists and is valid
	if mcpClient, exists := m.clients[id]; exists {
		if mcpClient.IsInitialized() {
			return mcpClient, nil
		}
		// If client exists but not initialized, clean it up and recreate
		mcpClient.Close()
		delete(m.clients, id)
	}

	// Create dedicated context for this client based on agent context
	clientCtx, cancelFunc := context.WithCancel(m.agent.context)

	// Create new client with dedicated context
	mcpClient := &MCPClient{
		id:         id,
		ctx:        clientCtx,
		cancelFunc: cancelFunc,
	}

	xlog.Debug("Created dedicated context for MCP client", "client", mcpClient.id, "context", fmt.Sprintf("%p", clientCtx))

	var err error
	switch config := serverConfig.(type) {
	case MCPServer:
		err = m.initHTTPClient(mcpClient, config)
	case MCPSTDIOServer:
		err = m.initSTDIOClient(mcpClient, config)
	default:
		mcpClient.Close() // Clean up context if we can't initialize
		return nil, fmt.Errorf("unsupported server config type: %T", serverConfig)
	}

	if err != nil {
		mcpClient.Close() // Clean up context on error
		return nil, fmt.Errorf("failed to initialize client %s: %w", id, err)
	}

	m.clients[id] = mcpClient
	return mcpClient, nil
}

func (m *MCPClientManager) initHTTPClient(mcpClient *MCPClient, config MCPServer) error {
	// Create the HTTP client with headers
	var mcpGOClient *client.Client
	var err error

	if config.Token != "" {
		mcpGOClient, err = client.NewStreamableHttpClient(config.URL,
			transport.WithHTTPHeaders(map[string]string{
				"Authorization": "Bearer " + config.Token,
			}))
	} else {
		mcpGOClient, err = client.NewStreamableHttpClient(config.URL)
	}

	if err != nil {
		return fmt.Errorf("failed to create HTTP client: %w", err)
	}

	// Start the client with its dedicated context
	if err := mcpGOClient.Start(mcpClient.ctx); err != nil {
		return fmt.Errorf("failed to start HTTP client: %w", err)
	}

	mcpClient.client = mcpGOClient
	return m.initializeClient(mcpClient)
}

func (m *MCPClientManager) initSTDIOClient(mcpClient *MCPClient, config MCPSTDIOServer) error {
	stdioClient := stdio.NewClient(m.agent.options.mcpBoxURL)
	p, err := stdioClient.CreateProcess(mcpClient.ctx,
		config.Cmd,
		config.Args,
		config.Env,
		m.agent.Character.Name)
	if err != nil {
		return fmt.Errorf("failed to create process: %w", err)
	}

	mcpClient.processID = p.ID

	read, writer, err := stdioClient.GetProcessIO(p.ID)
	if err != nil {
		return fmt.Errorf("failed to get process IO: %w", err)
	}

	// Create a WriteCloser wrapper for the writer
	writeCloser := &writeCloserWrapper{writer: writer}

	// Create STDIO transport using NewIO
	stdioTransport := transport.NewIO(read, writeCloser, nil)

	// Create a new client using the transport
	mcpGOClient := client.NewClient(stdioTransport)

	// Start the client with its dedicated context
	if err := mcpGOClient.Start(mcpClient.ctx); err != nil {
		return fmt.Errorf("failed to start STDIO client: %w", err)
	}

	mcpClient.client = mcpGOClient
	return m.initializeClient(mcpClient)
}

func (m *MCPClientManager) initializeClient(mcpClient *MCPClient) error {
	// Initialize the client using its dedicated context
	initReq := mcp.InitializeRequest{}
	initReq.Params.ProtocolVersion = mcp.LATEST_PROTOCOL_VERSION
	initReq.Params.ClientInfo = mcp.Implementation{
		Name:    "LocalAGI",
		Version: "1.0.0",
	}
	initReq.Params.Capabilities = mcp.ClientCapabilities{}

	response, err := mcpClient.client.Initialize(mcpClient.ctx, initReq)
	if err != nil {
		return fmt.Errorf("failed to initialize client: %w", err)
	}

	xlog.Debug("Client initialized", "client", mcpClient.id, "server", response.ServerInfo.Name)
	mcpClient.SetInitialized(true)
	return nil
}

func (m *MCPClientManager) CleanupAll() {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	for id, mcpClient := range m.clients {
		// Close the client's dedicated context
		mcpClient.Close()
		delete(m.clients, id)
	}

	// Stop all STDIO processes for this character
	if m.agent.options.mcpBoxURL != "" {
		stdioClient := stdio.NewClient(m.agent.options.mcpBoxURL)
		stdioClient.StopGroup(m.agent.Character.Name)
	}
}

func (m *MCPClientManager) GetClient(id string) (*MCPClient, bool) {
	m.mutex.RLock()
	defer m.mutex.RUnlock()
	mcpClient, exists := m.clients[id]
	return mcpClient, exists
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
	clientManager   *MCPClientManager
	clientID        string
	inputSchema     ToolInputSchema
	toolName        string
	toolDescription string
}

func (a *mcpAction) Plannable() bool {
	return true
}

func (m *mcpAction) Run(ctx context.Context, sharedState *types.AgentSharedState, params types.ActionParams) (types.ActionResult, error) {
	// Get the client
	mcpClient, exists := m.clientManager.GetClient(m.clientID)
	if !exists || !mcpClient.IsInitialized() {
		xlog.Error("Client not found or not initialized", "client", m.clientID, "tool", m.toolName)
		return types.ActionResult{}, fmt.Errorf("client %s not found or not initialized", m.clientID)
	}

	client := mcpClient.GetClient()
	if client == nil {
		xlog.Error("Client not available", "client", m.clientID, "tool", m.toolName)
		return types.ActionResult{}, fmt.Errorf("client not available for client %s", m.clientID)
	}

	// Use the client's dedicated context for the tool call
	clientCtx := mcpClient.GetContext()

	req := mcp.CallToolRequest{}
	req.Params.Name = m.toolName
	req.Params.Arguments = params

	xlog.Debug("Calling MCP tool", "client", m.clientID, "tool", m.toolName, "params", params)

	resp, err := client.CallTool(clientCtx, req)
	if err != nil {
		xlog.Error("Failed to call tool", "error", err.Error(), "client", m.clientID, "tool", m.toolName)
		return types.ActionResult{}, fmt.Errorf("failed to call tool %s: %w", m.toolName, err)
	}

	xlog.Debug("MCP response", "response", resp, "client", m.clientID, "tool", m.toolName)

	textResult := ""
	for _, c := range resp.Content {
		if textContent, ok := mcp.AsTextContent(c); ok {
			textResult += textContent.Text + "\n"
		} else if _, ok := mcp.AsImageContent(c); ok {
			xlog.Warn("Image content not supported yet", "tool", m.toolName)
		} else if _, ok := mcp.AsEmbeddedResource(c); ok {
			xlog.Warn("Resource content not supported yet", "tool", m.toolName)
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
	Type       string         `json:"type"`
	Properties map[string]any `json:"properties,omitempty"`
	Required   []string       `json:"required,omitempty"`
}

func (a *Agent) addTools(clientManager *MCPClientManager, clientID string) (types.Actions, error) {
	var generatedActions types.Actions

	mcpClient, exists := clientManager.GetClient(clientID)
	if !exists || !mcpClient.IsInitialized() {
		return nil, fmt.Errorf("client %s not found or not initialized", clientID)
	}

	client := mcpClient.GetClient()
	if client == nil {
		return nil, fmt.Errorf("client not available for client %s", clientID)
	}

	// Use the client's dedicated context
	clientCtx := mcpClient.GetContext()

	xlog.Debug("Listing tools for client", "client", clientID)

	// List tools using the client's dedicated context
	toolsReq := mcp.ListToolsRequest{}
	toolsResult, err := client.ListTools(clientCtx, toolsReq)
	if err != nil {
		xlog.Error("Failed to list tools", "error", err.Error(), "client", clientID)
		return nil, err
	}

	for _, t := range toolsResult.Tools {
		desc := ""
		if t.Description != "" {
			desc = t.Description
		}

		xlog.Debug("Tool", "name", t.Name, "description", desc, "client", clientID)

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

		// Create a new action with client manager + client ID + tool
		generatedActions = append(generatedActions, &mcpAction{
			clientManager:   clientManager,
			clientID:        clientID,
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

	// Prepare MCP box if needed
	if a.options.mcpPrepareScript != "" {
		xlog.Debug("Preparing MCP box", "script", a.options.mcpPrepareScript)
		stdioClient := stdio.NewClient(a.options.mcpBoxURL)
		stdioClient.RunProcess(a.context, "/bin/bash", []string{"-c", a.options.mcpPrepareScript}, []string{})
	}

	// MCP HTTP Servers
	for i, mcpServer := range a.options.mcpServers {
		clientID := fmt.Sprintf("http-%d-%s", i, mcpServer.URL)

		_, err := a.mcpClientManager.GetOrCreateClient(a.context, clientID, mcpServer)
		if err != nil {
			xlog.Error("Failed to create HTTP client", "server", mcpServer, "error", err.Error())
			continue
		}

		xlog.Debug("Adding tools for MCP HTTP server", "server", mcpServer, "client", clientID)
		actions, err := a.addTools(a.mcpClientManager, clientID)
		if err != nil {
			xlog.Error("Failed to add tools for MCP HTTP server", "server", mcpServer, "error", err.Error())
			continue
		}
		generatedActions = append(generatedActions, actions...)
	}

	// MCP STDIO Servers
	for i, mcpStdioServer := range a.options.mcpStdioServers {
		clientID := fmt.Sprintf("stdio-%d-%s", i, mcpStdioServer.Cmd)

		_, err := a.mcpClientManager.GetOrCreateClient(a.context, clientID, mcpStdioServer)
		if err != nil {
			xlog.Error("Failed to create STDIO client", "server", mcpStdioServer, "error", err.Error())
			continue
		}

		xlog.Debug("Adding tools for MCP STDIO server", "server", mcpStdioServer, "client", clientID)
		actions, err := a.addTools(a.mcpClientManager, clientID)
		if err != nil {
			xlog.Error("Failed to add tools for MCP STDIO server", "server", mcpStdioServer, "error", err.Error())
			continue
		}
		generatedActions = append(generatedActions, actions...)
	}

	a.mcpActions = generatedActions
	return err
}

func (a *Agent) closeMCPSTDIOServers() {
	if a.mcpClientManager != nil {
		a.mcpClientManager.CleanupAll()
	}
}

// RefreshMCPClients refreshes all MCP clients and reinitializes actions
func (a *Agent) RefreshMCPClients() error {
	a.Lock()
	defer a.Unlock()

	xlog.Debug("Refreshing MCP clients", "agent", a.Character.Name)

	// Clean up existing clients
	if a.mcpClientManager != nil {
		a.mcpClientManager.CleanupAll()
	}

	// Reinitialize MCP actions which will create new clients
	return a.initMCPActions()
}

// GetActiveMCPClients returns information about active MCP clients
func (a *Agent) GetActiveMCPClients() map[string]bool {
	if a.mcpClientManager == nil {
		return make(map[string]bool)
	}

	a.mcpClientManager.mutex.RLock()
	defer a.mcpClientManager.mutex.RUnlock()

	result := make(map[string]bool)
	for id, mcpClient := range a.mcpClientManager.clients {
		result[id] = mcpClient.IsInitialized()
	}

	return result
}
