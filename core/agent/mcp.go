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

// MCPSession represents a persistent MCP client session
type MCPSession struct {
	id           string
	client       *client.Client
	initialized  bool
	serverConfig any    // Can be MCPServer or MCPSTDIOServer
	sessionType  string // "http" or "stdio"
	processID    string // For STDIO servers
	mutex        sync.RWMutex
}

func (s *MCPSession) SessionID() string {
	return s.id
}

func (s *MCPSession) IsInitialized() bool {
	s.mutex.RLock()
	defer s.mutex.RUnlock()
	return s.initialized
}

func (s *MCPSession) SetInitialized(initialized bool) {
	s.mutex.Lock()
	defer s.mutex.Unlock()
	s.initialized = initialized
}

func (s *MCPSession) GetClient() *client.Client {
	s.mutex.RLock()
	defer s.mutex.RUnlock()
	return s.client
}

// MCPSessionManager manages persistent MCP client sessions
type MCPSessionManager struct {
	sessions map[string]*MCPSession
	mutex    sync.RWMutex
	agent    *Agent
}

func NewMCPSessionManager(agent *Agent) *MCPSessionManager {
	return &MCPSessionManager{
		sessions: make(map[string]*MCPSession),
		agent:    agent,
	}
}

func (m *MCPSessionManager) GetOrCreateSession(ctx context.Context, id string, serverConfig any) (*MCPSession, error) {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	// Check if session already exists and is valid
	if session, exists := m.sessions[id]; exists {
		if session.IsInitialized() {
			return session, nil
		}
		// If session exists but not initialized, remove it and recreate
		m.removeSessionUnsafe(id)
	}

	// Create new session
	session := &MCPSession{
		id:           id,
		serverConfig: serverConfig,
	}

	var err error
	switch config := serverConfig.(type) {
	case MCPServer:
		session.sessionType = "http"
		err = m.initHTTPSession(ctx, session, config)
	case MCPSTDIOServer:
		session.sessionType = "stdio"
		err = m.initSTDIOSession(ctx, session, config)
	default:
		return nil, fmt.Errorf("unsupported server config type: %T", serverConfig)
	}

	if err != nil {
		return nil, fmt.Errorf("failed to initialize session %s: %w", id, err)
	}

	m.sessions[id] = session
	return session, nil
}

func (m *MCPSessionManager) initHTTPSession(ctx context.Context, session *MCPSession, config MCPServer) error {
	var mcpClient *client.Client
	var err error

	if config.Token != "" {
		mcpClient, err = client.NewStreamableHttpClient(config.URL,
			transport.WithHTTPHeaders(map[string]string{
				"Authorization": "Bearer " + config.Token,
			}))
	} else {
		mcpClient, err = client.NewStreamableHttpClient(config.URL)
	}

	if err != nil {
		return fmt.Errorf("failed to create HTTP client: %w", err)
	}

	// Start the client
	if err := mcpClient.Start(ctx); err != nil {
		return fmt.Errorf("failed to start HTTP client: %w", err)
	}

	session.client = mcpClient
	return m.initializeClient(ctx, session)
}

func (m *MCPSessionManager) initSTDIOSession(ctx context.Context, session *MCPSession, config MCPSTDIOServer) error {
	stdioClient := stdio.NewClient(m.agent.options.mcpBoxURL)
	p, err := stdioClient.CreateProcess(ctx,
		config.Cmd,
		config.Args,
		config.Env,
		m.agent.Character.Name)
	if err != nil {
		return fmt.Errorf("failed to create process: %w", err)
	}

	session.processID = p.ID

	read, writer, err := stdioClient.GetProcessIO(p.ID)
	if err != nil {
		return fmt.Errorf("failed to get process IO: %w", err)
	}

	// Create a WriteCloser wrapper for the writer
	writeCloser := &writeCloserWrapper{writer: writer}

	// Create STDIO transport using NewIO
	stdioTransport := transport.NewIO(read, writeCloser, nil)

	// Create a new client using the transport
	mcpClient := client.NewClient(stdioTransport)

	// Start the client
	if err := mcpClient.Start(ctx); err != nil {
		return fmt.Errorf("failed to start STDIO client: %w", err)
	}

	session.client = mcpClient
	return m.initializeClient(ctx, session)
}

func (m *MCPSessionManager) initializeClient(ctx context.Context, session *MCPSession) error {
	// Initialize the client
	initReq := mcp.InitializeRequest{}
	initReq.Params.ProtocolVersion = mcp.LATEST_PROTOCOL_VERSION
	initReq.Params.ClientInfo = mcp.Implementation{
		Name:    "LocalAGI",
		Version: "1.0.0",
	}
	initReq.Params.Capabilities = mcp.ClientCapabilities{}

	response, err := session.client.Initialize(ctx, initReq)
	if err != nil {
		return fmt.Errorf("failed to initialize client: %w", err)
	}

	xlog.Debug("Client initialized", "session", session.id, "server", response.ServerInfo.Name)
	session.SetInitialized(true)
	return nil
}

func (m *MCPSessionManager) RemoveSession(id string) {
	m.mutex.Lock()
	defer m.mutex.Unlock()
	m.removeSessionUnsafe(id)
}

func (m *MCPSessionManager) removeSessionUnsafe(id string) {
	if session, exists := m.sessions[id]; exists {
		// Clean up STDIO process if needed
		if session.sessionType == "stdio" && session.processID != "" {
			// STDIO processes will be cleaned up by the group cleanup in CleanupAll
		}
		delete(m.sessions, id)
	}
}

func (m *MCPSessionManager) CleanupAll() {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	for id := range m.sessions {
		m.removeSessionUnsafe(id)
	}

	// Stop all STDIO processes for this character
	if m.agent.options.mcpBoxURL != "" {
		stdioClient := stdio.NewClient(m.agent.options.mcpBoxURL)
		stdioClient.StopGroup(m.agent.Character.Name)
	}
}

func (m *MCPSessionManager) GetSession(id string) (*MCPSession, bool) {
	m.mutex.RLock()
	defer m.mutex.RUnlock()
	session, exists := m.sessions[id]
	return session, exists
}

// CheckSessionHealth verifies if a session is still healthy and responsive
func (m *MCPSessionManager) CheckSessionHealth(ctx context.Context, sessionID string) error {
	session, exists := m.GetSession(sessionID)
	if !exists {
		return fmt.Errorf("session %s not found", sessionID)
	}

	if !session.IsInitialized() {
		return fmt.Errorf("session %s not initialized", sessionID)
	}

	client := session.GetClient()
	if client == nil {
		return fmt.Errorf("session %s has no client", sessionID)
	}

	// Try to list tools as a health check
	toolsReq := mcp.ListToolsRequest{}
	_, err := client.ListTools(ctx, toolsReq)
	if err != nil {
		return fmt.Errorf("session %s health check failed: %w", sessionID, err)
	}

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
	sessionManager  *MCPSessionManager
	sessionID       string
	inputSchema     ToolInputSchema
	toolName        string
	toolDescription string
}

func (a *mcpAction) Plannable() bool {
	return true
}

func (m *mcpAction) Run(ctx context.Context, sharedState *types.AgentSharedState, params types.ActionParams) (types.ActionResult, error) {
	// Get the session
	session, exists := m.sessionManager.GetSession(m.sessionID)
	if !exists || !session.IsInitialized() {
		xlog.Error("Session not found or not initialized", "session", m.sessionID, "tool", m.toolName)
		return types.ActionResult{}, fmt.Errorf("session %s not found or not initialized", m.sessionID)
	}

	client := session.GetClient()
	if client == nil {
		xlog.Error("Client not available", "session", m.sessionID, "tool", m.toolName)
		return types.ActionResult{}, fmt.Errorf("client not available for session %s", m.sessionID)
	}

	req := mcp.CallToolRequest{}
	req.Params.Name = m.toolName
	req.Params.Arguments = params

	xlog.Debug("Calling MCP tool", "session", m.sessionID, "tool", m.toolName, "params", params)

	resp, err := client.CallTool(ctx, req)
	if err != nil {
		xlog.Error("Failed to call tool", "error", err.Error(), "session", m.sessionID, "tool", m.toolName)

		// Check if this might be a session issue and try to recreate the session
		if healthErr := m.sessionManager.CheckSessionHealth(ctx, m.sessionID); healthErr != nil {
			xlog.Warn("Session health check failed, session may need refresh", "session", m.sessionID, "health_error", healthErr.Error())
		}

		return types.ActionResult{}, fmt.Errorf("failed to call tool %s: %w", m.toolName, err)
	}

	xlog.Debug("MCP response", "response", resp, "session", m.sessionID, "tool", m.toolName)

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

func (a *Agent) addTools(sessionManager *MCPSessionManager, sessionID string) (types.Actions, error) {
	var generatedActions types.Actions

	session, exists := sessionManager.GetSession(sessionID)
	if !exists || !session.IsInitialized() {
		return nil, fmt.Errorf("session %s not found or not initialized", sessionID)
	}

	mcpClient := session.GetClient()
	if mcpClient == nil {
		return nil, fmt.Errorf("client not available for session %s", sessionID)
	}

	xlog.Debug("Listing tools for session", "session", sessionID)

	// List tools
	toolsReq := mcp.ListToolsRequest{}
	toolsResult, err := mcpClient.ListTools(a.context, toolsReq)
	if err != nil {
		xlog.Error("Failed to list tools", "error", err.Error(), "session", sessionID)
		return nil, err
	}

	for _, t := range toolsResult.Tools {
		desc := ""
		if t.Description != "" {
			desc = t.Description
		}

		xlog.Debug("Tool", "name", t.Name, "description", desc, "session", sessionID)

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

		// Create a new action with session manager + session ID + tool
		generatedActions = append(generatedActions, &mcpAction{
			sessionManager:  sessionManager,
			sessionID:       sessionID,
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
		sessionID := fmt.Sprintf("http-%d-%s", i, mcpServer.URL)

		_, err := a.mcpSessionManager.GetOrCreateSession(a.context, sessionID, mcpServer)
		if err != nil {
			xlog.Error("Failed to create HTTP session", "server", mcpServer, "error", err.Error())
			continue
		}

		xlog.Debug("Adding tools for MCP HTTP server", "server", mcpServer, "session", sessionID)
		actions, err := a.addTools(a.mcpSessionManager, sessionID)
		if err != nil {
			xlog.Error("Failed to add tools for MCP HTTP server", "server", mcpServer, "error", err.Error())
			continue
		}
		generatedActions = append(generatedActions, actions...)
	}

	// MCP STDIO Servers
	for i, mcpStdioServer := range a.options.mcpStdioServers {
		sessionID := fmt.Sprintf("stdio-%d-%s", i, mcpStdioServer.Cmd)

		_, err := a.mcpSessionManager.GetOrCreateSession(a.context, sessionID, mcpStdioServer)
		if err != nil {
			xlog.Error("Failed to create STDIO session", "server", mcpStdioServer, "error", err.Error())
			continue
		}

		xlog.Debug("Adding tools for MCP STDIO server", "server", mcpStdioServer, "session", sessionID)
		actions, err := a.addTools(a.mcpSessionManager, sessionID)
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
	if a.mcpSessionManager != nil {
		a.mcpSessionManager.CleanupAll()
	}
}

// RefreshMCPSessions refreshes all MCP sessions and reinitializes actions
func (a *Agent) RefreshMCPSessions() error {
	a.Lock()
	defer a.Unlock()

	xlog.Debug("Refreshing MCP sessions", "agent", a.Character.Name)

	// Clean up existing sessions
	if a.mcpSessionManager != nil {
		a.mcpSessionManager.CleanupAll()
	}

	// Reinitialize MCP actions which will create new sessions
	return a.initMCPActions()
}

// GetActiveMCPSessions returns information about active MCP sessions
func (a *Agent) GetActiveMCPSessions() map[string]bool {
	if a.mcpSessionManager == nil {
		return make(map[string]bool)
	}

	a.mcpSessionManager.mutex.RLock()
	defer a.mcpSessionManager.mutex.RUnlock()

	result := make(map[string]bool)
	for id, session := range a.mcpSessionManager.sessions {
		result[id] = session.IsInitialized()
	}

	return result
}
