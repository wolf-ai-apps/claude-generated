// Package openai holds the subset of the OpenAI-compatible API that vLLM,
// SGLang and most inference servers expose, plus a Server-Sent Events reader.
package openai

import (
	"bufio"
	"bytes"
	"io"
	"strings"
)

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type StreamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

// ChatRequest is the body of POST /v1/chat/completions.
type ChatRequest struct {
	Model         string         `json:"model"`
	Messages      []Message      `json:"messages"`
	MaxTokens     int            `json:"max_tokens,omitempty"`
	Temperature   *float64       `json:"temperature,omitempty"`
	Stream        bool           `json:"stream,omitempty"`
	StreamOptions *StreamOptions `json:"stream_options,omitempty"`
	// IgnoreEOS is a vLLM extension: keep generating until max_tokens.
	// Benchmarks set it so output length is controlled.
	IgnoreEOS bool `json:"ignore_eos,omitempty"`
}

// PromptText concatenates message contents; used for token estimates and routing keys.
func (r *ChatRequest) PromptText() string {
	var b strings.Builder
	for _, m := range r.Messages {
		b.WriteString(m.Role)
		b.WriteString(": ")
		b.WriteString(m.Content)
		b.WriteString("\n")
	}
	return b.String()
}

type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type Delta struct {
	Role    string `json:"role,omitempty"`
	Content string `json:"content,omitempty"`
}

type ChunkChoice struct {
	Index        int     `json:"index"`
	Delta        Delta   `json:"delta"`
	FinishReason *string `json:"finish_reason"`
}

// ChatChunk is one streamed event (object = "chat.completion.chunk").
type ChatChunk struct {
	ID      string        `json:"id"`
	Object  string        `json:"object"`
	Created int64         `json:"created"`
	Model   string        `json:"model"`
	Choices []ChunkChoice `json:"choices"`
	Usage   *Usage        `json:"usage,omitempty"`
}

type Choice struct {
	Index        int     `json:"index"`
	Message      Message `json:"message"`
	FinishReason string  `json:"finish_reason"`
}

// ChatResponse is the non-streaming response.
type ChatResponse struct {
	ID      string   `json:"id"`
	Object  string   `json:"object"`
	Created int64    `json:"created"`
	Model   string   `json:"model"`
	Choices []Choice `json:"choices"`
	Usage   Usage    `json:"usage"`
}

// EstimateTokens is the classic ~4 characters per token heuristic for English text.
// Real systems call the model's tokenizer; this is good enough for budgets and routing.
func EstimateTokens(s string) int {
	n := (len(s) + 3) / 4
	if n == 0 && len(s) > 0 {
		n = 1
	}
	return n
}

// SSEReader yields the data payload of each Server-Sent Event.
type SSEReader struct {
	s *bufio.Scanner
}

func NewSSEReader(r io.Reader) *SSEReader {
	s := bufio.NewScanner(r)
	s.Buffer(make([]byte, 64*1024), 4*1024*1024)
	return &SSEReader{s: s}
}

// Next returns the next event's data. It returns io.EOF at end of stream.
// Multi-line data fields are joined with "\n" as the SSE spec requires.
func (r *SSEReader) Next() ([]byte, error) {
	var data [][]byte
	for r.s.Scan() {
		line := r.s.Bytes()
		if len(line) == 0 {
			if len(data) > 0 {
				return bytes.Join(data, []byte("\n")), nil
			}
			continue
		}
		if bytes.HasPrefix(line, []byte(":")) { // comment / keep-alive
			continue
		}
		if v, ok := bytes.CutPrefix(line, []byte("data:")); ok {
			data = append(data, bytes.Clone(bytes.TrimPrefix(v, []byte(" "))))
		}
	}
	if err := r.s.Err(); err != nil {
		return nil, err
	}
	if len(data) > 0 {
		return bytes.Join(data, []byte("\n")), nil
	}
	return nil, io.EOF
}
