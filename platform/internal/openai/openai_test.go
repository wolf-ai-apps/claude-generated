package openai

import (
	"io"
	"strings"
	"testing"
)

func TestSSEReader(t *testing.T) {
	in := ": keep-alive\n\ndata: {\"a\":1}\n\ndata: line1\ndata: line2\n\ndata: [DONE]\n\n"
	r := NewSSEReader(strings.NewReader(in))
	want := []string{`{"a":1}`, "line1\nline2", "[DONE]"}
	for _, w := range want {
		got, err := r.Next()
		if err != nil {
			t.Fatal(err)
		}
		if string(got) != w {
			t.Fatalf("got %q want %q", got, w)
		}
	}
	if _, err := r.Next(); err != io.EOF {
		t.Fatalf("want EOF, got %v", err)
	}
}

func TestEstimateTokens(t *testing.T) {
	cases := map[string]int{"": 0, "a": 1, "abcd": 1, "abcde": 2, strings.Repeat("x", 400): 100}
	for s, want := range cases {
		if got := EstimateTokens(s); got != want {
			t.Errorf("EstimateTokens(%d chars) = %d want %d", len(s), got, want)
		}
	}
}
