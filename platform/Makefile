.PHONY: test build images
test:
	go vet ./... && go test -race ./...
build:
	go build -o bin/ ./cmd/...
images:
	for c in gateway mockllm loadgen; do docker build -t academy/$$c --build-arg CMD=$$c . ; done
