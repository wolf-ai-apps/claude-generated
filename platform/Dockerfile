# Builds any of the academy's Go tools into a tiny static image.
#   docker build -t academy/gateway --build-arg CMD=gateway platform/
#   docker build -t academy/mockllm --build-arg CMD=mockllm platform/
FROM golang:1.24 AS build
ARG CMD=gateway
WORKDIR /src
COPY go.mod ./
COPY . .
RUN CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/app ./cmd/${CMD}

FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=build /out/app /app
USER nonroot
ENTRYPOINT ["/app"]
