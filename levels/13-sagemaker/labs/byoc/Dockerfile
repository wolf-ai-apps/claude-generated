# Bring-your-own-container: stock vLLM adapted to the SageMaker hosting contract.
# Contract: SageMaker starts the container with the argument "serve", sends GET /ping
# for health and POST /invocations for inference, both on port 8080. Model artifacts
# from S3 (if any) are mounted at /opt/ml/model. vLLM's OpenAI server exposes /ping and
# /invocations itself; this image only fixes the port, host and model location.
ARG VLLM_TAG=v0.11.0
FROM vllm/vllm-openai:${VLLM_TAG}
COPY serve /usr/local/bin/serve
RUN chmod +x /usr/local/bin/serve
ENTRYPOINT ["/usr/local/bin/serve"]
