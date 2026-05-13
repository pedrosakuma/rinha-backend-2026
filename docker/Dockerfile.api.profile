# syntax=docker/dockerfile:1.7
# Profile image: skips dataprep (uses pre-built local data/), builds API with debug symbols.
# Usage:
#   docker build -f docker/Dockerfile.api.profile -t rinha-api-profile:local .
ARG DOTNET_VERSION=10.0

############################
# 1. Build the .NET API with symbols
############################
FROM mcr.microsoft.com/dotnet/sdk:${DOTNET_VERSION}-noble AS build
RUN sed -i 's|http://archive.ubuntu.com|http://br.archive.ubuntu.com|g; s|http://security.ubuntu.com|http://br.archive.ubuntu.com|g' /etc/apt/sources.list.d/*.sources /etc/apt/sources.list 2>/dev/null || true; \
    apt-get update \
 && apt-get install -y --no-install-recommends clang zlib1g-dev \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY global.json ./
COPY src/Api/Api.csproj src/Api/
RUN dotnet restore src/Api/Api.csproj
COPY src/ ./src/
RUN dotnet publish src/Api/Api.csproj -c Release -o /publish/api --no-restore -r linux-x64 \
    -p:StripSymbols=false

############################
# 2. Data stage — reuse pre-built data from existing local image
#    (avoids re-running the preprocessor and keeps data/ out of build context)
############################
FROM rinha-api-profile:azure AS dataprep

############################
# 3. Final runtime image
############################
FROM mcr.microsoft.com/dotnet/runtime-deps:${DOTNET_VERSION}-noble-chiseled AS runtime
WORKDIR /app
ENV DOTNET_CLI_TELEMETRY_OPTOUT=1 \
    DOTNET_NOLOGO=1 \
    DOTNET_gcServer=0 \
    DOTNET_GCConserveMemory=9 \
    ASPNETCORE_URLS=http://+:9999 \
    PORT=9999 \
    VECTORS_PATH=/data/references.bin \
    LABELS_PATH=/data/labels.bin \
    VECTORS_Q8_PATH=/data/references_q8.bin \
    VECTORS_Q8_SOA_PATH=/data/references_q8_soa.bin \
    VECTORS_Q16_PATH=/data/references_q16.bin \
    VECTORS_Q16_SOA_PATH=/data/references_q16_soa.bin \
    VECTORS_Q16_BLOCKED_PATH=/data/references_q16_blocked.bin \
    IVF_BLOCK_OFFSETS_PATH=/data/ivf_block_offsets.bin \
    IVF_CENTROIDS_PATH=/data/ivf_centroids.bin \
    IVF_OFFSETS_PATH=/data/ivf_offsets.bin \
    IVF_BBOX_MIN_PATH=/data/ivf_bbox_min.bin \
    IVF_BBOX_MAX_PATH=/data/ivf_bbox_max.bin \
    MCC_RISK_PATH=/app/resources/mcc_risk.json \
    NORMALIZATION_PATH=/app/resources/normalization.json
COPY --from=build /publish/api/Rinha.Api /app/Rinha.Api
COPY resources/mcc_risk.json /app/resources/mcc_risk.json
COPY resources/normalization.json /app/resources/normalization.json
COPY resources/profile_fastpath2.json /app/resources/profile_fastpath2.json
COPY resources/selective_decision_tables.json /app/resources/selective_decision_tables.json
COPY --from=dataprep /data/references.bin /data/references.bin
COPY --from=dataprep /data/labels.bin /data/labels.bin
COPY --from=dataprep /data/references_q8.bin /data/references_q8.bin
COPY --from=dataprep /data/references_q8_soa.bin /data/references_q8_soa.bin
COPY --from=dataprep /data/references_q16.bin /data/references_q16.bin
COPY --from=dataprep /data/references_q16_soa.bin /data/references_q16_soa.bin
COPY --from=dataprep /data/references_q16_blocked.bin /data/references_q16_blocked.bin
COPY --from=dataprep /data/ivf_block_offsets.bin /data/ivf_block_offsets.bin
COPY --from=dataprep /data/ivf_centroids.bin /data/ivf_centroids.bin
COPY --from=dataprep /data/ivf_offsets.bin /data/ivf_offsets.bin
COPY --from=dataprep /data/ivf_bbox_min.bin /data/ivf_bbox_min.bin
COPY --from=dataprep /data/ivf_bbox_max.bin /data/ivf_bbox_max.bin
EXPOSE 9999
ENTRYPOINT ["/app/Rinha.Api"]
