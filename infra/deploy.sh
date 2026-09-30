#!/usr/bin/env bash
# Deployt die komplette Azure-Umgebung und befüllt den Index.
#
#   ./infra/deploy.sh <resource-group> [location]
#
# Voraussetzungen: az CLI (angemeldet), Rechte zum Anlegen von Rollenzuweisungen
# (Owner oder User Access Administrator auf der Resource Group).
set -euo pipefail

RG="${1:?Resource Group angeben}"
LOCATION="${2:-swedencentral}"   # Azure OpenAI + AI Search + Container Apps verfügbar

cd "$(dirname "$0")/.."

echo "▶ Resource Group $RG ($LOCATION)"
az group create -n "$RG" -l "$LOCATION" -o none

echo "▶ Bicep-Deployment (dauert ~5–8 Min.)"
OUT=$(az deployment group create -g "$RG" -f infra/main.bicep -p infra/main.bicepparam \
  --query properties.outputs -o json)

get() { echo "$OUT" | python3 -c "import sys,json; print(json.load(sys.stdin)['$1']['value'])"; }
APP_URL=$(get appUrl)
SEARCH=$(get searchEndpoint)
STORAGE_URL=$(get storageAccountUrl)
STORAGE_NAME=$(get storageAccountName)
AOAI=$(get openAIEndpoint)

echo "▶ Eigene Rechte für Upload + Indexierung vom Laptop aus vergeben"
ME=$(az ad signed-in-user show --query id -o tsv)
SCOPE_ST=$(az storage account show -n "$STORAGE_NAME" -g "$RG" --query id -o tsv)
SCOPE_SEARCH=$(az search service show -n "${SEARCH#https://}" -g "$RG" --query id -o tsv 2>/dev/null || true)
az role assignment create --assignee "$ME" --role "Storage Blob Data Contributor" --scope "$SCOPE_ST" -o none || true
[ -n "$SCOPE_SEARCH" ] && az role assignment create --assignee "$ME" --role "Search Index Data Contributor" --scope "$SCOPE_SEARCH" -o none || true
[ -n "$SCOPE_SEARCH" ] && az role assignment create --assignee "$ME" --role "Search Service Contributor" --scope "$SCOPE_SEARCH" -o none || true

cat > .env.azure <<EOF
ARR_LLM_PROVIDER=azure_openai
ARR_EMBEDDING_PROVIDER=azure_openai
ARR_VECTOR_STORE=azure_search
ARR_DOCUMENT_SOURCE=azure_blob
ARR_AZURE_OPENAI_ENDPOINT=$AOAI
ARR_AZURE_SEARCH_ENDPOINT=$SEARCH
ARR_AZURE_STORAGE_ACCOUNT_URL=$STORAGE_URL
ARR_EMBEDDING_DIMENSIONS=1024
EOF

echo "▶ Warte 60 s auf RBAC-Propagierung"; sleep 60
echo "▶ PDFs hochladen + Index in Azure AI Search aufbauen"
set -a; source .env.azure; set +a
uv run arr upload
uv run arr index

echo "✅ Fertig: $APP_URL"
