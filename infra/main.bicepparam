using './main.bicep'

param prefix = 'arrag'
param containerImage = 'ghcr.io/m0ritzd/annual-report-rag-agent:latest'
param searchSku = 'free'
param deployOpenAI = true
param embeddingDimensions = 1024
