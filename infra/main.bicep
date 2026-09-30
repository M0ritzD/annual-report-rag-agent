// =============================================================================
// Annual Report RAG Agent – Azure-Infrastruktur
//
// Erstellt:
//   * Storage Account + Blob-Container "reports"  (PDF-Ablage)
//   * Azure AI Search (Free/Basic)                 (Hybrid-Vektorindex)
//   * Azure OpenAI + Deployments (optional)        (Chat + Embeddings)
//   * Log Analytics + Container Apps Environment   (Hosting)
//   * Container App mit System-Managed-Identity    (API + Web-UI)
//   * RBAC-Zuweisungen: keine API-Keys im Container
//
// Deployment:  az deployment group create -g <rg> -f infra/main.bicep -p infra/main.bicepparam
// =============================================================================

targetScope = 'resourceGroup'

@description('Kurzer Präfix für alle Ressourcennamen (nur Kleinbuchstaben/Ziffern).')
@minLength(3)
@maxLength(12)
param prefix string = 'arrag'

param location string = resourceGroup().location

@description('Container-Image, z. B. ghcr.io/<user>/annual-report-rag-agent:latest')
param containerImage string

@allowed(['free', 'basic'])
@description('Free = 0 €, 50 MB, 3 Indizes. Für die drei Demo-Berichte reicht Free.')
param searchSku string = 'free'

@description('Azure OpenAI anlegen? false => Container nutzt Ollama unter ollamaBaseUrl ("Hybrid-Modus").')
param deployOpenAI bool = true

@description('Nur relevant wenn deployOpenAI=false: öffentlich erreichbarer Ollama-Endpunkt (z. B. via Tunnel).')
param ollamaBaseUrl string = ''

param openAILocation string = location
param chatModel string = 'gpt-4o-mini'
param chatModelVersion string = '2024-07-18'
param embeddingModel string = 'text-embedding-3-small'
param embeddingModelVersion string = '1'
param embeddingDimensions int = 1024

var suffix = uniqueString(resourceGroup().id)
var names = {
  storage: toLower('${prefix}st${take(suffix, 8)}')
  search: '${prefix}-search-${take(suffix, 6)}'
  openai: '${prefix}-aoai-${take(suffix, 6)}'
  logs: '${prefix}-logs'
  env: '${prefix}-env'
  app: '${prefix}-app'
}

// Rollen-IDs (integriert)
var roles = {
  blobDataReader: '2a2b9908-6ea1-4ae2-8e65-a410df84e7d1'
  searchIndexDataContributor: '8ebe5a00-799e-43f5-93ac-243d3dce84a7'
  searchServiceContributor: '7ca78c08-252a-4471-8644-bb5ff32d4ba0'
  cognitiveServicesOpenAIUser: '5e0bd9bd-7b93-4f28-af87-19fc36ad61bd'
}

// ----------------------------------------------------------------- Storage
resource storage 'Microsoft.Storage/storageAccounts@2023-05-01' = {
  name: names.storage
  location: location
  sku: { name: 'Standard_LRS' }
  kind: 'StorageV2'
  properties: {
    minimumTlsVersion: 'TLS1_2'
    allowBlobPublicAccess: false
    allowSharedKeyAccess: true // für den einmaligen Upload per Connection String; kann danach deaktiviert werden
  }
}

resource blobService 'Microsoft.Storage/storageAccounts/blobServices@2023-05-01' = {
  parent: storage
  name: 'default'
}

resource reportsContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-05-01' = {
  parent: blobService
  name: 'reports'
}

// ----------------------------------------------------------------- AI Search
resource search 'Microsoft.Search/searchServices@2024-06-01-preview' = {
  name: names.search
  location: location
  sku: { name: searchSku }
  properties: {
    replicaCount: 1
    partitionCount: 1
    // Entra-ID-Zugriff zusätzlich zu API-Keys erlauben
    authOptions: { aadOrApiKey: { aadAuthFailureMode: 'http401WithBearerChallenge' } }
  }
}

// ----------------------------------------------------------------- Azure OpenAI
resource openai 'Microsoft.CognitiveServices/accounts@2024-10-01' = if (deployOpenAI) {
  name: names.openai
  location: openAILocation
  kind: 'OpenAI'
  sku: { name: 'S0' }
  properties: {
    customSubDomainName: names.openai
    publicNetworkAccess: 'Enabled'
  }
}

resource chatDeployment 'Microsoft.CognitiveServices/accounts/deployments@2024-10-01' = if (deployOpenAI) {
  parent: openai
  name: chatModel
  sku: { name: 'GlobalStandard', capacity: 30 }
  properties: {
    model: { format: 'OpenAI', name: chatModel, version: chatModelVersion }
  }
}

resource embeddingDeployment 'Microsoft.CognitiveServices/accounts/deployments@2024-10-01' = if (deployOpenAI) {
  parent: openai
  name: embeddingModel
  sku: { name: 'Standard', capacity: 30 }
  properties: {
    model: { format: 'OpenAI', name: embeddingModel, version: embeddingModelVersion }
  }
  dependsOn: [chatDeployment] // Deployments nacheinander anlegen
}

// ----------------------------------------------------------------- Hosting
resource logs 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: names.logs
  location: location
  properties: {
    sku: { name: 'PerGB2018' }
    retentionInDays: 30
  }
}

resource env 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: names.env
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logs.properties.customerId
        sharedKey: logs.listKeys().primarySharedKey
      }
    }
  }
}

var commonEnv = [
  { name: 'ARR_VECTOR_STORE', value: 'azure_search' }
  { name: 'ARR_DOCUMENT_SOURCE', value: 'azure_blob' }
  { name: 'ARR_AZURE_SEARCH_ENDPOINT', value: 'https://${search.name}.search.windows.net' }
  { name: 'ARR_AZURE_STORAGE_ACCOUNT_URL', value: storage.properties.primaryEndpoints.blob }
  { name: 'ARR_EMBEDDING_DIMENSIONS', value: string(embeddingDimensions) }
]

var openAIEnv = deployOpenAI
  ? [
      { name: 'ARR_LLM_PROVIDER', value: 'azure_openai' }
      { name: 'ARR_EMBEDDING_PROVIDER', value: 'azure_openai' }
      { name: 'ARR_AZURE_OPENAI_ENDPOINT', value: openai!.properties.endpoint }
      { name: 'ARR_AZURE_OPENAI_CHAT_DEPLOYMENT', value: chatModel }
      { name: 'ARR_AZURE_OPENAI_EMBEDDING_DEPLOYMENT', value: embeddingModel }
    ]
  : [
      { name: 'ARR_LLM_PROVIDER', value: 'ollama' }
      { name: 'ARR_EMBEDDING_PROVIDER', value: 'ollama' }
      { name: 'ARR_OLLAMA_BASE_URL', value: ollamaBaseUrl }
    ]

resource app 'Microsoft.App/containerApps@2024-03-01' = {
  name: names.app
  location: location
  identity: { type: 'SystemAssigned' }
  properties: {
    managedEnvironmentId: env.id
    configuration: {
      ingress: {
        external: true
        targetPort: 8000
        transport: 'auto'
      }
    }
    template: {
      containers: [
        {
          name: 'api'
          image: containerImage
          resources: { cpu: json('0.5'), memory: '1Gi' }
          env: concat(commonEnv, openAIEnv)
          probes: [
            {
              type: 'Liveness'
              httpGet: { path: '/api/health', port: 8000 }
              periodSeconds: 30
            }
          ]
        }
      ]
      scale: { minReplicas: 0, maxReplicas: 2 } // scale-to-zero => kaum Kosten im Leerlauf
    }
  }
}

// ----------------------------------------------------------------- RBAC (Managed Identity)
resource blobReader 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: storage
  name: guid(storage.id, app.id, roles.blobDataReader)
  properties: {
    principalId: app.identity.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roles.blobDataReader)
  }
}

resource searchData 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: search
  name: guid(search.id, app.id, roles.searchIndexDataContributor)
  properties: {
    principalId: app.identity.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roles.searchIndexDataContributor)
  }
}

resource searchService 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: search
  name: guid(search.id, app.id, roles.searchServiceContributor)
  properties: {
    principalId: app.identity.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roles.searchServiceContributor)
  }
}

resource openAIUser 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (deployOpenAI) {
  scope: openai
  name: guid(names.openai, app.id, roles.cognitiveServicesOpenAIUser)
  properties: {
    principalId: app.identity.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roles.cognitiveServicesOpenAIUser)
  }
}

// ----------------------------------------------------------------- Outputs
output appUrl string = 'https://${app.properties.configuration.ingress.fqdn}'
output searchEndpoint string = 'https://${search.name}.search.windows.net'
output storageAccountUrl string = storage.properties.primaryEndpoints.blob
output storageAccountName string = storage.name
output openAIEndpoint string = deployOpenAI ? openai!.properties.endpoint : ''
output appPrincipalId string = app.identity.principalId
