# QRATUM TypeScript SDK

TypeScript/JavaScript client SDK for the QRATUM Platform API.

## Installation

```bash
npm install @qratum/sdk
```

## Quick Start

```typescript
import { QRATUMClient } from '@qratum/sdk';

// Create client with OAuth2 credentials
const client = new QRATUMClient({
  baseUrl: 'https://api.qratum.io/v1',
  clientId: 'your-client-id',
  clientSecret: 'your-client-secret',
});

// Or with API key
const clientWithApiKey = new QRATUMClient({
  baseUrl: 'https://api.qratum.io/v1',
  apiKey: 'qratum_your_api_key',
});

// Authenticate (required for OAuth2, optional for API key)
await client.authenticate();

// Submit a VQE job
const job = await client.jobs.submit({
  jobType: 'vqe',
  name: 'H2 Ground State',
  config: {
    molecule: 'H2',
    bond_length: 0.735,
    basis: 'sto-3g',
    max_iterations: 100,
  },
});

console.log(`Job submitted: ${job.jobId}`);
```

## API Reference

### Authentication

The SDK supports three authentication methods:

1. **OAuth2 Client Credentials** (recommended for applications)
2. **API Key** (for service-to-service communication)
3. **Pre-existing Access Token** (for custom auth flows)

```typescript
// OAuth2 Client Credentials
const client = new QRATUMClient({
  clientId: 'your-client-id',
  clientSecret: 'your-client-secret',
});
await client.authenticate();

// API Key
const client = new QRATUMClient({
  apiKey: 'qratum_your_api_key',
});

// Pre-existing Token
const client = new QRATUMClient({
  accessToken: 'your-jwt-token',
});
```

### Jobs API

```typescript
// Submit a job
const job = await client.jobs.submit({
  jobType: 'quantum_circuit',
  config: { circuit: [...], shots: 1024 },
  priority: 7,
  tags: ['production', 'benchmark'],
});

// List jobs
const { jobs, total } = await client.jobs.list({
  status: 'running',
  limit: 10,
});

// Get job details
const details = await client.jobs.get(jobId);

// Cancel a job
await client.jobs.cancel(jobId);

// Validate configuration
const { valid, errors, warnings } = await client.jobs.validate({
  jobType: 'vqe',
  config: { molecule: 'H2' },
});
```

### Status API (with WebSocket)

```typescript
// Get status via HTTP
const status = await client.status.get(jobId);
console.log(`Status: ${status.status}, Progress: ${status.progress * 100}%`);

// Subscribe to real-time updates
const unsubscribe = client.status.subscribe(jobId, (status) => {
  console.log(`Job ${status.jobId}: ${status.status} (${status.progress * 100}%)`);
  
  if (status.status === 'completed') {
    unsubscribe();
  }
});

// Disconnect WebSocket when done
client.disconnect();
```

### Results API

```typescript
// Get results
const results = await client.results.get(jobId);
console.log('Energy:', results.results.energy);
console.log('Fidelity:', results.metrics?.fidelity);

// List artifacts
const { artifacts } = await client.results.listArtifacts(jobId);
for (const artifact of artifacts) {
  console.log(`${artifact.filename} (${artifact.sizeBytes} bytes)`);
}

// Get visualization data
const viz = await client.results.getVisualization(jobId);
```

### Resources API

```typescript
// Get dashboard
const dashboard = await client.resources.getDashboard();
console.log(`Queue depth: ${dashboard.queueDepth}`);
console.log(`GPU utilization: ${dashboard.utilization.gpu * 100}%`);

// List clusters
const clusters = await client.resources.listClusters();
for (const cluster of clusters) {
  console.log(`${cluster.name} (${cluster.provider}): ${cluster.gpusAvailable}/${cluster.gpusTotal} GPUs`);
}

// Get quotas
const { quotas, usage } = await client.resources.getQuotas();
console.log(`GPU hours: ${usage.gpuHoursThisMonth}/${quotas.maxGpuHoursPerMonth}`);
```

## Supported Job Types

| Job Type | Description |
|----------|-------------|
| `quantum_circuit` | Quantum circuit simulation |
| `vqe` | Variational Quantum Eigensolver |
| `qaoa` | Quantum Approximate Optimization Algorithm |
| `tensor_analysis` | Tensor network analysis |
| `digital_twin` | Digital twin simulation |
| `cfd` | Computational Fluid Dynamics |
| `fea` | Finite Element Analysis |
| `orbital_mc` | Orbital Monte Carlo simulation |

## Error Handling

```typescript
import { QRATUMAPIError } from '@qratum/sdk';

try {
  await client.jobs.submit({ ... });
} catch (error) {
  if (error instanceof QRATUMAPIError) {
    console.error(`API Error: ${error.code} - ${error.description}`);
    console.error(`Status: ${error.statusCode}`);
  } else {
    throw error;
  }
}
```

## Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `baseUrl` | string | `https://api.qratum.io/v1` | API base URL |
| `clientId` | string | - | OAuth2 client ID |
| `clientSecret` | string | - | OAuth2 client secret |
| `accessToken` | string | - | Pre-existing access token |
| `apiKey` | string | - | API key |
| `timeout` | number | `30000` | Request timeout (ms) |
| `autoRefresh` | boolean | `true` | Auto-refresh tokens |

## License

Apache 2.0 - See [LICENSE](../../LICENSE) for details.
