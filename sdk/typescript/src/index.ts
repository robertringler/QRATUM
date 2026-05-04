/**
 * QRATUM Platform TypeScript SDK
 *
 * TypeScript/JavaScript client for the QRATUM Platform API.
 *
 * Features:
 * - OAuth2/OIDC authentication
 * - Job submission and management
 * - Real-time status monitoring (WebSocket)
 * - Results retrieval
 * - Resource allocation dashboard
 *
 * @example
 * ```typescript
 * import { QRATUMClient } from '@qratum/sdk';
 *
 * const client = new QRATUMClient({
 *   baseUrl: 'https://api.qratum.io/v1',
 *   clientId: 'your-client-id',
 *   clientSecret: 'your-client-secret',
 * });
 *
 * // Authenticate
 * await client.authenticate();
 *
 * // Submit a job
 * const job = await client.jobs.submit({
 *   jobType: 'vqe',
 *   config: { molecule: 'H2', basis: 'sto-3g' },
 * });
 *
 * // Subscribe to status updates
 * client.status.subscribe(job.jobId, (status) => {
 *   console.log(`Job ${job.jobId}: ${status.status} (${status.progress * 100}%)`);
 * });
 *
 * // Get results when complete
 * const results = await client.results.get(job.jobId);
 * ```
 */

// =============================================================================
// Types and Interfaces
// =============================================================================

/** Job status enum */
export type JobStatus = 'queued' | 'running' | 'completed' | 'failed' | 'cancelled';

/** Job type enum */
export type JobType =
  | 'quantum_circuit'
  | 'tensor_analysis'
  | 'vqe'
  | 'qaoa'
  | 'digital_twin'
  | 'cfd'
  | 'fea'
  | 'orbital_mc';

/** Cloud provider enum */
export type CloudProvider = 'aws' | 'gcp' | 'azure';

/** Cluster status enum */
export type ClusterStatus = 'online' | 'degraded' | 'offline';

/** Client configuration */
export interface QRATUMClientConfig {
  /** Base URL of the API (default: https://api.qratum.io/v1) */
  baseUrl?: string;
  /** OAuth2 client ID */
  clientId?: string;
  /** OAuth2 client secret */
  clientSecret?: string;
  /** Pre-existing access token */
  accessToken?: string;
  /** API key for service-to-service auth */
  apiKey?: string;
  /** Request timeout in milliseconds (default: 30000) */
  timeout?: number;
  /** Whether to automatically refresh tokens (default: true) */
  autoRefresh?: boolean;
}

/** Token response */
export interface TokenResponse {
  accessToken: string;
  tokenType: string;
  expiresIn: number;
  refreshToken?: string;
  scope?: string;
}

/** Job submission request */
export interface JobSubmitRequest {
  jobType: JobType;
  name?: string;
  config: Record<string, unknown>;
  priority?: number;
  timeoutSeconds?: number;
  tags?: string[];
  callbackUrl?: string;
}

/** Job submission response */
export interface JobSubmitResponse {
  jobId: string;
  status: JobStatus;
  submittedAt: string;
  estimatedDurationSeconds?: number;
}

/** Job summary */
export interface JobSummary {
  jobId: string;
  name?: string;
  jobType: JobType;
  status: JobStatus;
  submittedAt: string;
  startedAt?: string;
  completedAt?: string;
  progress: number;
}

/** Job detail */
export interface JobDetail extends JobSummary {
  config: Record<string, unknown>;
  priority: number;
  timeoutSeconds: number;
  tags?: string[];
  cluster?: string;
  workerId?: string;
  errorMessage?: string;
}

/** Job list response */
export interface JobListResponse {
  jobs: JobSummary[];
  total: number;
  limit: number;
  offset: number;
}

/** Job status response */
export interface JobStatusResponse {
  jobId: string;
  status: JobStatus;
  progress: number;
  message?: string;
  metrics?: StatusMetrics;
}

/** Status metrics */
export interface StatusMetrics {
  elapsedSeconds?: number;
  cpuUtilization?: number;
  memoryMb?: number;
  gpuUtilization?: number;
}

/** Job results response */
export interface JobResultsResponse {
  jobId: string;
  status: JobStatus;
  completedAt?: string;
  results: Record<string, unknown>;
  metrics?: ResultMetrics;
  visualizationUrl?: string;
}

/** Result metrics */
export interface ResultMetrics {
  executionTimeSeconds?: number;
  fidelity?: number;
  energy?: number;
  iterations?: number;
}

/** Artifact metadata */
export interface Artifact {
  artifactId: string;
  filename: string;
  contentType: string;
  sizeBytes: number;
  createdAt: string;
  downloadUrl?: string;
}

/** Artifact list response */
export interface ArtifactListResponse {
  artifacts: Artifact[];
}

/** Cluster summary */
export interface ClusterSummary {
  clusterId: string;
  name?: string;
  provider: CloudProvider;
  region: string;
  status: ClusterStatus;
  nodes: number;
  gpusAvailable: number;
  gpusTotal: number;
}

/** Resource dashboard */
export interface ResourceDashboard {
  timestamp: string;
  clusters: ClusterSummary[];
  queueDepth: number;
  jobsRunning: number;
  jobsQueued: number;
  utilization: {
    cpu: number;
    memory: number;
    gpu: number;
  };
}

/** Quota response */
export interface QuotaResponse {
  quotas: {
    maxConcurrentJobs: number;
    maxGpuHoursPerMonth: number;
    maxStorageGb: number;
  };
  usage: {
    concurrentJobs: number;
    gpuHoursThisMonth: number;
    storageGb: number;
  };
}

/** Validation response */
export interface ValidationResponse {
  valid: boolean;
  errors: string[];
  warnings: string[];
}

/** API error */
export interface APIError {
  error: string;
  errorDescription: string;
  requestId?: string;
}

/** Status update callback */
export type StatusCallback = (status: JobStatusResponse) => void;

// =============================================================================
// HTTP Client
// =============================================================================

/** HTTP request options */
interface RequestOptions {
  method: 'GET' | 'POST' | 'PUT' | 'DELETE';
  path: string;
  body?: unknown;
  params?: Record<string, string | number | undefined>;
  headers?: Record<string, string>;
}

/**
 * Base HTTP client for API requests.
 */
class HttpClient {
  private baseUrl: string;
  private timeout: number;
  private accessToken?: string;
  private apiKey?: string;

  constructor(config: QRATUMClientConfig) {
    this.baseUrl = config.baseUrl || 'https://api.qratum.io/v1';
    this.timeout = config.timeout || 30000;
    this.accessToken = config.accessToken;
    this.apiKey = config.apiKey;
  }

  setAccessToken(token: string): void {
    this.accessToken = token;
  }

  async request<T>(options: RequestOptions): Promise<T> {
    const url = this.buildUrl(options.path, options.params);
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      ...options.headers,
    };

    if (this.accessToken) {
      headers['Authorization'] = `Bearer ${this.accessToken}`;
    } else if (this.apiKey) {
      headers['X-API-Key'] = this.apiKey;
    }

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await fetch(url, {
        method: options.method,
        headers,
        body: options.body ? JSON.stringify(options.body) : undefined,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const error = await response.json() as APIError;
        throw new QRATUMAPIError(error.error, error.errorDescription, response.status);
      }

      return response.json() as Promise<T>;
    } catch (err) {
      clearTimeout(timeoutId);
      if (err instanceof QRATUMAPIError) {
        throw err;
      }
      throw new QRATUMAPIError('network_error', `Request failed: ${err}`);
    }
  }

  private buildUrl(path: string, params?: Record<string, string | number | undefined>): string {
    const url = new URL(path, this.baseUrl);
    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined) {
          url.searchParams.append(key, String(value));
        }
      });
    }
    return url.toString();
  }
}

// =============================================================================
// Error Classes
// =============================================================================

/**
 * API error class.
 */
export class QRATUMAPIError extends Error {
  constructor(
    public code: string,
    public description: string,
    public statusCode?: number,
  ) {
    super(`${code}: ${description}`);
    this.name = 'QRATUMAPIError';
  }
}

// =============================================================================
// API Modules
// =============================================================================

/**
 * Jobs API module.
 */
class JobsAPI {
  constructor(private http: HttpClient) {}

  /**
   * Submit a new job.
   */
  async submit(request: JobSubmitRequest): Promise<JobSubmitResponse> {
    const response = await this.http.request<{
      job_id: string;
      status: JobStatus;
      submitted_at: string;
      estimated_duration_seconds?: number;
    }>({
      method: 'POST',
      path: '/jobs',
      body: {
        job_type: request.jobType,
        name: request.name,
        config: request.config,
        priority: request.priority,
        timeout_seconds: request.timeoutSeconds,
        tags: request.tags,
        callback_url: request.callbackUrl,
      },
    });

    return {
      jobId: response.job_id,
      status: response.status,
      submittedAt: response.submitted_at,
      estimatedDurationSeconds: response.estimated_duration_seconds,
    };
  }

  /**
   * List jobs with optional filtering.
   */
  async list(options?: {
    status?: JobStatus;
    jobType?: JobType;
    limit?: number;
    offset?: number;
  }): Promise<JobListResponse> {
    const response = await this.http.request<{
      jobs: Array<{
        job_id: string;
        name?: string;
        job_type: JobType;
        status: JobStatus;
        submitted_at: string;
        started_at?: string;
        completed_at?: string;
        progress: number;
      }>;
      total: number;
      limit: number;
      offset: number;
    }>({
      method: 'GET',
      path: '/jobs',
      params: {
        status: options?.status,
        job_type: options?.jobType,
        limit: options?.limit,
        offset: options?.offset,
      },
    });

    return {
      jobs: response.jobs.map((j) => ({
        jobId: j.job_id,
        name: j.name,
        jobType: j.job_type,
        status: j.status,
        submittedAt: j.submitted_at,
        startedAt: j.started_at,
        completedAt: j.completed_at,
        progress: j.progress,
      })),
      total: response.total,
      limit: response.limit,
      offset: response.offset,
    };
  }

  /**
   * Get job details.
   */
  async get(jobId: string): Promise<JobDetail> {
    const response = await this.http.request<{
      job_id: string;
      name?: string;
      job_type: JobType;
      status: JobStatus;
      submitted_at: string;
      started_at?: string;
      completed_at?: string;
      progress: number;
      config: Record<string, unknown>;
      priority: number;
      timeout_seconds: number;
      tags?: string[];
      cluster?: string;
      worker_id?: string;
      error_message?: string;
    }>({
      method: 'GET',
      path: `/jobs/${jobId}`,
    });

    return {
      jobId: response.job_id,
      name: response.name,
      jobType: response.job_type,
      status: response.status,
      submittedAt: response.submitted_at,
      startedAt: response.started_at,
      completedAt: response.completed_at,
      progress: response.progress,
      config: response.config,
      priority: response.priority,
      timeoutSeconds: response.timeout_seconds,
      tags: response.tags,
      cluster: response.cluster,
      workerId: response.worker_id,
      errorMessage: response.error_message,
    };
  }

  /**
   * Cancel a job.
   */
  async cancel(jobId: string): Promise<{ jobId: string; status: JobStatus; cancelledAt: string }> {
    const response = await this.http.request<{
      job_id: string;
      status: JobStatus;
      cancelled_at: string;
    }>({
      method: 'DELETE',
      path: `/jobs/${jobId}`,
    });

    return {
      jobId: response.job_id,
      status: response.status,
      cancelledAt: response.cancelled_at,
    };
  }

  /**
   * Validate job configuration without submitting.
   */
  async validate(request: JobSubmitRequest): Promise<ValidationResponse> {
    return this.http.request<ValidationResponse>({
      method: 'POST',
      path: '/validate',
      body: {
        job_type: request.jobType,
        config: request.config,
      },
    });
  }
}

/**
 * Status API module with WebSocket support.
 */
class StatusAPI {
  private ws?: WebSocket;
  private subscriptions: Map<string, Set<StatusCallback>> = new Map();
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;

  constructor(
    private http: HttpClient,
    private config: QRATUMClientConfig,
  ) {}

  /**
   * Get job status.
   */
  async get(jobId: string): Promise<JobStatusResponse> {
    const response = await this.http.request<{
      job_id: string;
      status: JobStatus;
      progress: number;
      message?: string;
      metrics?: {
        elapsed_seconds?: number;
        cpu_utilization?: number;
        memory_mb?: number;
        gpu_utilization?: number;
      };
    }>({
      method: 'GET',
      path: `/jobs/${jobId}/status`,
    });

    return {
      jobId: response.job_id,
      status: response.status,
      progress: response.progress,
      message: response.message,
      metrics: response.metrics
        ? {
            elapsedSeconds: response.metrics.elapsed_seconds,
            cpuUtilization: response.metrics.cpu_utilization,
            memoryMb: response.metrics.memory_mb,
            gpuUtilization: response.metrics.gpu_utilization,
          }
        : undefined,
    };
  }

  /**
   * Subscribe to real-time status updates via WebSocket.
   */
  subscribe(jobId: string, callback: StatusCallback): () => void {
    // Add to subscriptions
    if (!this.subscriptions.has(jobId)) {
      this.subscriptions.set(jobId, new Set());
    }
    this.subscriptions.get(jobId)!.add(callback);

    // Connect WebSocket if not connected
    this.ensureConnected().then(() => {
      this.sendMessage({ action: 'subscribe', job_id: jobId });
    });

    // Return unsubscribe function
    return () => {
      const callbacks = this.subscriptions.get(jobId);
      if (callbacks) {
        callbacks.delete(callback);
        if (callbacks.size === 0) {
          this.subscriptions.delete(jobId);
          this.sendMessage({ action: 'unsubscribe', job_id: jobId });
        }
      }
    };
  }

  /**
   * Disconnect WebSocket.
   */
  disconnect(): void {
    if (this.ws) {
      this.ws.close();
      this.ws = undefined;
    }
    this.subscriptions.clear();
  }

  private async ensureConnected(): Promise<void> {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      return;
    }

    return new Promise((resolve, reject) => {
      // Properly convert HTTP(S) URL to WS(S) URL
      let wsUrl = this.config.baseUrl || 'https://api.qratum.io/v1';
      if (wsUrl.startsWith('https://')) {
        wsUrl = wsUrl.replace('https://', 'wss://');
      } else if (wsUrl.startsWith('http://')) {
        wsUrl = wsUrl.replace('http://', 'ws://');
      }
      const token = this.config.accessToken;

      this.ws = new WebSocket(`${wsUrl}/ws/status?token=${token}`);

      this.ws.onopen = () => {
        this.reconnectAttempts = 0;
        resolve();
      };

      this.ws.onerror = (error) => {
        reject(new QRATUMAPIError('websocket_error', `WebSocket error: ${error}`));
      };

      this.ws.onclose = () => {
        this.handleDisconnect();
      };

      this.ws.onmessage = (event) => {
        this.handleMessage(JSON.parse(event.data));
      };
    });
  }

  private handleMessage(message: {
    type: string;
    job_id?: string;
    status?: JobStatus;
    progress?: number;
    message?: string;
  }): void {
    if (message.type === 'status' && message.job_id) {
      const callbacks = this.subscriptions.get(message.job_id);
      if (callbacks) {
        const status: JobStatusResponse = {
          jobId: message.job_id,
          status: message.status || 'queued',
          progress: message.progress || 0,
          message: message.message,
        };
        callbacks.forEach((cb) => cb(status));
      }
    }
  }

  private handleDisconnect(): void {
    if (this.reconnectAttempts < this.maxReconnectAttempts && this.subscriptions.size > 0) {
      this.reconnectAttempts++;
      setTimeout(() => {
        this.ensureConnected().then(() => {
          // Re-subscribe to all jobs
          this.subscriptions.forEach((_, jobId) => {
            this.sendMessage({ action: 'subscribe', job_id: jobId });
          });
        });
      }, Math.pow(2, this.reconnectAttempts) * 1000);
    }
  }

  private sendMessage(message: Record<string, unknown>): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    }
  }
}

/**
 * Results API module.
 */
class ResultsAPI {
  constructor(private http: HttpClient) {}

  /**
   * Get job results.
   */
  async get(jobId: string, format?: 'json' | 'csv' | 'hdf5'): Promise<JobResultsResponse> {
    const response = await this.http.request<{
      job_id: string;
      status: JobStatus;
      completed_at?: string;
      results: Record<string, unknown>;
      metrics?: {
        execution_time_seconds?: number;
        fidelity?: number;
        energy?: number;
        iterations?: number;
      };
      visualization_url?: string;
    }>({
      method: 'GET',
      path: `/jobs/${jobId}/results`,
      params: { format },
    });

    return {
      jobId: response.job_id,
      status: response.status,
      completedAt: response.completed_at,
      results: response.results,
      metrics: response.metrics
        ? {
            executionTimeSeconds: response.metrics.execution_time_seconds,
            fidelity: response.metrics.fidelity,
            energy: response.metrics.energy,
            iterations: response.metrics.iterations,
          }
        : undefined,
      visualizationUrl: response.visualization_url,
    };
  }

  /**
   * List job artifacts.
   */
  async listArtifacts(jobId: string): Promise<ArtifactListResponse> {
    const response = await this.http.request<{
      artifacts: Array<{
        artifact_id: string;
        filename: string;
        content_type: string;
        size_bytes: number;
        created_at: string;
        download_url?: string;
      }>;
    }>({
      method: 'GET',
      path: `/jobs/${jobId}/artifacts`,
    });

    return {
      artifacts: response.artifacts.map((a) => ({
        artifactId: a.artifact_id,
        filename: a.filename,
        contentType: a.content_type,
        sizeBytes: a.size_bytes,
        createdAt: a.created_at,
        downloadUrl: a.download_url,
      })),
    };
  }

  /**
   * Get visualization data.
   */
  async getVisualization(jobId: string): Promise<Record<string, unknown>> {
    return this.http.request<Record<string, unknown>>({
      method: 'GET',
      path: `/jobs/${jobId}/visualization`,
    });
  }
}

/**
 * Resources API module.
 */
class ResourcesAPI {
  constructor(private http: HttpClient) {}

  /**
   * Get resource dashboard.
   */
  async getDashboard(): Promise<ResourceDashboard> {
    const response = await this.http.request<{
      timestamp: string;
      clusters: Array<{
        cluster_id: string;
        name?: string;
        provider: CloudProvider;
        region: string;
        status: ClusterStatus;
        nodes: number;
        gpus_available: number;
        gpus_total: number;
      }>;
      queue_depth: number;
      jobs_running: number;
      jobs_queued: number;
      utilization: {
        cpu: number;
        memory: number;
        gpu: number;
      };
    }>({
      method: 'GET',
      path: '/resources',
    });

    return {
      timestamp: response.timestamp,
      clusters: response.clusters.map((c) => ({
        clusterId: c.cluster_id,
        name: c.name,
        provider: c.provider,
        region: c.region,
        status: c.status,
        nodes: c.nodes,
        gpusAvailable: c.gpus_available,
        gpusTotal: c.gpus_total,
      })),
      queueDepth: response.queue_depth,
      jobsRunning: response.jobs_running,
      jobsQueued: response.jobs_queued,
      utilization: response.utilization,
    };
  }

  /**
   * List available clusters.
   */
  async listClusters(): Promise<ClusterSummary[]> {
    const response = await this.http.request<{
      clusters: Array<{
        cluster_id: string;
        name?: string;
        provider: CloudProvider;
        region: string;
        status: ClusterStatus;
        nodes: number;
        gpus_available: number;
        gpus_total: number;
      }>;
    }>({
      method: 'GET',
      path: '/resources/clusters',
    });

    return response.clusters.map((c) => ({
      clusterId: c.cluster_id,
      name: c.name,
      provider: c.provider,
      region: c.region,
      status: c.status,
      nodes: c.nodes,
      gpusAvailable: c.gpus_available,
      gpusTotal: c.gpus_total,
    }));
  }

  /**
   * Get quotas and usage.
   */
  async getQuotas(): Promise<QuotaResponse> {
    const response = await this.http.request<{
      quotas: {
        max_concurrent_jobs: number;
        max_gpu_hours_per_month: number;
        max_storage_gb: number;
      };
      usage: {
        concurrent_jobs: number;
        gpu_hours_this_month: number;
        storage_gb: number;
      };
    }>({
      method: 'GET',
      path: '/resources/quotas',
    });

    return {
      quotas: {
        maxConcurrentJobs: response.quotas.max_concurrent_jobs,
        maxGpuHoursPerMonth: response.quotas.max_gpu_hours_per_month,
        maxStorageGb: response.quotas.max_storage_gb,
      },
      usage: {
        concurrentJobs: response.usage.concurrent_jobs,
        gpuHoursThisMonth: response.usage.gpu_hours_this_month,
        storageGb: response.usage.storage_gb,
      },
    };
  }
}

// =============================================================================
// Main Client
// =============================================================================

/**
 * QRATUM Platform API Client.
 *
 * @example
 * ```typescript
 * const client = new QRATUMClient({
 *   baseUrl: 'https://api.qratum.io/v1',
 *   clientId: 'your-client-id',
 *   clientSecret: 'your-client-secret',
 * });
 *
 * await client.authenticate();
 *
 * const job = await client.jobs.submit({
 *   jobType: 'vqe',
 *   config: { molecule: 'H2' },
 * });
 * ```
 */
export class QRATUMClient {
  private http: HttpClient;
  private config: QRATUMClientConfig;
  private tokenExpiresAt?: Date;
  private refreshToken?: string;

  /** Jobs API */
  public jobs: JobsAPI;
  /** Status API */
  public status: StatusAPI;
  /** Results API */
  public results: ResultsAPI;
  /** Resources API */
  public resources: ResourcesAPI;

  constructor(config: QRATUMClientConfig) {
    this.config = {
      baseUrl: 'https://api.qratum.io/v1',
      timeout: 30000,
      autoRefresh: true,
      ...config,
    };

    this.http = new HttpClient(this.config);
    this.jobs = new JobsAPI(this.http);
    this.status = new StatusAPI(this.http, this.config);
    this.results = new ResultsAPI(this.http);
    this.resources = new ResourcesAPI(this.http);
  }

  /**
   * Authenticate with the API using client credentials.
   */
  async authenticate(): Promise<void> {
    if (this.config.accessToken) {
      this.http.setAccessToken(this.config.accessToken);
      return;
    }

    if (this.config.apiKey) {
      // API key auth doesn't need explicit authentication
      return;
    }

    if (!this.config.clientId || !this.config.clientSecret) {
      throw new QRATUMAPIError('auth_error', 'clientId and clientSecret are required');
    }

    const response = await fetch(`${this.config.baseUrl}/auth/token`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      body: new URLSearchParams({
        grant_type: 'client_credentials',
        client_id: this.config.clientId,
        client_secret: this.config.clientSecret,
      }),
    });

    if (!response.ok) {
      const error = await response.json() as { error: string; error_description: string };
      throw new QRATUMAPIError(error.error, error.error_description);
    }

    const token = await response.json() as {
      access_token: string;
      expires_in: number;
      refresh_token?: string;
    };

    this.http.setAccessToken(token.access_token);
    this.config.accessToken = token.access_token;
    this.refreshToken = token.refresh_token;
    this.tokenExpiresAt = new Date(Date.now() + token.expires_in * 1000);

    // Set up auto-refresh
    if (this.config.autoRefresh && this.refreshToken) {
      this.scheduleTokenRefresh(token.expires_in);
    }
  }

  /**
   * Disconnect and clean up resources.
   */
  disconnect(): void {
    this.status.disconnect();
  }

  private scheduleTokenRefresh(expiresIn: number): void {
    // Refresh 60 seconds before expiration
    const refreshIn = (expiresIn - 60) * 1000;
    if (refreshIn > 0) {
      setTimeout(() => this.refreshAccessToken(), refreshIn);
    }
  }

  private async refreshAccessToken(): Promise<void> {
    if (!this.refreshToken) return;

    try {
      const response = await fetch(`${this.config.baseUrl}/auth/refresh`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ refresh_token: this.refreshToken }),
      });

      if (response.ok) {
        const token = await response.json() as {
          access_token: string;
          expires_in: number;
          refresh_token?: string;
        };

        this.http.setAccessToken(token.access_token);
        this.config.accessToken = token.access_token;
        this.refreshToken = token.refresh_token || this.refreshToken;
        this.tokenExpiresAt = new Date(Date.now() + token.expires_in * 1000);

        this.scheduleTokenRefresh(token.expires_in);
      }
    } catch (error) {
      console.error('Token refresh failed:', error);
    }
  }
}

// Export default client
export default QRATUMClient;
