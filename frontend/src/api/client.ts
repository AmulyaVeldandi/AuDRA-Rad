import axios from "axios";

const API_BASE_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export type ProcessingStatus = "success" | "no_findings" | "requires_review" | "error";
export type HealthStatus = "healthy" | "degraded" | "unhealthy";
export type UrgencyLevel = "routine" | "priority" | "urgent" | "stat";

export interface ProcessReportRequest {
  report_text: string;
  patient_id?: string;
  patient_context?: Record<string, unknown>;
  report_id?: string;
  options?: Record<string, unknown>;
}

export interface Finding {
  finding_id: string;
  type: string;
  size_mm?: number;
  location: string;
  characteristics: string[];
  confidence: number;
}

export interface Recommendation {
  recommendation_id: string;
  follow_up_type: string;
  timeframe_months?: number;
  urgency: UrgencyLevel;
  reasoning: string;
  citation: string;
  confidence: number;
}

export interface Task {
  task_id: string;
  procedure: string;
  scheduled_date: string;
  reason: string;
  order_id?: string;
}

export interface ProcessReportResponse {
  status: ProcessingStatus;
  session_id: string;
  report_id?: string;
  findings: Finding[];
  recommendations: Recommendation[];
  tasks: Task[];
  processing_time_ms: number;
  message?: string;
  requires_human_review: boolean;
  decision_trace?: Array<Record<string, unknown>>;
}

export interface HealthResponse {
  status: HealthStatus;
  services: Record<string, HealthStatus>;
  version: string;
  timestamp: string;
}

export interface MetricsSnapshot {
  total_reports_processed: number;
  average_processing_time_ms: number;
  total_findings_detected: number;
  total_tasks_created: number;
  error_rate: number;
}

const apiClient = axios.create({
  baseURL: `${API_BASE_URL}/api/v1`,
  headers: {
    "Content-Type": "application/json",
  },
  timeout: 1000000,
});

export const api = {
  processReport: async (data: ProcessReportRequest): Promise<ProcessReportResponse> => {
    const response = await apiClient.post<ProcessReportResponse>("/process-report", data);
    return response.data;
  },
  getHealth: async (): Promise<HealthResponse> => {
    const response = await apiClient.get<HealthResponse>("/health");
    return response.data;
  },
  getMetrics: async (): Promise<MetricsSnapshot> => {
    const response = await apiClient.get<MetricsSnapshot>("/metrics");
    return response.data;
  },
  getSession: async (sessionId: string): Promise<ProcessReportResponse> => {
    const response = await apiClient.get<ProcessReportResponse>(
      `/session/${encodeURIComponent(sessionId)}`
    );
    return response.data;
  },
};

export default apiClient;
