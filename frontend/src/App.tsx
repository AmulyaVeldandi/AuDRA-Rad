import React, { useCallback, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { isAxiosError } from "axios";
import {
  api,
  ProcessReportRequest,
  ProcessReportResponse,
  HealthResponse as ApiHealthResponse,
} from "./api/client";
import { ReportUpload } from "./components/ReportUpload";
import { ReportViewer } from "./components/ReportViewer";
import { FindingsList } from "./components/FindingsList";
import { GuidelineMatches } from "./components/GuidelineMatches";
import { FollowUpOrders } from "./components/FollowUpOrders";
import { HealthIndicator } from "./components/HealthIndicator";
import { LoadingSpinner } from "./components/LoadingSpinner";
import { PatientSummary } from "./components/PatientSummary";

type ReportResult = ProcessReportResponse & { reportText: string };

const mapHealthResponse = (health?: ApiHealthResponse) =>
  health
    ? {
        status: health.status,
        services: health.services,
        timestamp: health.timestamp,
      }
    : undefined;

export const App: React.FC = () => {
  const [results, setResults] = useState<ReportResult | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [activePatientId, setActivePatientId] = useState<string | undefined>(undefined);

  const { data: healthData, isLoading: healthLoading } = useQuery({
    queryKey: ["health"],
    queryFn: api.getHealth,
    refetchInterval: 30000,
    refetchOnWindowFocus: false,
    staleTime: 15000,
  });

  const handleSubmit = useCallback(
    async (reportText: string, patientId?: string) => {
      setIsSubmitting(true);
      const payload: ProcessReportRequest = {
        report_text: reportText,
      };
      const normalizedPatientId = patientId?.trim();
      if (normalizedPatientId) {
        payload.patient_id = normalizedPatientId;
      }

      try {
        const response = await api.processReport(payload);
        setResults({ ...response, reportText });
        setActivePatientId(normalizedPatientId || undefined);
      } catch (error) {
        if (isAxiosError(error)) {
          const message =
            (typeof error.response?.data === "object" && error.response?.data !== null
              ? (error.response.data as { message?: string }).message
              : undefined) ??
            error.message ??
            "Failed to process the report.";
          throw new Error(message);
        }
        throw error instanceof Error ? error : new Error("Failed to process the report.");
      } finally {
        setIsSubmitting(false);
      }
    },
    []
  );

  return (
    <div className="app">
      <HealthIndicator healthData={mapHealthResponse(healthData)} isLoading={healthLoading} />
      <header className="app__header">
        <h1>🩺 AuDRA-Rad</h1>
        <p className="subtitle">
          Autonomous Radiology Follow-up Assistant
        </p>
        <p className="text-muted" style={{ marginTop: '8px', fontSize: '0.95rem' }}>
          Transform radiology findings into actionable follow-up care with AI-powered guideline matching and automated task generation.
        </p>
      </header>

      <main className="app__main">
        <section className="upload-section">
          <ReportUpload onSubmit={handleSubmit} isLoading={isSubmitting} />
        </section>

        {isSubmitting && !results && (
          <LoadingSpinner size="medium" message="Processing report..." />
        )}

        {results && (
          <section className="ehr-layout">
            <div className="ehr-column">
              <PatientSummary
                patientId={activePatientId}
                reportId={results.report_id}
                sessionId={results.session_id}
                status={results.status}
                requiresReview={results.requires_human_review}
                totalFindings={results.findings.length}
                totalRecommendations={results.recommendations.length}
                totalTasks={results.tasks.length}
              />
              <ReportViewer
                reportText={results.reportText}
                status={results.status}
                processingTimeMs={results.processing_time_ms}
                sessionId={results.session_id}
                message={results.message}
              />
              <GuidelineMatches recommendations={results.recommendations} />
            </div>
            <div className="ehr-column ehr-column--secondary">
              <FollowUpOrders tasks={results.tasks} patientId={activePatientId} />
              <FindingsList findings={results.findings} />
            </div>
          </section>
        )}
      </main>
    </div>
  );
};
