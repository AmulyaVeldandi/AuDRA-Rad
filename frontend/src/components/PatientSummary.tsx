import React from "react";
import { Activity, ClipboardCheck, Fingerprint } from "lucide-react";

interface PatientSummaryProps {
  patientId?: string;
  reportId?: string;
  sessionId?: string;
  status: string;
  totalFindings: number;
  totalRecommendations: number;
  totalTasks: number;
  requiresReview?: boolean;
}

const STATUS_LABELS: Record<
  string,
  { label: string; accent: string; background: string }
> = {
  success: {
    label: "Care Plan Ready",
    accent: "#16a34a",
    background: "rgba(34,197,94,0.18)",
  },
  no_findings: {
    label: "No Findings",
    accent: "#0ea5e9",
    background: "rgba(14,165,233,0.18)",
  },
  requires_review: {
    label: "Requires Review",
    accent: "#ea580c",
    background: "rgba(234,88,12,0.18)",
  },
  error: {
    label: "Review Failed",
    accent: "#dc2626",
    background: "rgba(220,38,38,0.18)",
  },
};

export const PatientSummary: React.FC<PatientSummaryProps> = ({
  patientId,
  reportId,
  sessionId,
  status,
  totalFindings,
  totalRecommendations,
  totalTasks,
  requiresReview,
}) => {
  const statusConfig = STATUS_LABELS[status] ?? STATUS_LABELS.success;

  return (
    <section className="ehr-card patient-summary">
      <header className="patient-summary__header">
        <div>
          <p className="patient-summary__eyebrow">Patient record</p>
          <h2>Patient dashboard</h2>
          <p className="text-muted">
            Snapshot of the current encounter and outstanding follow-up work.
          </p>
        </div>
        <span
          className="patient-summary__status"
          style={{
            color: statusConfig.accent,
            background: statusConfig.background,
          }}
        >
          {requiresReview && status !== "requires_review"
            ? "Flagged for Review"
            : statusConfig.label}
        </span>
      </header>

      <div className="patient-summary__grid">
        <div>
          <span className="patient-summary__label">Patient ID</span>
          <strong>{patientId ?? "Not provided"}</strong>
        </div>
        <div>
          <span className="patient-summary__label">Report ID</span>
          <strong>{reportId}</strong>
        </div>
        <div>
          <span className="patient-summary__label">Session</span>
          <strong>{sessionId}</strong>
        </div>
      </div>

      <div className="patient-summary__metrics">
        <div className="patient-summary__metric">
          <Fingerprint size={32} />
          <div>
            <p className="patient-summary__metric-label">Structured findings</p>
            <strong>{totalFindings}</strong>
          </div>
        </div>
        <div className="patient-summary__metric">
          <Activity size={32} />
          <div>
            <p className="patient-summary__metric-label">Guideline matches</p>
            <strong>{totalRecommendations}</strong>
          </div>
        </div>
        <div className="patient-summary__metric">
          <ClipboardCheck size={32} />
          <div>
            <p className="patient-summary__metric-label">Follow-up orders</p>
            <strong>{totalTasks}</strong>
          </div>
        </div>
      </div>
    </section>
  );
};

