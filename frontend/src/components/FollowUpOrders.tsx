import React, { useMemo } from "react";
import { CalendarDays, ClipboardList, Hash, Stethoscope } from "lucide-react";
import type { Task } from "../api/client";

export interface FollowUpOrdersProps {
  tasks: Task[];
  patientId?: string;
}

const classifySchedule = (dateString: string) => {
  const target = new Date(dateString);
  const today = new Date();
  const diffDays = Math.ceil((target.getTime() - today.getTime()) / (1000 * 60 * 60 * 24));

  if (diffDays <= 7) {
    return { label: "Due soon", className: "order-chip order-chip--urgent" };
  }
  if (diffDays <= 30) {
    return { label: "Scheduled", className: "order-chip order-chip--scheduled" };
  }
  return { label: "Planned", className: "order-chip order-chip--planned" };
};

export const FollowUpOrders: React.FC<FollowUpOrdersProps> = ({ tasks, patientId }) => {
  const sortedTasks = useMemo(
    () =>
      [...tasks].sort(
        (a, b) =>
          new Date(a.scheduled_date).getTime() - new Date(b.scheduled_date).getTime()
      ),
    [tasks]
  );

  return (
    <section className="ehr-card orders-panel">
      <header className="orders-panel__header">
        <div>
          <p className="patient-summary__eyebrow">Care coordination</p>
          <h2>Follow-up orders</h2>
          <p className="text-muted">
            Orders are automatically prepared with guideline-backed reasoning.
          </p>
        </div>
        <div className="orders-panel__context">
          <Stethoscope size={18} />
          <span>{patientId ? `Patient ${patientId}` : "Patient ID pending"}</span>
        </div>
      </header>

      {sortedTasks.length === 0 ? (
        <div className="orders-panel__empty">
          <ClipboardList size={32} />
          <p>No follow-up orders were generated for this report.</p>
        </div>
      ) : (
        <div className="orders-panel__table">
          {sortedTasks.map((task) => {
            const scheduleState = classifySchedule(task.scheduled_date);
            return (
              <article key={task.task_id} className="orders-panel__row">
                <div className="orders-panel__procedure">
                  <Hash size={18} />
                  <div>
                    <strong>{task.procedure}</strong>
                    <span className="orders-panel__order-id">
                      {task.order_id ?? task.task_id}
                    </span>
                  </div>
                </div>
                <div className="orders-panel__date">
                  <CalendarDays size={18} />
                  <span>{new Date(task.scheduled_date).toLocaleDateString()}</span>
                </div>
                <div className="orders-panel__reason">{task.reason}</div>
                <span className={scheduleState.className}>{scheduleState.label}</span>
              </article>
            );
          })}
        </div>
      )}
    </section>
  );
};

