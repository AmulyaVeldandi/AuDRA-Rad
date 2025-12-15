# AuDRA-Rad: Autonomous Radiology Follow-up Assistant

Closing the deadliest gap in radiology workflows by turning unstructured reports into tracked, guideline-backed care plans.

---

## Overview

AuDRA-Rad combines NVIDIA NIM foundation models, retrieval-augmented generation (RAG), and FHIR-compliant integrations to extract critical findings, match the right guideline, and create follow-up orders inside the EHR. Hospitals can increase follow-up compliance, reduce liability, and deliver timely cancer care.

---

## Prerequisites

- AWS account with permissions to create and manage EKS, ECR, VPC, IAM, and OpenSearch resources
- Local tooling: `kubectl`, `eksctl`, `docker`, `awscli`, `helm`, `python3`
- NVIDIA NIM access via [build.nvidia.com](https://build.nvidia.com) with API keys and registry entitlement
- Hackathon-issued $100 AWS credits (track spend carefully)
- Optional: `terraform` or AWS CDK if you prefer infrastructure as code

---

## Quick Start

```bash
   # Clone repo
   git clone https://github.com/yourusername/audra-rad.git
   cd audra-rad
   
   # Setup environment
   cp .env.example .env
   # Edit .env with your credentials
   
   # Index guidelines locally
   docker-compose up -d  # Start local OpenSearch
   python scripts/index_guidelines.py --local
   python scripts/seed_sample_data.py --dry-run  # Preview demo payloads
   python scripts/test_nim_connection.py --embeddings --llm
   
   # Run locally
   uvicorn src.api.app:app --reload
   
   # Test
   curl -X POST http://localhost:8000/api/v1/process-report \
     -H "Content-Type: application/json" \
     -d @data/sample_reports/fhir_chest_ct_ggo.json
```

- The `.env` file stores NIM endpoints, API keys, and AWS credentials consumed by the RAG pipeline.
- Local OpenSearch runs via `docker-compose` with default ports (see `docker-compose.yml` for overrides).

---

## Environment Configuration

Populate `.env` (or your shell) with the values below before running the CLI helpers or API:

| Variable | Purpose | Required | Example |
| --- | --- | --- | --- |
| `ENVIRONMENT` | Controls logging format & env validation | always | `dev` |
| `NIM_LLM_ENDPOINT` | Nemotron NIM base URL | for LLM features | `https://integrate.api.nvidia.com/v1` |
| `NIM_LLM_API_KEY` | Nemotron API key | for LLM features | `nvai-...` |
| `NIM_EMBEDDING_ENDPOINT` | NV-Embed base URL | for guideline indexing | `https://integrate.api.nvidia.com/v1` |
| `NIM_EMBEDDING_API_KEY` | NV-Embed API key | for guideline indexing | `nvai-...` |
| `OPENSEARCH_ENDPOINT` | OpenSearch endpoint URL | staging/prod or custom local port | `https://search-your-domain...` |
| `AWS_REGION` | Region for OpenSearch Serverless signing | staging/prod | `us-west-2` |

If `ENVIRONMENT` is set to `staging` or `prod`, the application enforces that all NIM and OpenSearch variables are present. For local development with `docker-compose`, you can leave `OPENSEARCH_ENDPOINT` unset to use `http://localhost:9200`.

---

## Deploy to AWS EKS

1. **Confirm prerequisites**
   - AWS CLI configured (`aws sts get-caller-identity`)
   - `eksctl`, `kubectl`, `helm`, and `docker` installed
   - NVIDIA NGC/NIM registry access verified (`docker login nvcr.io`)
2. **Provision infrastructure**
   - Create EKS cluster and GPU-capable node group
   - Configure IAM roles, OpenSearch Serverless collection, and ECR repository
   - Allocate an ACM certificate and Route 53 hosted zone if exposing a public endpoint
3. **Build and ship the container**
   - `docker build -t <account>.dkr.ecr.<region>.amazonaws.com/audra-rad:<tag> .`
   - `aws ecr get-login-password | docker login`
   - `docker push` to ECR
4. **Configure Kubernetes addons**
   - Install AWS VPC CNI, Cluster Autoscaler, metrics server, and ALB Ingress Controller
   - Apply GPU device plugin DaemonSet (`kubectl apply -f nvidia-device-plugin.yml`)
5. **Deploy the application**
   - Apply secrets (`kubectl create secret generic audra-env --from-env-file=.env`)
   - Apply manifests in `deployment/kubernetes` (`kubectl apply -f deployment/kubernetes/`)
6. **Validate**
   - `kubectl get pods -n audra` ensures workloads are ready
   - `kubectl logs deploy/audra-api -n audra` checks application startup
   - Hit the ALB endpoint `/healthz` and `/docs`

**Cost estimate:** ~\$3/hour for a single g5.xlarge node, OpenSearch Serverless collection, and supporting services. Expect to burn through hackathon credits in ~30 hours of continuous runtime.

**Budget tips:** pause node groups overnight, delete unused ALBs, and stop OpenSearch collections when idle.

See `docs/DEPLOYMENT.md` for copy-paste commands, infrastructure diagrams, and troubleshooting screenshots.

---

## Architecture Diagram

![Architecture Diagram](assets/architecture_diagram.png)

- **Radiology ingestion**: FHIR-compatible parser normalizes raw reports into structured findings.
- **Guideline retrieval**: OpenSearch Serverless stores embedded medical guidelines indexed via the RAG pipeline.
- **Reasoning engine**: NVIDIA Nemotron NIM evaluates findings against the guideline corpus to generate recommendations.
- **Validation & safety**: Custom validators enforce guardrails, flag high-risk edge cases, and log decisions.
- **EHR integration**: FastAPI service exposes REST endpoints and pushes tasks/orders back to hospital systems.
- **Observability**: CloudWatch metrics, structured logs, and AWS X-Ray traces support operations at scale.

More system internals live in `docs/ARCHITECTURE.md`.

---

## API Documentation

- Interactive docs are available at `https://<your-domain>/docs` (FastAPI Swagger UI).
- Programmatic schema: `https://<your-domain>/openapi.json`.
- Refer to `docs/API.md` for request/response examples, status codes, and error dictionary.

Example requests:

```bash
curl -X POST https://<your-domain>/api/v1/process-report \
  -H "Authorization: Bearer $AUDRA_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d @data/sample_reports/fhir_chest_ct_ggo.json

curl -X GET https://<your-domain>/api/v1/orders?status=pending \
  -H "Authorization: Bearer $AUDRA_API_TOKEN"
```

---

## Demo Video

[![Watch the demo](assets/demo_thumbnail.png)](demo_video.mp4)

The video walks through ingesting a real CT report, guideline retrieval with citations, autonomous EHR follow-up order creation, and clinician-facing alerts inside the dashboard.

---

## Examples

Check out the `examples/` directory for hands-on demonstrations:

### Ollama (Local LLM)
Run AuDRA-Rad with local Llama 3.1 8B via Ollama:
```bash
# Quick test
python examples/ollama/test_ollama_simple.py

# Full radiology analysis suite
python examples/ollama/test_radiology_analysis.py

# Interactive notebook
jupyter notebook examples/ollama/ollama_radiology_analysis.ipynb
```

See [`examples/ollama/README.md`](examples/ollama/README.md) for setup and detailed usage.

### MIMIC-IV Exploration
Analyze real-world radiology reports from MIMIC-IV:
```bash
jupyter notebook examples/mimic_exploration/explore_mimic_data.ipynb
```

See [`examples/mimic_exploration/README.md`](examples/mimic_exploration/README.md) for data access and analysis examples.

### Agent Demonstrations
Explore the original agent notebooks:
- [`notebooks/01_explore_reports.ipynb`](notebooks/01_explore_reports.ipynb) - Report exploration
- [`notebooks/02_test_retrieval.ipynb`](notebooks/02_test_retrieval.ipynb) - Guideline retrieval
- [`notebooks/03_agent_demo.ipynb`](notebooks/03_agent_demo.ipynb) - Full agent workflow

---

## Testing

```bash
pytest tests/ -v
./scripts/test_deployment.sh
```

- The deployment smoke test checks Kubernetes manifests, required secrets, and service readiness.
- Provide `NVIDIA_API_KEY` and `OPENSEARCH_URL` in your environment before running integration tests.

---

## CLI Utilities

- `scripts/index_guidelines.py` - chunk guideline markdown files, call NV-Embed, and upsert vectors. Flags:
  - `--local` to force `http://localhost:9200`
  - `--drop-existing` to recreate the index
- `scripts/seed_sample_data.py` - generate demo ServiceRequests for the bundled sample DiagnosticReports. Defaults to the mock EHR; pass `--remote --ehr-base-url=https://your-ehr` to hit a live endpoint.
- `scripts/test_nim_connection.py` - smoke test NV-Embed and Nemotron connectivity. Disable individual checks with `--no-embeddings` or `--no-llm`.

---

## Cost Management

- **Monitor usage**
  - `kubectl top nodes -n kube-system` (requires metrics-server)
  - `aws cloudwatch get-metric-statistics --namespace AWS/EKS --metric-name node_cpu_utilization ...`
- **Pause infrastructure**
  - `eksctl scale nodegroup --cluster audra --name gpu-workers --nodes 0`
  - `aws opensearchserverless delete-collection --id audra-guidelines` (recreate when needed)
- **Set alerts**
  - Create a CloudWatch billing alarm at \$80 to preserve buffer
  - Enable AWS Budgets email + SNS notifications for the hackathon credits

Always delete the cluster (`eksctl delete cluster --name audra`) after demos to avoid surprise charges.

---

## Troubleshooting

- **Image pull errors**: Run `docker login nvcr.io` and `aws ecr get-login-password` before `kubectl apply`. Ensure the node IAM role has `AmazonEC2ContainerRegistryReadOnly`.
- **Pods stuck in Init**: Confirm GPU nodes are present (`kubectl get nodes -l nvidia.com/gpu.present=true`) and the NVIDIA device plugin DaemonSet is running.
- **OpenSearch connection refused**: Verify security group rules, correct endpoint URL in `.env`, and that the collection is started.
- **Ingress 502/504**: Check ALB target group health, confirm FastAPI service responds at `/healthz`, and inspect `kubectl logs deploy/audra-api -n audra`.
- **View logs**: `kubectl logs -n audra deploy/audra-api -f`, `kubectl logs -n kube-system ds/nvidia-device-plugin-daemonset`, and `aws logs tail /aws/eks/audra/cluster --follow`.

Need help? Reach the team at `support@audra-rad.dev`.

---

## License & Acknowledgments

- Licensed under the [MIT License](LICENSE).
- Built with support from NVIDIA NIM, AWS credit program, and the open radiology community.
- Medical guideline content courtesy of the Fleischner Society, ACR, and broader evidence-based medicine contributors.

---

## Additional Resources

- `docs/ARCHITECTURE.md` - component deep dive, sequence diagrams, and data contracts
- `docs/DEPLOYMENT.md` - detailed AWS setup with screenshots and IaC snippets
- `data/guidelines/` - curated guideline corpus (Fleischner 2017, more coming)
- `tests/` - unit and integration tests covering parsers, validators, and end-to-end flows

---

**Built for the NVIDIA + AWS Agentic AI Hackathon 2025 - every finding deserves follow-through.**
