# AuDRA-Rad Deployment Guide

Use this runbook to deploy AuDRA-Rad on cloud platforms (AWS, Azure, GCP) or locally with Docker. Follow the sections in order and capture screenshots for status reports (add them under `assets/screenshots/`).

---

## AWS Setup

### 1. Configure credentials and environment

```bash
export AWS_REGION=us-west-2
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
aws configure sso  # or aws configure if using access keys
```

- Validate identity: `aws sts get-caller-identity`
- Enable eksctl IAM helper: `aws iam create-service-linked-role --aws-service-name eks.amazonaws.com` (idempotent)

> Screenshot: `assets/screenshots/aws-console-identity.png` (AWS console showing logged-in account)

### 2. Create the EKS control plane

```bash
eksctl create cluster \
  --name audra \
  --region $AWS_REGION \
  --version 1.29 \
  --managed \
  --nodegroup-name cpu-ops \
  --node-type m6i.xlarge \
  --nodes 2 \
  --nodes-min 2 \
  --nodes-max 4 \
  --with-oidc
```

- This creates an initial general-purpose node group for system pods.
- Confirm success: `aws eks describe-cluster --name audra --region $AWS_REGION`

> Screenshot: `assets/screenshots/eks-cluster.png`

### 3. Configure IAM mappings

```bash
eksctl create iamidentitymapping \
  --region $AWS_REGION \
  --cluster audra \
  --arn arn:aws:iam::$AWS_ACCOUNT_ID:role/AWSReservedSSO_AudraAdmin \
  --username admin \
  --group system:masters
```

- Map service accounts for ALB controller later using IAM Roles for Service Accounts (IRSA).

### 4. Set up Amazon ECR

```bash
aws ecr create-repository \
  --repository-name audra-rad \
  --image-tag-mutability IMMUTABLE \
  --encryption-configuration encryptionType=KMS

aws ecr get-login-password --region $AWS_REGION \
  | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com
```

> Screenshot: `assets/screenshots/ecr-repo.png`

### 5. Create OpenSearch Serverless collection

```bash
aws opensearchserverless create-collection \
  --name audra-guidelines \
  --type VECTORSEARCH

aws opensearchserverless create-security-policy \
  --name audra-network \
  --type network \
  --policy '{"rules":[{"resourceType":"collection","resource":["collection/audra-guidelines"],"allowFromPublic":false,"source":["<vpc-endpoint-id>"]}]}'
```

- Replace `<vpc-endpoint-id>` with the VPC endpoint created via the console or CloudFormation.
- Grant data access to the EKS node role with `aws opensearchserverless create-access-policy`.

> Screenshot: `assets/screenshots/opensearch-collection.png`

### 6. Tag resources and set budgets

```bash
aws budgets create-budget --account-id $AWS_ACCOUNT_ID --budget file://docs/budgets/budget.json
```

- Tag cluster nodes and OpenSearch with appropriate cost center tags for tracking.

---

## LLM Service Setup

### Cloud-based LLMs
1. **Choose your LLM provider** (OpenAI, Anthropic, Azure OpenAI, etc.)
2. **Generate an API key** from your provider's console
3. **Update secrets:** add `LLM_API_KEY`, `LLM_ENDPOINT`, and model configuration to `.env`

### Local Ollama Setup
1. **Install Ollama** from `https://ollama.ai`
2. **Pull your model:** `ollama pull llama3.1:8b`
3. **Update .env:** set `LLM_ENDPOINT=http://localhost:11434`

Troubleshooting:
- Authentication errors -> verify API key is valid and has not expired
- Connection refused -> ensure the LLM service endpoint is accessible from your deployment

---

## Kubernetes Configuration

### 1. Install core addons

```bash
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
helm repo add eks https://aws.github.io/eks-charts
helm repo update
```

### 2. Install AWS Load Balancer Controller

```bash
eksctl utils associate-iam-oidc-provider --cluster audra --region $AWS_REGION --approve

curl -o alb-iam-policy.json https://raw.githubusercontent.com/kubernetes-sigs/aws-load-balancer-controller/v2.7.0/docs/install/iam_policy.json
aws iam create-policy \
  --policy-name AWSLoadBalancerControllerIAMPolicy \
  --policy-document file://alb-iam-policy.json

eksctl create iamserviceaccount \
  --cluster audra \
  --namespace kube-system \
  --name aws-load-balancer-controller \
  --attach-policy-arn arn:aws:iam::$AWS_ACCOUNT_ID:policy/AWSLoadBalancerControllerIAMPolicy \
  --region $AWS_REGION \
  --override-existing-serviceaccounts \
  --approve

helm install aws-load-balancer-controller eks/aws-load-balancer-controller \
  -n kube-system \
  --set clusterName=audra \
  --set serviceAccount.create=false \
  --set serviceAccount.name=aws-load-balancer-controller
```

### 3. Add GPU node group

```bash
eksctl create nodegroup \
  --cluster audra \
  --region $AWS_REGION \
  --name gpu-workers \
  --node-type g5.xlarge \
  --nodes 1 \
  --nodes-min 0 \
  --nodes-max 4 \
  --node-taints nvidia.com/gpu=true:NoSchedule \
  --labels accelerators=nvidia \
  --managed

kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.15.0/nvidia-device-plugin.yml
```

### 4. Enable autoscaling

```bash
kubectl apply -f https://github.com/kubernetes/autoscaler/releases/download/cluster-autoscaler-1.29.0/cluster-autoscaler-autodiscover.yaml
kubectl -n kube-system set env deployment/cluster-autoscaler \
  AWS_REGION=$AWS_REGION \
  CLUSTER_NAME=audra \
  --containers=cluster-autoscaler
```

- Patch Cluster Autoscaler to use `--balance-similar-node-groups=true` and `--skip-nodes-with-local-storage=false` per AWS docs.

Troubleshooting:
- ALB controller stuck in `CrashLoopBackOff`: re-run IRSA creation or check policy ARN.
- GPU pods pending: ensure nodes joined (`kubectl get nodes -l accelerators=nvidia`).

---

## Application Deployment

### 1. Build and push the image

```bash
export IMAGE_TAG=v0.1.$(date +%Y%m%d%H%M)
docker build -t audra-rad:$IMAGE_TAG .
docker tag audra-rad:$IMAGE_TAG $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/audra-rad:$IMAGE_TAG
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/audra-rad:$IMAGE_TAG
```

### 2. Prepare namespace and secrets

```bash
kubectl create namespace audra
kubectl create secret generic audra-env \
  --namespace audra \
  --from-env-file=.env
kubectl apply -f deployment/kubernetes/storage/opensearch-secret.yaml
```

### 3. Apply manifests

```bash
kubectl apply -n audra -f deployment/kubernetes/configmaps/
kubectl apply -n audra -f deployment/kubernetes/services/
kubectl apply -n audra -f deployment/kubernetes/deployments/
kubectl apply -n audra -f deployment/kubernetes/ingress/
```

- Update `deployment/kubernetes/deployments/api.yaml` with the pushed image tag.
- Watch rollout: `kubectl rollout status deployment/audra-api -n audra`

### 4. Verify

```bash
kubectl get pods -n audra
kubectl logs deployment/audra-api -n audra | tail -n 50
kubectl get ingress -n audra
```

> Screenshot: `assets/screenshots/audra-dashboard.png` (FastAPI docs page)

Troubleshooting:
- Ingress `Pending` -> annotate ingress with `alb.ingress.kubernetes.io/scheme: internet-facing` or ensure subnets tagged for ALB.
- 503 errors -> check service selectors and `kubectl describe ingress` for target health.

---

## Post-Deployment

### DNS and TLS

1. Create an ACM certificate in `$AWS_REGION` for your domain.
2. Annotate ingress with Certificate ARN:
   ```yaml
   alb.ingress.kubernetes.io/certificate-arn: arn:aws:acm:$AWS_REGION:$AWS_ACCOUNT_ID:certificate/...
   ```
3. Map Route 53 record to the ALB target DNS name.

### Monitoring

- Enable EKS add-on *Amazon CloudWatch Observability* via console or `aws eks update-cluster-config`.
- Install Prometheus/Grafana stack using `helm install kube-prometheus-stack prometheus-community/kube-prometheus-stack`.
- Create CloudWatch dashboards for node CPU, GPU utilization, and API latency.

### Log aggregation

```bash
aws logs create-log-group --log-group-name /aws/eks/audra/cluster
kubectl apply -f deployment/kubernetes/logging/fluent-bit.yaml
```

- Point Fluent Bit to CloudWatch Logs and (optional) OpenSearch for analytics.

### Operational checklist

- Confirm `/healthz` returns 200.
- Run smoke test script: `./scripts/test_deployment.sh --cluster audra --namespace audra`.
- Update runbook with ALB DNS, Route 53 record, and admin contacts.

---

## Scaling

### Horizontal Pod Autoscaler (HPA)

```bash
kubectl autoscale deployment audra-api \
  --namespace audra \
  --min=2 \
  --max=6 \
  --cpu-percent=60
```

- Tune for GPU utilization via custom metrics (e.g., DCGM exporter).

### Cluster/node autoscaling

- Ensure `cluster-autoscaler` has proper IAM permissions (`AutoScalingFullAccess`).
- Set nodegroup limits: `eksctl scale nodegroup --cluster audra --name gpu-workers --nodes 2` during peak.

### Cost optimization

- Use `eksctl scale nodegroup --nodes 0` to park GPU nodes when idle.
- Enable Graviton-based CPU node groups for system pods to cut costs.
- Turn off OpenSearch collection with `aws opensearchserverless delete-collection --id audra-guidelines` (recreate via script when needed).

---

## Backup & Recovery

### Vector database backups

```bash
aws opensearchserverless batch-get-collection --ids audra-guidelines
python scripts/export_guidelines.py --out s3://audra-backups/guidelines-$(date +%F).jsonl
```

- Schedule the export via AWS Lambda or EventBridge every 12 hours.

### Configuration backups

```bash
kubectl get all -n audra -o yaml > backups/audra-workloads-$(date +%F).yaml
kubectl get secrets -n audra -o yaml > backups/audra-secrets-$(date +%F).yaml
aws s3 sync backups/ s3://audra-backups/k8s/
```

### Disaster recovery plan

1. Recreate EKS cluster from `eksctl get cluster --name audra -o yaml > infra/cluster.yaml`.
2. Restore secrets from encrypted S3 bucket (KMS-protected).
3. Rebuild OpenSearch collection and import the exported guideline embeddings.
4. Redeploy application manifests and validate via smoke tests.

> Screenshot: `assets/screenshots/backup-dashboard.png`

---

## Troubleshooting Reference

| Issue | Symptom | Resolution |
|-------|---------|------------|
| IAM/RBAC policy missing | Ingress controller Pod `AccessDenied` | Re-verify service account permissions and IAM/RBAC bindings |
| GPU scheduling | Pods Pending with GPU requirements | Scale GPU node pools, confirm device plugin is running |
| Vector database auth | 403 response from vector store | Update access policy to include service accounts and compute roles |
| LLM latency | API requests >10s | Check endpoint health, increase replicas, or optimize model configuration |
| Budget overrun | Cloud console alert | Scale down node pools, stop non-essential services |

For unresolved issues, check application logs and cloud provider documentation for your specific platform.

---

Capture and drop screenshots mentioned above into `assets/screenshots/` so the README and this guide can embed them automatically.


