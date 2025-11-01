# QuASIM Infrastructure — Stage I

This guide brings up a production-ready EKS cluster with GPU node groups and installs platform components via ArgoCD (app-of-apps).

## Prereqs
- Terraform ≥ 1.7, kubectl ≥ 1.29, helm ≥ 3.14, aws CLI configured
- Route53 hosted zone (or external DNS option), ACM for public certs (optional)

## Quick Start
1) **Provision VPC+EKS**
```bash
cd infra/terraform/eks
terraform init && terraform apply -var-file=prod.tfvars
aws eks update-kubeconfig --name $(terraform output -raw cluster_name) --region $(terraform output -raw region)
```

2. **Install ArgoCD (bootstrap)**

```bash
kubectl create namespace argocd || true
helm repo add argo https://argoproj.github.io/argo-helm
helm upgrade --install argocd argo/argo-cd \
  -n argocd -f ../../helm/values/argocd-values.yaml
```

3. **App-of-Apps (GitOps all platform services)**

```bash
kubectl apply -n argocd -f ../../helm/argocd-apps/projects.yaml
kubectl apply -n argocd -f ../../helm/argocd-apps/app-of-apps.yaml
```

4. **Verify**

* `kubectl -n monitoring get pods` → Prometheus/Grafana up
* `kubectl -n core get pods` → Cilium ready, cert-manager ready
* `kubectl -n security get pods` → Gatekeeper/Vault ready

## Validation & Testing

Run repository sanity checks locally before opening a pull request:

```bash
make test
```

The harness parses all Helm/Kustomize YAML manifests to ensure syntactic
correctness and, when the Terraform CLI is available, also runs `terraform init`
with the backend disabled followed by `terraform validate` for each module. Any
missing tooling is reported as a skipped check so contributors on lightweight
environments still receive actionable feedback.

## Namespaces

* `core`: CNI, cert-manager, ingress
* `mlops`: pipelines, registries (later stages)
* `inference`: KServe/Triton/vLLM (later stages)
* `monitoring`: Prometheus/Grafana/Loki/Tempo
* `security`: Vault, Gatekeeper

## GPU Scheduling

GPU nodes are tainted and labeled:

```
nodeSelector: { accelerator: nvidia }
tolerations:
- key: "accelerator"
  operator: "Equal"
  value: "nvidia"
  effect: "NoSchedule"
```

(AMD uses `accelerator: amd`)

## Diagram

```mermaid
flowchart LR
  dev[Git (main)] -->|ArgoCD| k8s[Kubernetes/EKS]
  subgraph k8s
    CNI[Cilium]
    GITOPS[ArgoCD]
    SEC[Vault + Gatekeeper]
    TLS[cert-manager]
    ING[Ingress NGINX]
    MON[Prometheus/Grafana/Loki/Tempo]
  end
  users[QuASIM Teams] --> ING
  MON <-- metrics/logs/traces --> k8s
```
