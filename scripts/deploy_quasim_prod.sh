#!/usr/bin/env bash
# Deploy QuASIM to the current kubectl context using the repo Helm chart and the production values file.
# Usage: ./scripts/deploy_quasim_prod.sh [kube_namespace] [helm_release_name]
set -euo pipefail

NAMESPACE="${1:-quasim-platform}"
RELEASE="${2:-quasim-platform}"
CHART_DIR="infra/helm/quasim-platform"
VALUES_FILE="${CHART_DIR}/values-prod.yaml"

command -v kubectl >/dev/null 2>&1 || { echo "kubectl required but not found"; exit 1; }
command -v helm >/dev/null 2>&1 || { echo "helm required but not found"; exit 1; }

echo "Using kube-context: $(kubectl config current-context)"
echo "Namespace: ${NAMESPACE}, Helm release: ${RELEASE}"

# Create namespace if missing
if ! kubectl get namespace "${NAMESPACE}" >/dev/null 2>&1; then
  echo "Creating namespace ${NAMESPACE}"
  kubectl create namespace "${NAMESPACE}"
else
  echo "Namespace ${NAMESPACE} already exists"
fi

# Ensure imagePullSecret exists (user must create it ahead of time if using a private registry)
echo "Make sure imagePullSecret (reg-credentials) exists in namespace ${NAMESPACE} if images are in a private registry."

# Deploy/upgrade with Helm
helm upgrade --install "${RELEASE}" "${CHART_DIR}" \
  --namespace "${NAMESPACE}" \
  --values "${VALUES_FILE}" \
  --wait

# Wait for deployments to finish rolling out
echo "Waiting for API deployment rollout..."
kubectl rollout status deployment/"${RELEASE}"-api -n "${NAMESPACE}" --timeout=5m || echo "API rollout timed out or not present"

echo "Waiting for QC worker rollout..."
kubectl rollout status deployment/"${RELEASE}"-qc-worker -n "${NAMESPACE}" --timeout=5m || echo "QC worker rollout timed out or not present"

echo "Deploy complete. Use 'kubectl get pods -n ${NAMESPACE}' to check pod status."
