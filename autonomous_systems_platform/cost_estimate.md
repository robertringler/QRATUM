# Cost Estimate

| Tier | Assumptions | Monthly Estimate |
| ---- | ----------- | ---------------- |
| Low  | 1 x m6i.xlarge control plane, 1 x g5.xlarge GPU node (spot), 100 GB EBS, 100 GB S3 | ~$2,300 |
| Medium | 3 x m6i.2xlarge, 2 x p4d.24xlarge, 1 TB EBS, 1 TB S3, AWS Load Balancer | ~$45,000 |
| High | 5 x m6i.4xlarge, 4 x p5.48xlarge, 2 TB EBS, 5 TB S3, Enterprise support | ~$180,000 |

*Azure*: AKS with Standard_D8s_v5 and ND A100 v4 nodes (≈10% premium).

*GCP*: GKE with n2-standard-8 and a2-ultragpu-2g nodes (≈5% discount).
