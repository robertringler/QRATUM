output "cluster_name" {
  value = module.eks.cluster_name
}

output "cluster_endpoint" {
  value = module.eks.cluster_endpoint
}

output "repository_url" {
  value = aws_ecr_repository.backend.repository_url
}
