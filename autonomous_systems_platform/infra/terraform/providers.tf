provider "kubernetes" {
  host                   = module.eks.cluster_endpoint
  cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)
  token                  = data.aws_eks_cluster_auth.token.token
}

data "aws_eks_cluster_auth" "token" {
  name = module.eks.cluster_name
}
