terraform {
  required_version = ">= 1.4.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = ">= 2.20"
    }
    helm = {
      source  = "hashicorp/helm"
      version = ">= 2.9"
    }
  }
}

locals {
  project = "autonomous_systems-platform"
  tags = {
    Application = "QuASIM"
    Vertical    = "Autonomous Systems"
    Environment = var.environment
  }
}

provider "aws" {
  region = var.region
}

data "aws_caller_identity" "current" {}

module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "${local.project}-vpc"
  cidr = var.vpc_cidr

  azs             = var.availability_zones
  private_subnets = var.private_subnets
  public_subnets  = var.public_subnets

  enable_nat_gateway = true
  single_nat_gateway = true
  tags = local.tags
}

module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 20.8"

  cluster_name    = "${local.project}"
  cluster_version = var.kubernetes_version
  subnet_ids      = module.vpc.private_subnets
  vpc_id          = module.vpc.vpc_id

  eks_managed_node_groups = {
    general = {
      instance_types = [var.general_instance_type]
      desired_size   = 2
      min_size       = 1
      max_size       = 5
      labels = { role = "general" }
    }
    gpu = {
      instance_types = [var.gpu_instance_type]
      desired_size   = 1
      min_size       = 0
      max_size       = 4
      labels = { accelerator = "nvidia" }
      taints = [{ key = "accelerator", value = "nvidia", effect = "NO_SCHEDULE" }]
    }
  }

  tags = local.tags
}

resource "aws_ecr_repository" "backend" {
  name = "${local.project}-backend"
  image_tag_mutability = "MUTABLE"
  image_scanning_configuration { scan_on_push = true }
  tags = local.tags
}

resource "aws_security_group" "alb" {
  name        = "${local.project}-alb"
  description = "Allow HTTP/HTTPS"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = local.tags
}

resource "aws_iam_role" "service_account" {
  name = "${local.project}-sa"
  assume_role_policy = data.aws_iam_policy_document.eks_sa.json
  tags = local.tags
}

data "aws_iam_policy_document" "eks_sa" {
  statement {
    actions = ["sts:AssumeRole"]
    effect  = "Allow"
    principals {
      type        = "Federated"
      identifiers = [module.eks.oidc_provider_arn]
    }
    condition {
      test     = "StringEquals"
      variable = "${module.eks.oidc_provider}:sub"
      values   = ["system:serviceaccount/autonomous-systems/autonomous_systems-backend"]
    }
  }
}
