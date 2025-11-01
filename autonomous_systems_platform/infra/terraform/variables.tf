variable "region" { type = string }
variable "environment" { type = string default = "dev" }
variable "kubernetes_version" { type = string default = "1.29" }
variable "vpc_cidr" { type = string default = "10.50.0.0/16" }
variable "availability_zones" { type = list(string) }
variable "public_subnets" { type = list(string) }
variable "private_subnets" { type = list(string) }
variable "general_instance_type" { type = string default = "m6i.xlarge" }
variable "gpu_instance_type" { type = string default = "p4d.24xlarge" }
