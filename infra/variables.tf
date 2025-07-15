variable "project" {
  description = "Google Cloud project ID"
  type        = string
}

variable "region" {
  description = "Google Cloud region"
  type        = string
  default     = "us-east1"
}

variable "service_name" {
  description = "Name of the Cloud Run service"
  type        = string
  default     = "intrusion-api"
}
