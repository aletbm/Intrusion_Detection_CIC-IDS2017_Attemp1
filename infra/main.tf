provider "google" {
  project = var.project
  region  = var.region
}

# Cloud Run service
resource "google_cloud_run_service" "intrusion_api" {
  name     = "intrusion-api"
  location = var.region

  template {
    spec {
      containers {
        image = "gcr.io/${var.project}/intrusion-api"
        ports {
          container_port = 8080
        }
      }
    }
  }

  traffic {
    percent         = 100
    latest_revision = true
  }
}

# IAM for public access to the Cloud Run service
resource "google_cloud_run_service_iam_member" "noauth" {
  service  = google_cloud_run_service.intrusion_api.name
  location = var.region
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# Bucket for storing prediction outputs
resource "google_storage_bucket" "predictions" {
  name          = "${var.project}-predictions"
  location      = var.region
  force_destroy = true
  uniform_bucket_level_access = true

  lifecycle_rule {
    action {
      type = "Delete"
    }
    condition {
      age = 30
    }
  }
}

# Bucket for MLflow model registry and artifacts
resource "google_storage_bucket" "mlflow_models" {
  name          = "${var.project}-mlflow-models"
  location      = var.region
  force_destroy = true
  uniform_bucket_level_access = true

  lifecycle_rule {
    action {
      type = "Delete"
    }
    condition {
      age = 90
    }
  }
}

#terraform -chdir=infra init
#terraform -chdir=infra plan
#terraform -chdir=infra apply -auto-approve
