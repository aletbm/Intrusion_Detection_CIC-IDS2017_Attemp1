provider "google" {
  project = var.project_id
  region  = var.region
}

resource "google_storage_bucket" "model_bucket" {
  name     = "${var.project_id}-mlflow-models"
  location = var.region
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
