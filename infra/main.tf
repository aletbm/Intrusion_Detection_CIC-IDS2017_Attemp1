provider "google" {
  project = var.project
  region  = var.region
}

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

resource "google_cloud_run_service_iam_member" "noauth" {
  service = google_cloud_run_service.intrusion_api.name
  location = var.region
  role     = "roles/run.invoker"
  member   = "allUsers"
}

#terraform -chdir=infra init
#terraform -chdir=infra plan
#terraform -chdir=infra apply -auto-approve
