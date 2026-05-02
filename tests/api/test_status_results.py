"""Tests for status and results endpoints."""


class TestStatusEndpoints:
    """Tests for status monitoring endpoints."""

    def test_get_job_status(self, api_client, auth_headers):
        """Test getting job status."""
        # Submit a job
        submit_response = api_client.post(
            "/v1/jobs",
            headers=auth_headers,
            json={
                "job_type": "vqe",
                "config": {"molecule": "H2"},
            },
        )
        job_id = submit_response.json()["job_id"]

        # Get status
        response = api_client.get(f"/v1/jobs/{job_id}/status", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == job_id
        assert data["status"] == "queued"
        assert "progress" in data

    def test_get_job_status_not_found(self, api_client, auth_headers):
        """Test getting status for non-existent job."""
        response = api_client.get(
            "/v1/jobs/00000000-0000-0000-0000-000000000000/status",
            headers=auth_headers,
        )
        assert response.status_code == 404


class TestResultsEndpoints:
    """Tests for results retrieval endpoints."""

    def test_get_job_results_not_completed(self, api_client, auth_headers):
        """Test getting results for incomplete job."""
        # Submit a job
        submit_response = api_client.post(
            "/v1/jobs",
            headers=auth_headers,
            json={
                "job_type": "vqe",
                "config": {"molecule": "H2"},
            },
        )
        job_id = submit_response.json()["job_id"]

        # Try to get results (job is still queued)
        response = api_client.get(f"/v1/jobs/{job_id}/results", headers=auth_headers)
        assert response.status_code == 400

    def test_list_job_artifacts(self, api_client, auth_headers):
        """Test listing job artifacts."""
        # Submit a job
        submit_response = api_client.post(
            "/v1/jobs",
            headers=auth_headers,
            json={
                "job_type": "vqe",
                "config": {"molecule": "H2"},
            },
        )
        job_id = submit_response.json()["job_id"]

        # List artifacts (will be empty or mock for incomplete job)
        response = api_client.get(f"/v1/jobs/{job_id}/artifacts", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "artifacts" in data


class TestResourcesEndpoints:
    """Tests for resource allocation endpoints."""

    def test_get_resources_dashboard(self, api_client, auth_headers):
        """Test getting resource dashboard."""
        response = api_client.get("/v1/resources", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "timestamp" in data
        assert "clusters" in data
        assert "queue_depth" in data
        assert "utilization" in data

    def test_list_clusters(self, api_client, auth_headers):
        """Test listing clusters."""
        response = api_client.get("/v1/resources/clusters", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "clusters" in data
        assert len(data["clusters"]) > 0
        # Check cluster structure
        cluster = data["clusters"][0]
        assert "cluster_id" in cluster
        assert "provider" in cluster
        assert "region" in cluster
        assert "status" in cluster

    def test_get_quotas(self, api_client, auth_headers):
        """Test getting quotas."""
        response = api_client.get("/v1/resources/quotas", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "quotas" in data
        assert "usage" in data
        assert "max_concurrent_jobs" in data["quotas"]
        assert "concurrent_jobs" in data["usage"]

    def test_resources_unauthorized(self, api_client):
        """Test resources endpoints without authentication."""
        response = api_client.get("/v1/resources")
        assert response.status_code == 401
