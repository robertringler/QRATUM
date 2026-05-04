"""Tests for job management endpoints."""


class TestJobEndpoints:
    """Tests for job management endpoints."""

    def test_submit_job_vqe(self, api_client, auth_headers):
        """Test submitting a VQE job."""
        response = api_client.post(
            "/v1/jobs",
            headers=auth_headers,
            json={
                "job_type": "vqe",
                "name": "Test VQE Job",
                "config": {
                    "molecule": "H2",
                    "bond_length": 0.735,
                    "basis": "sto-3g",
                    "max_iterations": 100,
                },
                "priority": 5,
                "timeout_seconds": 3600,
                "tags": ["test"],
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "queued"
        assert "submitted_at" in data

    def test_submit_job_quantum_circuit(self, api_client, auth_headers):
        """Test submitting a quantum circuit job."""
        response = api_client.post(
            "/v1/jobs",
            headers=auth_headers,
            json={
                "job_type": "quantum_circuit",
                "config": {
                    "circuit": [["H", 0], ["CNOT", 0, 1]],
                    "shots": 1024,
                },
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert data["status"] == "queued"

    def test_submit_job_api_key(self, api_client, api_key_headers):
        """Test submitting a job with API key authentication."""
        response = api_client.post(
            "/v1/jobs",
            headers=api_key_headers,
            json={
                "job_type": "qaoa",
                "config": {
                    "problem": "maxcut",
                    "p_layers": 3,
                },
            },
        )
        assert response.status_code == 201

    def test_submit_job_invalid_config(self, api_client, auth_headers):
        """Test submitting a job with invalid configuration."""
        response = api_client.post(
            "/v1/jobs",
            headers=auth_headers,
            json={
                "job_type": "vqe",
                "config": {},  # Missing required 'molecule' field
            },
        )
        assert response.status_code == 400

    def test_submit_job_unauthorized(self, api_client):
        """Test submitting a job without authentication."""
        response = api_client.post(
            "/v1/jobs",
            json={
                "job_type": "vqe",
                "config": {"molecule": "H2"},
            },
        )
        assert response.status_code == 401

    def test_list_jobs(self, api_client, auth_headers):
        """Test listing jobs."""
        # First submit a job
        api_client.post(
            "/v1/jobs",
            headers=auth_headers,
            json={
                "job_type": "vqe",
                "config": {"molecule": "H2"},
            },
        )

        # List jobs
        response = api_client.get("/v1/jobs", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "jobs" in data
        assert "total" in data
        assert "limit" in data
        assert "offset" in data

    def test_list_jobs_with_filter(self, api_client, auth_headers):
        """Test listing jobs with status filter."""
        response = api_client.get("/v1/jobs?status=queued", headers=auth_headers)
        assert response.status_code == 200

    def test_get_job(self, api_client, auth_headers):
        """Test getting job details."""
        # Submit a job
        submit_response = api_client.post(
            "/v1/jobs",
            headers=auth_headers,
            json={
                "job_type": "qaoa",
                "name": "Test QAOA",
                "config": {"problem": "maxcut"},
            },
        )
        job_id = submit_response.json()["job_id"]

        # Get job details
        response = api_client.get(f"/v1/jobs/{job_id}", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == job_id
        assert data["name"] == "Test QAOA"
        assert data["job_type"] == "qaoa"

    def test_get_job_not_found(self, api_client, auth_headers):
        """Test getting non-existent job."""
        response = api_client.get(
            "/v1/jobs/00000000-0000-0000-0000-000000000000", headers=auth_headers
        )
        assert response.status_code == 404

    def test_cancel_job(self, api_client, auth_headers):
        """Test cancelling a job."""
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

        # Cancel the job
        response = api_client.delete(f"/v1/jobs/{job_id}", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "cancelled"

    def test_validate_job(self, api_client, auth_headers):
        """Test job validation."""
        response = api_client.post(
            "/v1/validate",
            headers=auth_headers,
            json={
                "job_type": "vqe",
                "config": {"molecule": "H2"},
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is True

    def test_validate_job_invalid(self, api_client, auth_headers):
        """Test job validation with invalid config."""
        response = api_client.post(
            "/v1/validate",
            headers=auth_headers,
            json={
                "job_type": "vqe",
                "config": {},  # Missing molecule
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is False
        assert len(data["errors"]) > 0
