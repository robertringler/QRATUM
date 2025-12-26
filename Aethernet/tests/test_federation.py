"""Tests for Aethernet Federation Module."""

import pytest

from Aethernet.core.federation import (
    AirGappedReplicator,
    FederationCoordinator,
    FederationRegistry,
    FederationSite,
    ReplicationMode,
    SiteCredentials,
    SiteType,
    SyncStatus,
)


class TestSiteCredentials:
    """Tests for SiteCredentials."""

    def test_credential_creation(self):
        """Test creating credentials."""
        creds = SiteCredentials(
            site_id="site_001",
            site_name="Primary Site",
            public_key="pk_abc",
            endpoint="https://primary.example.com",
            zone_classification="Z1",
            created_at="2025-01-01T00:00:00Z",
        )
        assert creds.site_id == "site_001"
        assert creds.zone_classification == "Z1"

    def test_credential_hash(self):
        """Test credential hash is deterministic."""
        creds1 = SiteCredentials("s1", "name", "pk", "ep", "Z1", "2025-01-01T00:00:00Z")
        creds2 = SiteCredentials("s1", "name", "pk", "ep", "Z1", "2025-01-01T00:00:00Z")
        assert creds1.compute_hash() == creds2.compute_hash()

    def test_credential_serialization(self):
        """Test credential serialization."""
        creds = SiteCredentials("s1", "name", "pk", "ep", "Z1", "2025-01-01T00:00:00Z")
        serialized = creds.serialize()
        assert "credential_hash" in serialized
        assert serialized["site_id"] == "s1"


class TestFederationSite:
    """Tests for FederationSite."""

    def test_site_creation(self):
        """Test creating a site."""
        creds = SiteCredentials("s1", "Site 1", "pk", "ep", "Z1", "2025-01-01T00:00:00Z")
        site = FederationSite(
            credentials=creds,
            site_type=SiteType.PRIMARY,
        )
        assert site.sync_status == SyncStatus.PENDING
        assert site.is_air_gapped() is False

    def test_air_gapped_site(self):
        """Test air-gapped site detection."""
        creds = SiteCredentials("s1", "Z3 Archive", "pk", None, "Z3", "2025-01-01T00:00:00Z")
        site = FederationSite(
            credentials=creds,
            site_type=SiteType.AIR_GAPPED,
        )
        assert site.is_air_gapped() is True

    def test_z3_is_air_gapped(self):
        """Test Z3 zone is treated as air-gapped."""
        creds = SiteCredentials("s1", "Z3 Site", "pk", "ep", "Z3", "2025-01-01T00:00:00Z")
        site = FederationSite(
            credentials=creds,
            site_type=SiteType.ARCHIVE,
        )
        assert site.is_air_gapped() is True

    def test_site_serialization(self):
        """Test site serialization."""
        creds = SiteCredentials("s1", "Site 1", "pk", "ep", "Z1", "2025-01-01T00:00:00Z")
        site = FederationSite(
            credentials=creds,
            site_type=SiteType.PRIMARY,
        )
        serialized = site.serialize()
        assert "credentials" in serialized
        assert serialized["site_type"] == "primary"


class TestFederationRegistry:
    """Tests for FederationRegistry."""

    def test_registry_creation(self):
        """Test creating registry."""
        registry = FederationRegistry()
        assert len(registry.sites) == 0
        assert registry.primary_site_id is None

    def test_register_site(self):
        """Test registering a site."""
        registry = FederationRegistry()
        site = registry.register_site(
            site_name="Primary",
            public_key="pk_primary",
            endpoint="https://primary.example.com",
            site_type=SiteType.PRIMARY,
        )
        assert site is not None
        assert site.site_type == SiteType.PRIMARY
        assert registry.primary_site_id == site.credentials.site_id

    def test_z3_site_manual_replication(self):
        """Test Z3 site gets manual replication mode."""
        registry = FederationRegistry()
        site = registry.register_site(
            site_name="Z3 Archive",
            public_key="pk_z3",
            endpoint=None,
            site_type=SiteType.AIR_GAPPED,
            zone="Z3",
        )
        assert site.replication_mode == ReplicationMode.MANUAL

    def test_update_sync_status(self):
        """Test updating sync status."""
        registry = FederationRegistry()
        site = registry.register_site("Site1", "pk", "ep", SiteType.REPLICA)
        result = registry.update_sync_status(
            site.credentials.site_id,
            SyncStatus.SYNCED,
            synced_height=100,
            synced_hash="hash_100",
        )
        assert result is True
        assert site.sync_status == SyncStatus.SYNCED
        assert site.last_synced_height == 100

    def test_get_sites_by_type(self):
        """Test getting sites by type."""
        registry = FederationRegistry()
        registry.register_site("P1", "pk1", "ep1", SiteType.PRIMARY)
        registry.register_site("R1", "pk2", "ep2", SiteType.REPLICA)
        registry.register_site("R2", "pk3", "ep3", SiteType.REPLICA)

        replicas = registry.get_sites_by_type(SiteType.REPLICA)
        assert len(replicas) == 2

    def test_get_air_gapped_sites(self):
        """Test getting air-gapped sites."""
        registry = FederationRegistry()
        registry.register_site("P1", "pk1", "ep1", SiteType.PRIMARY, zone="Z1")
        registry.register_site("A1", "pk2", None, SiteType.AIR_GAPPED, zone="Z3")

        air_gapped = registry.get_air_gapped_sites()
        assert len(air_gapped) == 1

    def test_get_healthy_sites(self):
        """Test getting healthy sites."""
        registry = FederationRegistry()
        site1 = registry.register_site("S1", "pk1", "ep1", SiteType.PRIMARY)
        site2 = registry.register_site("S2", "pk2", "ep2", SiteType.REPLICA)
        site2.is_healthy = False

        healthy = registry.get_healthy_sites()
        assert len(healthy) == 1
        assert healthy[0].credentials.site_id == site1.credentials.site_id


class TestAirGappedReplicator:
    """Tests for AirGappedReplicator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.registry = FederationRegistry()
        self.primary = self.registry.register_site(
            "Primary", "pk_primary", "ep", SiteType.PRIMARY, zone="Z1"
        )
        self.z3_site = self.registry.register_site(
            "Z3 Archive", "pk_z3", None, SiteType.AIR_GAPPED, zone="Z3"
        )
        self.replicator = AirGappedReplicator(self.registry)

    def test_create_archive_bundle(self):
        """Test creating archive bundle."""
        blocks = [{"height": 1}, {"height": 2}]
        state = {"key": "value"}

        bundle = self.replicator.create_archive_bundle(
            source_site_id=self.primary.credentials.site_id,
            target_site_id=self.z3_site.credentials.site_id,
            start_height=1,
            end_height=2,
            blocks_data=blocks,
            state_snapshot=state,
        )

        assert bundle is not None
        assert bundle.start_height == 1
        assert bundle.end_height == 2

    def test_create_bundle_requires_z3_target(self):
        """Test bundle creation requires Z3 target."""
        replica = self.registry.register_site("Replica", "pk_r", "ep", SiteType.REPLICA, zone="Z1")

        with pytest.raises(ValueError):
            self.replicator.create_archive_bundle(
                self.primary.credentials.site_id,
                replica.credentials.site_id,
                1,
                2,
                [],
                {},
            )

    def test_verify_bundle(self):
        """Test bundle verification."""
        blocks = [{"height": 1}]
        state = {"key": "value"}

        bundle = self.replicator.create_archive_bundle(
            self.primary.credentials.site_id,
            self.z3_site.credentials.site_id,
            1,
            1,
            blocks,
            state,
        )

        assert self.replicator.verify_bundle(bundle) is True

    def test_apply_bundle(self):
        """Test applying bundle."""
        blocks = [{"height": 1}]
        state = {"key": "value"}

        bundle = self.replicator.create_archive_bundle(
            self.primary.credentials.site_id,
            self.z3_site.credentials.site_id,
            1,
            1,
            blocks,
            state,
        )

        result = self.replicator.apply_bundle(bundle, blocks, state)
        assert result is True
        assert self.z3_site.sync_status == SyncStatus.SYNCED

    def test_apply_bundle_wrong_data(self):
        """Test applying bundle with wrong data fails."""
        blocks = [{"height": 1}]
        state = {"key": "value"}

        bundle = self.replicator.create_archive_bundle(
            self.primary.credentials.site_id,
            self.z3_site.credentials.site_id,
            1,
            1,
            blocks,
            state,
        )

        wrong_blocks = [{"height": 2}]  # Different data
        result = self.replicator.apply_bundle(bundle, wrong_blocks, state)
        assert result is False

    def test_verify_replay(self):
        """Test replay verification."""
        verification = self.replicator.verify_replay(
            site_id=self.z3_site.credentials.site_id,
            start_height=1,
            end_height=10,
            blocks_data=[],
            expected_state_hash="expected_hash",
        )

        assert verification is not None
        assert verification.is_valid is True

    def test_get_verifications_for_site(self):
        """Test getting verifications for a site."""
        self.replicator.verify_replay(
            self.z3_site.credentials.site_id,
            1,
            10,
            [],
            "hash1",
        )
        self.replicator.verify_replay(
            self.z3_site.credentials.site_id,
            11,
            20,
            [],
            "hash2",
        )

        verifications = self.replicator.get_verifications_for_site(self.z3_site.credentials.site_id)
        assert len(verifications) == 2


class TestFederationCoordinator:
    """Tests for FederationCoordinator."""

    def test_coordinator_creation(self):
        """Test creating coordinator."""
        coordinator = FederationCoordinator()
        assert coordinator.registry is not None
        assert coordinator.replicator is not None

    def test_register_primary_site(self):
        """Test registering primary site."""
        coordinator = FederationCoordinator()
        site = coordinator.register_primary_site(
            site_name="Primary",
            public_key="pk_primary",
            endpoint="https://primary.example.com",
        )
        assert site.site_type == SiteType.PRIMARY
        assert coordinator.registry.primary_site_id == site.credentials.site_id

    def test_register_replica_site(self):
        """Test registering replica site."""
        coordinator = FederationCoordinator()
        coordinator.register_primary_site("P", "pk", "ep")
        replica = coordinator.register_replica_site(
            site_name="Replica",
            public_key="pk_replica",
            endpoint="https://replica.example.com",
        )
        assert replica.site_type == SiteType.REPLICA

    def test_register_air_gapped_archive(self):
        """Test registering air-gapped archive."""
        coordinator = FederationCoordinator()
        archive = coordinator.register_air_gapped_archive(
            site_name="Z3 Archive",
            public_key="pk_z3",
        )
        assert archive.site_type == SiteType.AIR_GAPPED
        assert archive.credentials.zone_classification == "Z3"
        assert archive.replication_mode == ReplicationMode.MANUAL

    def test_create_z3_archive_bundle(self):
        """Test creating Z3 archive bundle."""
        coordinator = FederationCoordinator()
        coordinator.register_primary_site("P", "pk_p", "ep")
        archive = coordinator.register_air_gapped_archive("Z3", "pk_z3")

        bundle = coordinator.create_z3_archive_bundle(
            target_site_id=archive.credentials.site_id,
            start_height=1,
            end_height=100,
            blocks_data=[{"height": i} for i in range(1, 101)],
            state_snapshot={"root": "hash"},
        )

        assert bundle is not None
        assert bundle.target_site_id == archive.credentials.site_id

    def test_verify_z3_archive(self):
        """Test verifying Z3 archive."""
        coordinator = FederationCoordinator()
        coordinator.register_primary_site("P", "pk_p", "ep")
        archive = coordinator.register_air_gapped_archive("Z3", "pk_z3")

        blocks = [{"height": 1}]
        bundle = coordinator.create_z3_archive_bundle(
            archive.credentials.site_id,
            1,
            1,
            blocks,
            {"root": "hash"},
        )

        verification = coordinator.verify_z3_archive(
            archive.credentials.site_id,
            bundle,
            blocks,
            "expected_hash",
        )

        assert verification is not None
        assert verification.site_id == archive.credentials.site_id

    def test_federation_status(self):
        """Test federation status reporting."""
        coordinator = FederationCoordinator()
        coordinator.register_primary_site("P", "pk_p", "ep")
        coordinator.register_replica_site("R1", "pk_r1", "ep")
        coordinator.register_air_gapped_archive("Z3", "pk_z3")

        status = coordinator.get_federation_status()
        assert status["total_sites"] == 3
        assert status["air_gapped_sites"] == 1
        assert status["sites_by_type"]["primary"] == 1
        assert status["sites_by_type"]["replica"] == 1
        assert status["sites_by_type"]["air_gapped"] == 1
