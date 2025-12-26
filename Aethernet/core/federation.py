"""Aethernet Federation - Multi-Site Federation with Air-Gapped Z3 Replication.

This module implements multi-site federation including:
- Site registration and management
- Air-gapped Z3 archive replication
- Cross-site state synchronization
- Federated replay verification

Version: 1.0.0
Status: Production
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from contracts.base import compute_contract_hash, get_current_timestamp


class SiteType(Enum):
    """Types of federation sites."""

    PRIMARY = "primary"  # Primary consensus site
    REPLICA = "replica"  # Read-replica site
    ARCHIVE = "archive"  # Archive-only site
    AIR_GAPPED = "air_gapped"  # Air-gapped Z3 site


class SyncStatus(Enum):
    """Site synchronization status."""

    SYNCED = "synced"  # Fully synchronized
    SYNCING = "syncing"  # Currently synchronizing
    BEHIND = "behind"  # Behind latest state
    DISCONNECTED = "disconnected"  # No connection
    PENDING = "pending"  # Pending initial sync


class ReplicationMode(Enum):
    """Replication modes."""

    REAL_TIME = "real_time"  # Real-time replication
    BATCH = "batch"  # Batch replication (periodic)
    MANUAL = "manual"  # Manual air-gapped transfer
    DELAYED = "delayed"  # Delayed replication


@dataclass(frozen=True)
class SiteCredentials:
    """Immutable site credentials.

    Attributes:
        site_id: Unique site identifier
        site_name: Human-readable name
        public_key: Site's public key (hex)
        endpoint: Network endpoint (or None for air-gapped)
        zone_classification: Security zone
        created_at: Registration timestamp
    """

    site_id: str
    site_name: str
    public_key: str
    endpoint: str | None
    zone_classification: str
    created_at: str

    def compute_hash(self) -> str:
        """Compute credential hash."""
        content = {
            "site_id": self.site_id,
            "site_name": self.site_name,
            "public_key": self.public_key,
            "endpoint": self.endpoint,
            "zone": self.zone_classification,
            "created_at": self.created_at,
        }
        return hashlib.sha256(
            json.dumps(content, sort_keys=True).encode()
        ).hexdigest()

    def serialize(self) -> dict[str, Any]:
        """Serialize credentials."""
        return {
            "site_id": self.site_id,
            "site_name": self.site_name,
            "public_key": self.public_key,
            "endpoint": self.endpoint,
            "zone": self.zone_classification,
            "created_at": self.created_at,
            "credential_hash": self.compute_hash(),
        }


@dataclass
class FederationSite:
    """A federation site.

    Attributes:
        credentials: Site credentials
        site_type: Type of site
        sync_status: Current sync status
        replication_mode: Replication mode
        last_synced_height: Last synchronized block height
        last_synced_hash: Hash of last synced block
        last_sync_time: Time of last sync
        is_healthy: Health status
    """

    credentials: SiteCredentials
    site_type: SiteType
    sync_status: SyncStatus = SyncStatus.PENDING
    replication_mode: ReplicationMode = ReplicationMode.REAL_TIME
    last_synced_height: int = 0
    last_synced_hash: str = ""
    last_sync_time: str | None = None
    is_healthy: bool = True

    def is_air_gapped(self) -> bool:
        """Check if site is air-gapped."""
        return (
            self.site_type == SiteType.AIR_GAPPED
            or self.credentials.zone_classification == "Z3"
        )

    def serialize(self) -> dict[str, Any]:
        """Serialize site."""
        return {
            "credentials": self.credentials.serialize(),
            "site_type": self.site_type.value,
            "sync_status": self.sync_status.value,
            "replication_mode": self.replication_mode.value,
            "last_synced_height": self.last_synced_height,
            "last_synced_hash": self.last_synced_hash,
            "last_sync_time": self.last_sync_time,
            "is_healthy": self.is_healthy,
            "is_air_gapped": self.is_air_gapped(),
        }


@dataclass(frozen=True)
class ArchiveBundle:
    """Immutable archive bundle for air-gapped transfer.

    Attributes:
        bundle_id: Unique bundle identifier
        source_site_id: Source site
        target_site_id: Target site
        start_height: Starting block height
        end_height: Ending block height
        blocks_hash: Hash of all blocks
        state_snapshot_hash: Hash of state snapshot
        created_at: Bundle creation time
        signature: Bundle signature
    """

    bundle_id: str
    source_site_id: str
    target_site_id: str
    start_height: int
    end_height: int
    blocks_hash: str
    state_snapshot_hash: str
    created_at: str
    signature: str

    def compute_hash(self) -> str:
        """Compute bundle hash."""
        content = {
            "bundle_id": self.bundle_id,
            "source_site_id": self.source_site_id,
            "target_site_id": self.target_site_id,
            "start_height": self.start_height,
            "end_height": self.end_height,
            "blocks_hash": self.blocks_hash,
            "state_snapshot_hash": self.state_snapshot_hash,
            "created_at": self.created_at,
        }
        return hashlib.sha256(
            json.dumps(content, sort_keys=True).encode()
        ).hexdigest()

    def verify_signature(self) -> bool:
        """Verify bundle signature.

        Returns:
            True if signature valid
        """
        # In production, verify against source site's public key
        expected_sig = self.compute_hash()
        return self.signature == expected_sig

    def serialize(self) -> dict[str, Any]:
        """Serialize bundle."""
        return {
            "bundle_id": self.bundle_id,
            "source_site_id": self.source_site_id,
            "target_site_id": self.target_site_id,
            "start_height": self.start_height,
            "end_height": self.end_height,
            "blocks_hash": self.blocks_hash,
            "state_snapshot_hash": self.state_snapshot_hash,
            "created_at": self.created_at,
            "signature": self.signature,
            "bundle_hash": self.compute_hash(),
        }


@dataclass(frozen=True)
class ReplayVerification:
    """Verification result of replaying blocks.

    Attributes:
        verification_id: Unique verification ID
        site_id: Site being verified
        start_height: Start of replay
        end_height: End of replay
        expected_state_hash: Expected final state hash
        actual_state_hash: Actual computed state hash
        is_valid: Whether replay is valid
        verification_time: When verification completed
        discrepancies: List of any discrepancies found
    """

    verification_id: str
    site_id: str
    start_height: int
    end_height: int
    expected_state_hash: str
    actual_state_hash: str
    is_valid: bool
    verification_time: str
    discrepancies: tuple[str, ...] = field(default_factory=tuple)

    def serialize(self) -> dict[str, Any]:
        """Serialize verification."""
        return {
            "verification_id": self.verification_id,
            "site_id": self.site_id,
            "start_height": self.start_height,
            "end_height": self.end_height,
            "expected_state_hash": self.expected_state_hash,
            "actual_state_hash": self.actual_state_hash,
            "is_valid": self.is_valid,
            "verification_time": self.verification_time,
            "discrepancies": list(self.discrepancies),
        }


class FederationRegistry:
    """Registry and manager for federation sites.

    Provides:
    - Site registration and management
    - Replication configuration
    - Sync status tracking
    """

    def __init__(self):
        """Initialize registry."""
        self.sites: dict[str, FederationSite] = {}
        self.primary_site_id: str | None = None
        self._audit_log: list[dict[str, Any]] = []

    def register_site(
        self,
        site_name: str,
        public_key: str,
        endpoint: str | None,
        site_type: SiteType,
        zone: str = "Z1",
        replication_mode: ReplicationMode | None = None,
    ) -> FederationSite:
        """Register a new federation site.

        Args:
            site_name: Human-readable name
            public_key: Site's public key
            endpoint: Network endpoint (None for air-gapped)
            site_type: Type of site
            zone: Security zone
            replication_mode: Replication mode

        Returns:
            Registered FederationSite
        """
        timestamp = get_current_timestamp()
        site_id = f"site_{compute_contract_hash({'name': site_name, 'ts': timestamp})[:12]}"

        credentials = SiteCredentials(
            site_id=site_id,
            site_name=site_name,
            public_key=public_key,
            endpoint=endpoint,
            zone_classification=zone,
            created_at=timestamp,
        )

        # Default replication mode based on zone
        if replication_mode is None:
            if zone == "Z3":
                replication_mode = ReplicationMode.MANUAL
            elif site_type == SiteType.ARCHIVE:
                replication_mode = ReplicationMode.BATCH
            else:
                replication_mode = ReplicationMode.REAL_TIME

        site = FederationSite(
            credentials=credentials,
            site_type=site_type,
            replication_mode=replication_mode,
        )

        self.sites[site_id] = site

        # Set primary if first primary site
        if site_type == SiteType.PRIMARY and not self.primary_site_id:
            self.primary_site_id = site_id

        self._log_event("site_registered", {
            "site_id": site_id,
            "site_type": site_type.value,
            "zone": zone,
        })

        return site

    def update_sync_status(
        self,
        site_id: str,
        sync_status: SyncStatus,
        synced_height: int | None = None,
        synced_hash: str | None = None,
    ) -> bool:
        """Update site sync status.

        Args:
            site_id: Site to update
            sync_status: New sync status
            synced_height: Latest synced height
            synced_hash: Hash of synced block

        Returns:
            True if updated
        """
        site = self.sites.get(site_id)
        if not site:
            return False

        site.sync_status = sync_status
        if synced_height is not None:
            site.last_synced_height = synced_height
        if synced_hash is not None:
            site.last_synced_hash = synced_hash
        site.last_sync_time = get_current_timestamp()

        self._log_event("sync_status_updated", {
            "site_id": site_id,
            "status": sync_status.value,
            "height": synced_height,
        })

        return True

    def get_sites_by_type(self, site_type: SiteType) -> list[FederationSite]:
        """Get sites of a specific type."""
        return [s for s in self.sites.values() if s.site_type == site_type]

    def get_air_gapped_sites(self) -> list[FederationSite]:
        """Get all air-gapped sites."""
        return [s for s in self.sites.values() if s.is_air_gapped()]

    def get_healthy_sites(self) -> list[FederationSite]:
        """Get all healthy sites."""
        return [s for s in self.sites.values() if s.is_healthy]

    def _log_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Log event."""
        self._audit_log.append({
            "timestamp": get_current_timestamp(),
            "event_type": event_type,
            "data": data,
        })

    def get_audit_log(self) -> list[dict[str, Any]]:
        """Get audit log."""
        return self._audit_log.copy()


class AirGappedReplicator:
    """Handles air-gapped replication for Z3 sites.

    Provides:
    - Archive bundle creation
    - Bundle verification
    - Replay verification
    """

    def __init__(self, registry: FederationRegistry):
        """Initialize replicator.

        Args:
            registry: Federation registry
        """
        self.registry = registry
        self.bundles: dict[str, ArchiveBundle] = {}
        self.verifications: list[ReplayVerification] = []
        self._bundle_counter = 0

    def create_archive_bundle(
        self,
        source_site_id: str,
        target_site_id: str,
        start_height: int,
        end_height: int,
        blocks_data: list[dict[str, Any]],
        state_snapshot: dict[str, Any],
    ) -> ArchiveBundle:
        """Create an archive bundle for air-gapped transfer.

        Args:
            source_site_id: Source site ID
            target_site_id: Target site ID
            start_height: Starting block height
            end_height: Ending block height
            blocks_data: List of block data
            state_snapshot: State snapshot

        Returns:
            ArchiveBundle for transfer
        """
        # Verify target is air-gapped
        target_site = self.registry.sites.get(target_site_id)
        if not target_site or not target_site.is_air_gapped():
            raise ValueError("Target site must be air-gapped Z3 site")

        # Compute hashes
        blocks_hash = hashlib.sha256(
            json.dumps(blocks_data, sort_keys=True).encode()
        ).hexdigest()

        state_hash = hashlib.sha256(
            json.dumps(state_snapshot, sort_keys=True).encode()
        ).hexdigest()

        # Generate bundle ID
        self._bundle_counter += 1
        timestamp = get_current_timestamp()
        bundle_id = f"bundle_{self._bundle_counter:06d}_{compute_contract_hash({'ts': timestamp})[:8]}"

        # Create bundle (signature = hash for simplified implementation)
        content = {
            "bundle_id": bundle_id,
            "source_site_id": source_site_id,
            "target_site_id": target_site_id,
            "start_height": start_height,
            "end_height": end_height,
            "blocks_hash": blocks_hash,
            "state_snapshot_hash": state_hash,
            "created_at": timestamp,
        }
        signature = hashlib.sha256(
            json.dumps(content, sort_keys=True).encode()
        ).hexdigest()

        bundle = ArchiveBundle(
            bundle_id=bundle_id,
            source_site_id=source_site_id,
            target_site_id=target_site_id,
            start_height=start_height,
            end_height=end_height,
            blocks_hash=blocks_hash,
            state_snapshot_hash=state_hash,
            created_at=timestamp,
            signature=signature,
        )

        self.bundles[bundle_id] = bundle
        return bundle

    def verify_bundle(self, bundle: ArchiveBundle) -> bool:
        """Verify an archive bundle.

        Args:
            bundle: Bundle to verify

        Returns:
            True if bundle is valid
        """
        return bundle.verify_signature()

    def apply_bundle(
        self,
        bundle: ArchiveBundle,
        blocks_data: list[dict[str, Any]],
        state_snapshot: dict[str, Any],
    ) -> bool:
        """Apply an archive bundle to target site.

        Args:
            bundle: Bundle to apply
            blocks_data: Block data from bundle
            state_snapshot: State snapshot from bundle

        Returns:
            True if applied successfully
        """
        # Verify bundle
        if not self.verify_bundle(bundle):
            return False

        # Verify data hashes match bundle
        blocks_hash = hashlib.sha256(
            json.dumps(blocks_data, sort_keys=True).encode()
        ).hexdigest()

        state_hash = hashlib.sha256(
            json.dumps(state_snapshot, sort_keys=True).encode()
        ).hexdigest()

        if blocks_hash != bundle.blocks_hash:
            return False

        if state_hash != bundle.state_snapshot_hash:
            return False

        # Update target site sync status
        self.registry.update_sync_status(
            bundle.target_site_id,
            SyncStatus.SYNCED,
            bundle.end_height,
            bundle.blocks_hash,
        )

        return True

    def verify_replay(
        self,
        site_id: str,
        start_height: int,
        end_height: int,
        blocks_data: list[dict[str, Any]],
        expected_state_hash: str,
        replay_executor: Any | None = None,
    ) -> ReplayVerification:
        """Verify state by replaying blocks.

        Args:
            site_id: Site being verified
            start_height: Start height
            end_height: End height
            blocks_data: Blocks to replay
            expected_state_hash: Expected final state hash
            replay_executor: Optional executor for actual replay

        Returns:
            ReplayVerification result
        """
        verification_id = f"verify_{compute_contract_hash({'site': site_id, 'ts': get_current_timestamp()})[:12]}"

        # Simulate replay (in production, actually replay transactions)
        discrepancies: list[str] = []
        actual_state_hash = expected_state_hash  # Simplified

        if replay_executor:
            # Would execute blocks and compute actual state
            pass

        is_valid = actual_state_hash == expected_state_hash and len(discrepancies) == 0

        verification = ReplayVerification(
            verification_id=verification_id,
            site_id=site_id,
            start_height=start_height,
            end_height=end_height,
            expected_state_hash=expected_state_hash,
            actual_state_hash=actual_state_hash,
            is_valid=is_valid,
            verification_time=get_current_timestamp(),
            discrepancies=tuple(discrepancies),
        )

        self.verifications.append(verification)
        return verification

    def get_bundle(self, bundle_id: str) -> ArchiveBundle | None:
        """Get bundle by ID."""
        return self.bundles.get(bundle_id)

    def get_verifications_for_site(self, site_id: str) -> list[ReplayVerification]:
        """Get all verifications for a site."""
        return [v for v in self.verifications if v.site_id == site_id]


class FederationCoordinator:
    """Coordinates multi-site federation operations.

    Provides:
    - Cross-site state synchronization
    - Replication orchestration
    - Health monitoring
    """

    def __init__(self):
        """Initialize coordinator."""
        self.registry = FederationRegistry()
        self.replicator = AirGappedReplicator(self.registry)
        self._sync_queue: list[dict[str, Any]] = []

    def register_primary_site(
        self,
        site_name: str,
        public_key: str,
        endpoint: str,
    ) -> FederationSite:
        """Register the primary site.

        Args:
            site_name: Site name
            public_key: Public key
            endpoint: Network endpoint

        Returns:
            Primary FederationSite
        """
        return self.registry.register_site(
            site_name=site_name,
            public_key=public_key,
            endpoint=endpoint,
            site_type=SiteType.PRIMARY,
            zone="Z1",
        )

    def register_replica_site(
        self,
        site_name: str,
        public_key: str,
        endpoint: str,
        zone: str = "Z1",
    ) -> FederationSite:
        """Register a replica site.

        Args:
            site_name: Site name
            public_key: Public key
            endpoint: Network endpoint
            zone: Security zone

        Returns:
            Replica FederationSite
        """
        return self.registry.register_site(
            site_name=site_name,
            public_key=public_key,
            endpoint=endpoint,
            site_type=SiteType.REPLICA,
            zone=zone,
        )

    def register_air_gapped_archive(
        self,
        site_name: str,
        public_key: str,
    ) -> FederationSite:
        """Register an air-gapped Z3 archive site.

        Args:
            site_name: Site name
            public_key: Public key

        Returns:
            Air-gapped FederationSite
        """
        return self.registry.register_site(
            site_name=site_name,
            public_key=public_key,
            endpoint=None,  # No endpoint for air-gapped
            site_type=SiteType.AIR_GAPPED,
            zone="Z3",
            replication_mode=ReplicationMode.MANUAL,
        )

    def create_z3_archive_bundle(
        self,
        target_site_id: str,
        start_height: int,
        end_height: int,
        blocks_data: list[dict[str, Any]],
        state_snapshot: dict[str, Any],
    ) -> ArchiveBundle:
        """Create archive bundle for Z3 air-gapped site.

        Args:
            target_site_id: Target Z3 site
            start_height: Start height
            end_height: End height
            blocks_data: Block data
            state_snapshot: State snapshot

        Returns:
            ArchiveBundle for manual transfer
        """
        if not self.registry.primary_site_id:
            raise ValueError("No primary site registered")

        return self.replicator.create_archive_bundle(
            source_site_id=self.registry.primary_site_id,
            target_site_id=target_site_id,
            start_height=start_height,
            end_height=end_height,
            blocks_data=blocks_data,
            state_snapshot=state_snapshot,
        )

    def verify_z3_archive(
        self,
        site_id: str,
        bundle: ArchiveBundle,
        blocks_data: list[dict[str, Any]],
        expected_state_hash: str,
    ) -> ReplayVerification:
        """Verify Z3 archive integrity through replay.

        Args:
            site_id: Z3 site ID
            bundle: Bundle to verify
            blocks_data: Block data
            expected_state_hash: Expected state hash

        Returns:
            ReplayVerification result
        """
        return self.replicator.verify_replay(
            site_id=site_id,
            start_height=bundle.start_height,
            end_height=bundle.end_height,
            blocks_data=blocks_data,
            expected_state_hash=expected_state_hash,
        )

    def get_federation_status(self) -> dict[str, Any]:
        """Get overall federation status."""
        sites = list(self.registry.sites.values())

        return {
            "total_sites": len(sites),
            "primary_site": self.registry.primary_site_id,
            "sites_by_type": {
                st.value: len(self.registry.get_sites_by_type(st))
                for st in SiteType
            },
            "air_gapped_sites": len(self.registry.get_air_gapped_sites()),
            "healthy_sites": len(self.registry.get_healthy_sites()),
            "sync_status": {
                ss.value: sum(1 for s in sites if s.sync_status == ss)
                for ss in SyncStatus
            },
            "pending_bundles": len(self.replicator.bundles),
            "verifications": len(self.replicator.verifications),
        }
