import pytest

from qconstitution.articles import ArticleSet, ConstitutionalArticle
from qconstitution.charter import Charter, ConstitutionalVersion
from qconstitution.upgrade import ConstitutionalUpgradeProposal, UpgradePath
from qconstitution.validator import ValidationError, validate_node_config
from qnode.config import NodeConfig


def test_constitution_change_enforces_new_constraints():
    base_articles = ArticleSet(
        [
            ConstitutionalArticle(
                article_id="safety-envelope",
                description="Nodes must declare syscalls",
                constraints={"scope": ["node"], "requires_allowed_syscalls": True},
            )
        ]
    )
    charter = Charter(
        active=ConstitutionalVersion(version_id="v0", enacted_tick=0, articles=base_articles)
    )
    config = NodeConfig(node_id="n1", identity_ref="id", allowed_syscalls=["read"])
    validate_node_config(config, charter=charter, ledger_enabled=False)

    proposal = ConstitutionalUpgradeProposal(
        proposal_id="p-ledger",
        proposer="council",
        added_articles=[
            ConstitutionalArticle(
                article_id="audit-ledger",
                description="Ledger must be enabled",
                constraints={"scope": ["node"], "requires_ledger": True},
            )
        ],
        target_version="v1",
    )
    upgraded = UpgradePath(threshold=1).apply(charter, proposal, votes=["council"])
    with pytest.raises(ValidationError):
        validate_node_config(config, charter=upgraded, ledger_enabled=False)
    validate_node_config(config, charter=upgraded, ledger_enabled=True)
