from qconstitution.articles import ConstitutionalArticle
from qconstitution.charter import default_charter
from qconstitution.upgrade import ConstitutionalUpgradeProposal, UpgradePath


def test_upgrade_proposal_applied_with_quorum():
    charter = default_charter()
    proposal = ConstitutionalUpgradeProposal(
        proposal_id="p1",
        proposer="nodeA",
        added_articles=[
            ConstitutionalArticle(
                article_id="integrity-proof",
                description="Proof obligations must be satisfied.",
                constraints={},
            )
        ],
        target_version="v2",
    )
    path = UpgradePath(threshold=2)
    new_charter = path.apply(charter, proposal, votes=["nodeA", "nodeB"])
    assert new_charter.active.version_id == "v2"
    assert "integrity-proof" in new_charter.article_index()
