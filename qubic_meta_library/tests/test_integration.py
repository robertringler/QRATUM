"""Integration tests for Qubic Meta Library."""

from qubic_meta_library.services import (Dashboard, ExecutionEngine,
                                         PatentAnalyzer, PromptLoader,
                                         SynergyMapper)


class TestIntegration:
    """Integration tests for full workflow."""

    def test_full_workflow(self):
        """Test complete workflow from loading to dashboard generation."""

        # Step 1: Load data
        loader = PromptLoader()
        domains = loader.load_domains()
        prompts = loader.load_all_prompts()

        assert len(domains) == 20
        assert len(prompts) > 0

        # Step 2: Map synergies
        mapper = SynergyMapper()
        clusters = mapper.load_clusters()
        synergies = mapper.find_synergies(prompts, domains)

        assert len(clusters) > 0
        assert len(synergies) > 0

        # Step 3: Analyze patents
        analyzer = PatentAnalyzer()
        high_value = analyzer.extract_high_value_prompts(prompts)
        patent_report = analyzer.generate_patent_pipeline_report(prompts)

        assert len(high_value) > 0
        assert patent_report["total_prompts"] == len(prompts)

        # Step 4: Setup execution
        engine = ExecutionEngine()
        pipelines = engine.load_pipelines()
        assignments = engine.assign_prompts_to_pipelines(prompts)

        assert len(pipelines) > 0
        assert len(assignments) > 0

        # Step 5: Generate dashboard
        dash = Dashboard()
        kpis = dash.calculate_kpis(prompts, domains, clusters, pipelines)

        assert "prompt_metrics" in kpis
        assert "domain_metrics" in kpis
        assert "cluster_metrics" in kpis
        assert "pipeline_metrics" in kpis

        # Verify executive summary generation
        summary = dash.generate_executive_summary(prompts, domains, clusters, pipelines)
        assert "Qubic Meta Library Executive Summary" in summary

    def test_high_value_workflow(self):
        """Test workflow focused on high-value prompts."""

        # Load data
        loader = PromptLoader()
        loader.load_domains()
        prompts = loader.load_all_prompts()

        # Get high-value prompts
        high_value = loader.get_high_value_prompts(threshold=0.85)

        # Analyze for patents
        analyzer = PatentAnalyzer()
        patent_clusters = analyzer.identify_patent_clusters(prompts)
        cross_domain = analyzer.analyze_cross_domain_opportunities(prompts)

        assert len(high_value) > 0
        assert len(patent_clusters) > 0
        assert len(cross_domain) >= 0

        # Verify high-value prompts have good scores
        for prompt in high_value:
            assert prompt.patentability_score >= 0.85
            assert prompt.commercial_potential >= 0.85

    def test_synergy_workflow(self):
        """Test workflow focused on synergy mapping."""

        # Load data
        loader = PromptLoader()
        loader.load_domains()
        loader.load_all_prompts()

        # Map synergies
        mapper = SynergyMapper()
        clusters = mapper.load_clusters()

        # Get clusters by type
        two_domain = mapper.get_clusters_by_type("two-domain")
        multi_domain = mapper.get_clusters_by_type("multi-domain")
        full_stack = mapper.get_clusters_by_type("full-stack")

        assert len(two_domain) > 0
        assert len(multi_domain) > 0
        assert len(full_stack) >= 1

        # Generate cluster report
        report = mapper.generate_cluster_report()
        assert report["total_clusters"] == len(clusters)
        assert report["total_revenue_projection"] > 0

    def test_execution_workflow(self):
        """Test workflow focused on pipeline execution."""

        # Load data
        loader = PromptLoader()
        loader.load_domains()
        prompts = loader.load_all_prompts()

        # Setup execution
        engine = ExecutionEngine()
        pipelines = engine.load_pipelines()
        engine.assign_prompts_to_pipelines(prompts)

        # Validate configuration
        validation = engine.validate_pipeline_configuration()
        assert validation["valid"] is True

        # Get ready pipelines
        ready = engine.get_ready_pipelines()
        assert len(ready) > 0

        # Simulate execution
        for pipeline in ready[:2]:
            result = engine.execute_pipeline(pipeline.id, dry_run=True)
            assert result["status"] == "simulated"

        # Generate execution report
        report = engine.generate_execution_report()
        assert report["total_pipelines"] == len(pipelines)

    def test_domain_coverage(self):
        """Test that all domains are properly configured."""

        loader = PromptLoader()
        domains = loader.load_domains()

        # Verify all tiers are present (4 tiers for 20 domains, tier 5 is integration)
        tier_counts = {}
        for domain in domains.values():
            tier_counts[domain.tier] = tier_counts.get(domain.tier, 0) + 1

        assert len(tier_counts) == 4  # 4 tiers for domains
        assert tier_counts[1] == 5  # Tier 1: D1-D5
        assert tier_counts[2] == 5  # Tier 2: D6-D10
        assert tier_counts[3] == 5  # Tier 3: D11-D15
        assert tier_counts[4] == 5  # Tier 4: D16-D20

        # Verify platforms are assigned
        platforms = set()
        for domain in domains.values():
            platforms.add(domain.primary_platform)

        assert "QuASIM" in platforms
        assert "QStack" in platforms
        assert "QNimbus" in platforms

    def test_keystone_identification(self):
        """Test keystone prompt identification."""

        loader = PromptLoader()
        loader.load_domains()
        prompts = loader.load_all_prompts()

        # Get prompts with keystones
        with_keystones = [p for p in prompts.values() if p.keystone_nodes]

        assert len(with_keystones) > 0

        # Verify keystones are in high-value prompts
        high_value = loader.get_high_value_prompts(threshold=0.85)
        keystone_in_hv = [p for p in high_value if p.keystone_nodes]

        # Most keystone prompts should be high-value
        assert len(keystone_in_hv) > 0
