"""Main CLI entry point for Qubic Meta Library."""

import json

import click

from qubic_meta_library.services import (Dashboard, ExecutionEngine,
                                         PatentAnalyzer, PromptLoader,
                                         SynergyMapper)


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Qubic Meta Library - 10,000-Prompt System for R&D and IP Generation."""

    pass


@cli.command()
@click.option("--config-dir", type=click.Path(exists=True), help="Configuration directory")
@click.option("--data-dir", type=click.Path(exists=True), help="Data directory")
def load_all(config_dir, data_dir):
    """Load all prompts and domains."""

    loader = PromptLoader(config_dir=config_dir, data_dir=data_dir)

    click.echo("Loading domains...")
    domains = loader.load_domains()
    click.echo(f"✓ Loaded {len(domains)} domains")

    click.echo("Loading prompts...")
    prompts = loader.load_all_prompts()
    click.echo(f"✓ Loaded {len(prompts)} prompts")

    # Show summary by domain
    click.echo("\nPrompts by domain:")
    for domain_id in sorted(domains.keys()):
        domain_prompts = loader.get_prompts_by_domain(domain_id)
        click.echo(f"  {domain_id}: {len(domain_prompts)} prompts")


@cli.command()
@click.option("--config-dir", type=click.Path(exists=True), help="Configuration directory")
@click.option("--data-dir", type=click.Path(exists=True), help="Data directory")
@click.option("--threshold", type=float, default=0.8, help="High-value threshold")
def high_value(config_dir, data_dir, threshold):
    """Extract high-value prompts."""

    loader = PromptLoader(config_dir=config_dir, data_dir=data_dir)
    loader.load_domains()
    loader.load_all_prompts()

    high_value_prompts = loader.get_high_value_prompts(threshold)

    click.echo(f"Found {len(high_value_prompts)} high-value prompts (threshold: {threshold})")
    click.echo("\nTop 10 high-value prompts:")
    for i, prompt in enumerate(high_value_prompts[:10], 1):
        click.echo(
            f"{i}. ID {prompt.id}: {prompt.description[:60]}... "
            f"(P: {prompt.patentability_score:.2f}, C: {prompt.commercial_potential:.2f})"
        )


@cli.command()
@click.option("--config-dir", type=click.Path(exists=True), help="Configuration directory")
@click.option("--data-dir", type=click.Path(exists=True), help="Data directory")
@click.option("--output", type=click.Path(), help="Output file for patent report")
def analyze_patents(config_dir, data_dir, output):
    """Analyze patent opportunities."""

    loader = PromptLoader(config_dir=config_dir, data_dir=data_dir)
    loader.load_domains()
    prompts = loader.load_all_prompts()

    analyzer = PatentAnalyzer()

    click.echo("Analyzing patent opportunities...")
    report = analyzer.generate_patent_pipeline_report(prompts)

    click.echo(f"\n✓ Total prompts: {report['total_prompts']}")
    click.echo(f"✓ High-value patents: {report['high_value_count']}")
    click.echo(f"✓ Premium patents: {report['premium_value_count']}")
    click.echo(f"✓ Cross-domain opportunities: {report['cross_domain_opportunities']}")

    click.echo("\nTop 5 patent opportunities:")
    for i, opp in enumerate(report["top_10_opportunities"][:5], 1):
        click.echo(
            f"{i}. ID {opp['prompt_id']}: {opp['description'][:50]}... "
            f"(P: {opp['patentability']:.2f})"
        )

    if output:
        with open(output, "w") as f:
            json.dump(report, f, indent=2)
        click.echo(f"\n✓ Report saved to {output}")


@cli.command()
@click.option("--config-dir", type=click.Path(exists=True), help="Configuration directory")
@click.option("--data-dir", type=click.Path(exists=True), help="Data directory")
def map_synergies(config_dir, data_dir):
    """Map synergy clusters and cross-domain connections."""

    loader = PromptLoader(config_dir=config_dir, data_dir=data_dir)
    domains = loader.load_domains()
    prompts = loader.load_all_prompts()

    mapper = SynergyMapper(config_dir=config_dir)
    clusters = mapper.load_clusters()

    click.echo(f"Loaded {len(clusters)} synergy clusters")

    # Find synergies
    synergies = mapper.find_synergies(prompts, domains)

    click.echo("\nSynergy clusters:")
    for cluster_id, prompt_ids in list(synergies.items())[:10]:
        if cluster_id in clusters:
            cluster = clusters[cluster_id]
            click.echo(
                f"  {cluster_id} ({cluster.name}): {len(prompt_ids)} prompts, "
                f"{len(cluster.domains)} domains"
            )

    # Generate report
    report = mapper.generate_cluster_report()
    click.echo(f"\n✓ Total revenue projection: ${report['total_revenue_projection']:,.0f}")


@cli.command()
@click.option("--config-dir", type=click.Path(exists=True), help="Configuration directory")
@click.option("--data-dir", type=click.Path(exists=True), help="Data directory")
@click.option("--dry-run", is_flag=True, help="Simulate execution without processing")
def execute_pipelines(config_dir, data_dir, dry_run):
    """Execute or simulate pipeline execution."""

    loader = PromptLoader(config_dir=config_dir, data_dir=data_dir)
    loader.load_domains()
    prompts = loader.load_all_prompts()

    engine = ExecutionEngine(config_dir=config_dir)
    pipelines = engine.load_pipelines()

    click.echo(f"Loaded {len(pipelines)} pipelines")

    # Assign prompts to pipelines
    assignments = engine.assign_prompts_to_pipelines(prompts)

    click.echo("\nPipeline assignments:")
    for pipeline_id, prompt_ids in assignments.items():
        if pipeline_id in pipelines:
            pipeline = pipelines[pipeline_id]
            click.echo(f"  {pipeline.name} (Phase {pipeline.phase}): {len(prompt_ids)} prompts")

    # Get ready pipelines
    ready = engine.get_ready_pipelines()
    click.echo(f"\n✓ {len(ready)} pipelines ready for execution")

    if dry_run:
        click.echo("\n--- Dry Run Mode ---")
        for pipeline in ready[:3]:
            result = engine.execute_pipeline(pipeline.id, dry_run=True)
            click.echo(f"  {result['pipeline_name']}: {result['prompt_count']} prompts")


@cli.command()
@click.option("--config-dir", type=click.Path(exists=True), help="Configuration directory")
@click.option("--data-dir", type=click.Path(exists=True), help="Data directory")
@click.option("--output", type=click.Path(), help="Output file for dashboard data")
def dashboard(config_dir, data_dir, output):
    """Generate KPI dashboard and metrics."""

    loader = PromptLoader(config_dir=config_dir, data_dir=data_dir)
    domains = loader.load_domains()
    prompts = loader.load_all_prompts()

    mapper = SynergyMapper(config_dir=config_dir)
    clusters = mapper.load_clusters()

    engine = ExecutionEngine(config_dir=config_dir)
    pipelines = engine.load_pipelines()
    engine.assign_prompts_to_pipelines(prompts)

    dash = Dashboard()
    kpis = dash.calculate_kpis(prompts, domains, clusters, pipelines)

    # Generate executive summary
    summary = dash.generate_executive_summary(prompts, domains, clusters, pipelines)
    click.echo(summary)

    if output:
        with open(output, "w") as f:
            json.dump(kpis, f, indent=2)
        click.echo(f"\n✓ Dashboard data saved to {output}")


@cli.command()
@click.option("--config-dir", type=click.Path(exists=True), help="Configuration directory")
def validate(config_dir):
    """Validate pipeline configuration."""

    engine = ExecutionEngine(config_dir=config_dir)
    engine.load_pipelines()

    click.echo("Validating pipeline configuration...")
    validation = engine.validate_pipeline_configuration()

    if validation["valid"]:
        click.echo("✓ Configuration is valid")
    else:
        click.echo(f"✗ Configuration has {validation['error_count']} errors")

    if validation["errors"]:
        click.echo("\nErrors:")
        for error in validation["errors"]:
            click.echo(f"  • {error}")

    if validation["warnings"]:
        click.echo(f"\nWarnings ({validation['warning_count']}):")
        for warning in validation["warnings"][:5]:
            click.echo(f"  • {warning}")


if __name__ == "__main__":
    cli()
