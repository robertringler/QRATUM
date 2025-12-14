"""
Generate complete 10,000-prompt library for all 20 domains.

This script generates CSV files with realistic prompts, patentability scores,
commercial potential, keystone nodes, synergy connections, and execution layers.
"""

import csv
import random
from pathlib import Path

# Domain definitions with their characteristics
DOMAINS = {
    "D3": {
        "name": "Multi-Agent AI & Swarm",
        "id_range": (201, 300),
        "platform": "QStack",
        "categories": [
            "Swarm Intelligence",
            "Multi-Agent Systems",
            "Distributed AI",
            "Coordination Algorithms",
            "Emergent Behavior",
        ],
        "connected_domains": ["D2", "D8", "D11", "D12"],
    },
    "D5": {
        "name": "Environmental & Climate Systems",
        "id_range": (401, 500),
        "platform": "QNimbus",
        "categories": [
            "Climate Modeling",
            "Weather Prediction",
            "Ecosystem Simulation",
            "Carbon Capture",
            "Environmental Monitoring",
        ],
        "connected_domains": ["D2", "D10", "D18", "D19"],
    },
    "D7": {
        "name": "Advanced Materials & Nanotech",
        "id_range": (601, 700),
        "platform": "QuASIM",
        "categories": [
            "Nanomaterial Synthesis",
            "Surface Engineering",
            "Nanostructure Design",
            "Quantum Dots",
            "2D Materials",
        ],
        "connected_domains": ["D1", "D4", "D6", "D13"],
    },
    "D8": {
        "name": "AI & Autonomous Systems",
        "id_range": (701, 800),
        "platform": "QStack",
        "categories": [
            "Computer Vision",
            "Autonomous Navigation",
            "Reinforcement Learning",
            "Decision Systems",
            "Sensor Fusion",
        ],
        "connected_domains": ["D3", "D6", "D11", "D20"],
    },
    "D9": {
        "name": "Biomedical & Synthetic Biology",
        "id_range": (801, 900),
        "platform": "QuASIM",
        "categories": [
            "Biomedical Devices",
            "Tissue Engineering",
            "Gene Therapy",
            "Drug Delivery",
            "Diagnostic Systems",
        ],
        "connected_domains": ["D4", "D14", "D16", "D19"],
    },
    "D10": {
        "name": "Climate Science & Geoengineering",
        "id_range": (901, 1000),
        "platform": "QNimbus",
        "categories": [
            "Climate Intervention",
            "Carbon Removal",
            "Solar Radiation Management",
            "Ocean Alkalinization",
            "Atmospheric Chemistry",
        ],
        "connected_domains": ["D5", "D13", "D18", "D19"],
    },
    "D11": {
        "name": "Advanced Robotics & Automation",
        "id_range": (1001, 1500),
        "platform": "QStack",
        "categories": [
            "Robotic Manipulation",
            "Human-Robot Interaction",
            "Collaborative Robotics",
            "Industrial Automation",
            "Soft Robotics",
        ],
        "connected_domains": ["D3", "D8", "D12", "D20"],
    },
    "D12": {
        "name": "IoT & Sensor Networks",
        "id_range": (1501, 2000),
        "platform": "QNimbus",
        "categories": [
            "Sensor Fusion",
            "Edge Computing",
            "Network Protocols",
            "IoT Security",
            "Smart Devices",
        ],
        "connected_domains": ["D3", "D8", "D16", "D20"],
    },
    "D13": {
        "name": "Next-Gen Energy Systems",
        "id_range": (2001, 2500),
        "platform": "QuASIM",
        "categories": [
            "Fusion Reactor Design",
            "Advanced Batteries",
            "Supercapacitors",
            "Hydrogen Storage",
            "Grid Integration",
        ],
        "connected_domains": ["D2", "D7", "D10", "D17"],
    },
    "D14": {
        "name": "Synthetic Life & Biofabrication",
        "id_range": (2501, 3000),
        "platform": "QuASIM",
        "categories": [
            "Synthetic Organisms",
            "Biofabrication",
            "Genetic Circuits",
            "Metabolic Engineering",
            "Biological Computing",
        ],
        "connected_domains": ["D4", "D9", "D19", "D1"],
    },
    "D15": {
        "name": "High-Fidelity Simulation",
        "id_range": (3001, 3500),
        "platform": "QuASIM",
        "categories": [
            "Multi-Physics Simulation",
            "Digital Twins",
            "Real-Time Rendering",
            "CFD/FEA Integration",
            "Surrogate Models",
        ],
        "connected_domains": ["D1", "D2", "D6", "D11"],
    },
    "D17": {
        "name": "Space Exploration & Colonization",
        "id_range": (4501, 5500),
        "platform": "QNimbus",
        "categories": [
            "Habitat Design",
            "Life Support Systems",
            "In-Situ Resource Utilization",
            "Radiation Shielding",
            "Orbital Mechanics",
        ],
        "connected_domains": ["D6", "D13", "D14", "D19"],
    },
    "D18": {
        "name": "Ocean Systems & Marine Tech",
        "id_range": (5501, 6500),
        "platform": "QNimbus",
        "categories": [
            "Ocean Current Modeling",
            "Marine Robotics",
            "Desalination",
            "Offshore Energy",
            "Marine Ecology",
        ],
        "connected_domains": ["D5", "D10", "D11", "D20"],
    },
    "D19": {
        "name": "Agriculture & Food Systems",
        "id_range": (6501, 7500),
        "platform": "QStack",
        "categories": [
            "Precision Agriculture",
            "Crop Optimization",
            "Vertical Farming",
            "Supply Chain",
            "Soil Microbiome",
        ],
        "connected_domains": ["D5", "D9", "D10", "D20"],
    },
    "D20": {
        "name": "Urban Systems & Smart Cities",
        "id_range": (7501, 8500),
        "platform": "QNimbus",
        "categories": [
            "Urban Planning",
            "Traffic Optimization",
            "Smart Infrastructure",
            "Energy Management",
            "Waste Systems",
        ],
        "connected_domains": ["D8", "D11", "D12", "D19"],
    },
}


def generate_prompt_description(domain_id: str, category: str, prompt_num: int) -> str:
    """Generate a realistic prompt description."""
    domain_name = DOMAINS[domain_id]["name"]

    descriptions = {
        "Swarm Intelligence": f"Swarm-based optimization for {['resource allocation', 'task scheduling', 'path planning'][prompt_num % 3]}",
        "Multi-Agent Systems": f"Multi-agent coordination for {['distributed sensing', 'collaborative planning', 'emergent behavior'][prompt_num % 3]}",
        "Climate Modeling": f"High-resolution climate simulation for {['regional forecasting', 'extreme events', 'long-term trends'][prompt_num % 3]}",
        "Nanomaterial Synthesis": f"Synthesis pathways for {['graphene', 'carbon nanotubes', 'quantum dots'][prompt_num % 3]} with enhanced properties",
        "Computer Vision": f"Vision-based {['object detection', 'scene understanding', 'motion tracking'][prompt_num % 3]} for autonomous systems",
        "Biomedical Devices": f"Medical device simulation for {['implants', 'sensors', 'drug delivery'][prompt_num % 3]}",
        "Climate Intervention": f"Geoengineering approach for {['carbon removal', 'solar management', 'ocean alkalinization'][prompt_num % 3]}",
        "Robotic Manipulation": f"Advanced manipulation for {['assembly', 'sorting', 'delicate handling'][prompt_num % 3]} tasks",
        "Sensor Fusion": f"Multi-sensor fusion for {['localization', 'tracking', 'recognition'][prompt_num % 3]} applications",
        "Fusion Reactor Design": f"Fusion reactor {['plasma containment', 'blanket design', 'divertor optimization'][prompt_num % 3]}",
        "Synthetic Organisms": f"Synthetic biology design for {['biofuel production', 'waste remediation', 'biosensing'][prompt_num % 3]}",
        "Multi-Physics Simulation": f"Coupled simulation of {['fluid-structure', 'thermal-mechanical', 'electromagnetic'][prompt_num % 3]} interactions",
        "Habitat Design": f"Space habitat design for {['lunar base', 'Mars colony', 'orbital station'][prompt_num % 3]}",
        "Ocean Current Modeling": f"Ocean dynamics simulation for {['current prediction', 'upwelling zones', 'temperature distribution'][prompt_num % 3]}",
        "Precision Agriculture": f"Precision farming for {['irrigation', 'fertilization', 'pest management'][prompt_num % 3]} optimization",
        "Urban Planning": f"Smart city planning for {['traffic flow', 'energy distribution', 'waste management'][prompt_num % 3]}",
    }

    base_desc = descriptions.get(category, f"{category} simulation and optimization")
    return f"{base_desc} using {domain_name.lower()} approaches"


def generate_patentability_score(is_keystone: bool = False) -> float:
    """Generate realistic patentability score."""
    if is_keystone:
        return round(random.uniform(0.88, 0.97), 2)
    return round(random.uniform(0.70, 0.95), 2)


def generate_commercial_potential(is_keystone: bool = False) -> float:
    """Generate realistic commercial potential score."""
    if is_keystone:
        return round(random.uniform(0.85, 0.96), 2)
    return round(random.uniform(0.68, 0.94), 2)


def generate_keystone_nodes(domain_id: str, is_keystone: bool = False) -> str:
    """Generate keystone technology nodes."""
    if not is_keystone:
        return ""

    nodes = {
        "D3": ["Swarm algorithms", "Multi-agent coordination"],
        "D5": ["Climate modeling", "Ecosystem simulation"],
        "D7": ["Nanofabrication", "Surface chemistry"],
        "D8": ["Deep learning", "Sensor fusion"],
        "D9": ["Tissue engineering", "Gene therapy"],
        "D10": ["Carbon capture", "Geoengineering"],
        "D11": ["Robotic control", "Human-robot interaction"],
        "D12": ["Edge computing", "IoT protocols"],
        "D13": ["Fusion physics", "Advanced storage"],
        "D14": ["Synthetic biology", "Metabolic engineering"],
        "D15": ["Multi-physics", "Digital twins"],
        "D17": ["Life support", "ISRU"],
        "D18": ["Ocean modeling", "Marine robotics"],
        "D19": ["Crop optimization", "Precision farming"],
        "D20": ["Smart infrastructure", "Urban analytics"],
    }

    return ";".join(nodes.get(domain_id, ["Technology node 1", "Technology node 2"]))


def generate_prompts_for_domain(domain_id: str, output_dir: Path):
    """Generate all prompts for a domain."""
    domain = DOMAINS[domain_id]
    start_id, end_id = domain["id_range"]

    # Determine keystone prompt IDs (7 per domain, evenly distributed)
    total_prompts = end_id - start_id + 1
    keystone_interval = total_prompts // 7
    keystone_ids = {start_id + i * keystone_interval for i in range(7)}

    filename = f"d{domain_id[1:].zfill(2)}_{domain['name'].lower().replace(' ', '_').replace('&', 'and')}.csv"
    filepath = output_dir / filename

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "category",
                "description",
                "domain",
                "patentability_score",
                "commercial_potential",
                "keystone_nodes",
                "synergy_connections",
                "execution_layers",
                "phase_deployment",
                "output_type",
            ],
        )
        writer.writeheader()

        for prompt_id in range(start_id, end_id + 1):
            is_keystone = prompt_id in keystone_ids
            category = domain["categories"][(prompt_id - start_id) % len(domain["categories"])]

            # Select synergy connections (1-2 domains)
            num_connections = 1 if random.random() < 0.6 else 2
            synergy_conns = random.sample(domain["connected_domains"], num_connections)

            # Determine phase based on ID (earlier IDs in earlier phases)
            if prompt_id < start_id + total_prompts * 0.3:
                phase = 1
            elif prompt_id < start_id + total_prompts * 0.6:
                phase = 2
            elif prompt_id < start_id + total_prompts * 0.85:
                phase = 3
            else:
                phase = 4

            # Output types
            output_types = [
                "simulation",
                "model",
                "analysis",
                "optimization",
                "design",
                "prediction",
            ]
            output_type = output_types[(prompt_id - start_id) % len(output_types)]

            writer.writerow(
                {
                    "id": prompt_id,
                    "category": category,
                    "description": generate_prompt_description(
                        domain_id, category, prompt_id - start_id
                    ),
                    "domain": domain_id,
                    "patentability_score": generate_patentability_score(is_keystone),
                    "commercial_potential": generate_commercial_potential(is_keystone),
                    "keystone_nodes": generate_keystone_nodes(domain_id, is_keystone),
                    "synergy_connections": ";".join(synergy_conns),
                    "execution_layers": domain["platform"],
                    "phase_deployment": phase,
                    "output_type": output_type,
                }
            )

    print(f"✓ Generated {total_prompts} prompts for {domain_id} ({domain['name']})")


def main():
    """Generate all missing domain prompt files."""
    output_dir = Path(__file__).parent.parent / "data" / "prompts"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating complete prompt library for missing domains...")
    print(f"Output directory: {output_dir}")
    print()

    for domain_id in sorted(DOMAINS.keys()):
        generate_prompts_for_domain(domain_id, output_dir)

    print()
    print("✓ All domain prompts generated successfully!")
    print(
        f"Total new prompts generated: {sum(d['id_range'][1] - d['id_range'][0] + 1 for d in DOMAINS.values())}"
    )


if __name__ == "__main__":
    main()
