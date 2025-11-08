#!/usr/bin/env python3
"""Generate Software Bill of Materials (SBOM) in SPDX 2.3 format"""

import json
import subprocess
from datetime import datetime


def generate_sbom(output_file="sbom.spdx.json"):
    """Generate SBOM from project dependencies"""

    sbom = {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": "QuASIM SBOM",
        "documentNamespace": f"https://github.com/robertringler/QuASIM/sbom/{datetime.now().isoformat()}",
        "creationInfo": {
            "created": datetime.now().isoformat(),
            "creators": ["Tool: QuASIM SBOM Generator"],
            "licenseListVersion": "3.21",
        },
        "packages": [],
    }

    # Get pip packages
    try:
        result = subprocess.run(
            ["pip", "list", "--format=json"], capture_output=True, text=True, check=True
        )
        packages = json.loads(result.stdout)

        for pkg in packages:
            sbom["packages"].append(
                {
                    "SPDXID": f"SPDXRef-Package-{pkg['name']}",
                    "name": pkg["name"],
                    "versionInfo": pkg["version"],
                    "downloadLocation": f"https://pypi.org/project/{pkg['name']}/{pkg['version']}/",
                    "filesAnalyzed": False,
                    "supplier": "Organization: PyPI",
                }
            )
    except Exception as e:
        print(f"Error generating SBOM: {e}")

    # Write SBOM
    with open(output_file, "w") as f:
        json.dump(sbom, f, indent=2)

    print(f"✓ SBOM generated: {output_file}")
    return output_file


if __name__ == "__main__":
    generate_sbom()
