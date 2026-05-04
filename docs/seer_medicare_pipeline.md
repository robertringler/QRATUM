# SEER-Medicare Pipeline Documentation

## Overview

The SEER-Medicare pipeline provides tools for processing SEER-Medicare linked data
to build treatment timelines and integrate with QRATUM oncology analysis.

**RESEARCH USE ONLY** - This pipeline produces research hypotheses and retrospective
analyses. All outputs require clinical validation before any interpretation.

## Installation

The pipeline is part of the QRATUM package. No additional installation is required
beyond the base QRATUM dependencies.

```bash
# Install QRATUM with dev dependencies
pip install -e ".[dev]"
```

## Quick Start

### 1. Prepare Your Data

Place your SEER-Medicare data files in local directories:
- **SEER Registry Files:** CSV files containing cancer registry data
- **Medicare Claims Files:** CSV files for MEDPAR, NCH, OUTSAF, PDE, etc.

Expected file naming conventions:
- MEDPAR files: `*medpar*.csv` or `*inpatient*.csv`
- NCH/Carrier files: `*nch*.csv` or `*carrier*.csv`
- Outpatient files: `*outsaf*.csv` or `*outpatient*.csv`
- Part D files: `*pde*.csv` or `*partd*.csv`

### 2. Run the Pipeline

```bash
python scripts/run_seer_medicare_pipeline.py \
    --seer_dir /path/to/seer/data \
    --claims_dir /path/to/medicare/claims \
    --cancer_site "lung" \
    --histology "adenocarcinoma" \
    --stage "IV" \
    --diagnosis_year_min 2010 \
    --diagnosis_year_max 2018 \
    --lookback_days 365 \
    --followup_days 1095 \
    --seed 42 \
    --output_dir artifacts/my_analysis
```

### 3. Generate Report

```bash
python scripts/build_seer_medicare_report.py \
    --run_dir artifacts/my_analysis
```

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    SEER-Medicare Pipeline                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │
│  │ SEER Files  │───▶│  Registry   │───▶│   Linked Patient    │  │
│  │   (CSV)     │    │   Parser    │    │      Records        │  │
│  └─────────────┘    └─────────────┘    └─────────────────────┘  │
│                                                 │                │
│  ┌─────────────┐    ┌─────────────┐             │                │
│  │   Claims    │───▶│   Claims    │─────────────┘                │
│  │   Files     │    │   Parsers   │                              │
│  └─────────────┘    └─────────────┘                              │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                     Cohort Builder                          │ │
│  │  - Cancer site/histology/stage filters                      │ │
│  │  - Diagnosis year bounds                                    │ │
│  │  - Enrollment requirements (if available)                   │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                            │                                     │
│                            ▼                                     │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                Treatment Timeline Builder                   │ │
│  │  - Identify treatment events from claims                    │ │
│  │  - Group into episodes / lines of therapy                   │ │
│  │  - Calculate time-to-treatment metrics                      │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                            │                                     │
│                            ▼                                     │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                   QRATUM Integration                        │ │
│  │  - Feature engineering (proxies)                            │ │
│  │  - CausalOncologyGraph construction                         │ │
│  │  - XENONInterventionSearch                                  │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                            │                                     │
│                            ▼                                     │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                    Report Generation                        │ │
│  │  - Markdown/PDF report                                      │ │
│  │  - DUA compliance checklist                                 │ │
│  │  - Reproducibility artifacts                                │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Input File Formats

### SEER Registry Files (CSV)

Expected columns (various aliases supported):
- `PATIENT_ID` or `patient_id`: Unique patient identifier
- `DATE_OF_DIAGNOSIS` or `DX_DATE`: Diagnosis date (YYYYMMDD or YYYY-MM-DD)
- `PRIMARY_SITE` or `SITE`: Cancer primary site
- `HISTOLOGY` or `HIST`: Histology type
- `STAGE` or `AJCC_STAGE`: Stage at diagnosis
- `AGE` or `AGE_AT_DX`: Age at diagnosis
- `SEX` or `GENDER`: Patient sex
- `VITAL_STATUS` or `VS`: Alive/Dead status
- `SURVIVAL` or `SURV_MOS`: Survival months

### Medicare Claims Files (CSV)

**MEDPAR (Inpatient):**
- `BENE_ID`: Beneficiary ID
- `ADMSN_DT`: Admission date
- `DRG_CD`: DRG code
- `ICD_DGNS_CD1`...`ICD_DGNS_CD25`: Diagnosis codes
- `ICD_PRCDR_CD1`...`ICD_PRCDR_CD25`: Procedure codes

**NCH/Carrier:**
- `BENE_ID`: Beneficiary ID
- `CLM_FROM_DT`: Claim from date
- `HCPCS_CD`: HCPCS/CPT code
- `PRVDR_SPCLTY`: Provider specialty

**OUTSAF (Outpatient):**
- `BENE_ID`: Beneficiary ID
- `CLM_FROM_DT`: Claim from date
- `HCPCS_CD`: HCPCS/CPT code
- `REV_CNTR`: Revenue center code

**PDE (Part D):**
- `BENE_ID`: Beneficiary ID
- `SRVC_DT`: Service date
- `PROD_SRVC_ID`: NDC code

## Output Artifacts

Each pipeline run creates a directory with:

```
artifacts/seer_medicare_run_YYYYMMDD_HHMMSS/
├── run_config.json           # Pipeline parameters
├── environment.txt           # Python version + pip freeze
├── git_commit.txt            # Git commit hash
├── dataset_manifest.json     # File names, sizes, SHA256 hashes
├── cohort_counts.json        # Cohort statistics (suppressed)
├── timeline_summary.json     # Treatment timeline statistics
├── qratum_sequences.json     # QRATUM hypothesis sequences
├── dua_checklist.json        # DUA compliance checklist
└── REPORT.md                 # Publication-quality report
```

## Code Mapping Extension

Treatment identification relies on configurable code mappings. The default
mappings are in:

```
qratum/oncology/registry/seer_medicare/resources/code_maps/treatment_codes.yaml
```

To add custom mappings, create a YAML file:

```yaml
mappings:
  - code: "J9999"
    code_system: "hcpcs"
    treatment_type: "targeted"
    drug_name: "my_drug"
    description: "My custom drug"
    confidence: 0.8
```

Then use `--code_mapping_file` to load it:

```bash
python scripts/run_seer_medicare_pipeline.py \
    --code_mapping_file /path/to/my_mappings.yaml \
    ...
```

## DUA Compliance

### Privacy Controls

1. **Patient Key Hashing:** All patient identifiers are hashed using SHA256
   with a configurable salt. Original IDs are never stored or logged.

2. **Cell Size Suppression:** Any aggregate count < 11 is suppressed by default
   (configurable via `--min_cell_size`).

3. **Safe Logging:** The `SafeLogger` class automatically redacts potential
   identifiers from log messages.

4. **No Network Upload:** All processing runs locally. No data is transmitted.

### DUA Checklist

Each run generates a `dua_checklist.json` with verification items:

- [ ] No patient-level identifiers in output
- [ ] All cell counts >= minimum threshold
- [ ] No network upload of data
- [ ] Outputs stored in designated directory
- [ ] Data manifest includes file hashes
- [ ] Reproducibility artifacts generated

## Reproducibility

### Deterministic Runs

The pipeline uses a fixed random seed (default: 42) for all stochastic operations.
Results are deterministic given the same inputs and configuration.

### Environment Capture

Each run captures:
- Python version
- Complete `pip freeze` output
- Git commit hash

### Data Manifest

File hashes (SHA256) are computed for all input files, allowing verification
that the exact same data was used.

## Limitations & Caveats

### Medicare Population Bias
- Cohort limited to Medicare beneficiaries (age 65+)
- Results may not generalize to younger populations

### Claims-Based Limitations
- Treatment timing reflects billing dates, not administration
- Diagnosis codes may not capture full clinical picture
- Drug identification relies on J-code mappings

### Proxy Variables
All features derived from claims are **proxies**, not direct measurements:
- **Tumor burden proxy:** Based on utilization patterns
- **Immune engagement proxy:** Based on treatment codes
- **Toxicity proxy:** Based on healthcare burden

### Research Use Only
This pipeline generates hypotheses for research exploration.
Results require clinical validation through prospective studies.

## API Reference

### Core Modules

#### `qratum.oncology.registry.seer_medicare.schema`
Data models for normalized records:
- `ClaimEvent`: Normalized claim event
- `RegistryCase`: SEER registry case
- `TreatmentEvent`: Identified treatment
- `PatientTimeline`: Complete patient timeline
- `CohortDefinition`: Cohort criteria

#### `qratum.oncology.registry.seer_medicare.privacy`
Privacy utilities:
- `PrivacyConfig`: Configuration for privacy controls
- `SafeLogger`: Privacy-preserving logger
- `create_patient_key()`: Hash patient identifiers
- `suppress_small_counts()`: Cell size suppression

#### `qratum.oncology.registry.seer_medicare.io`
File I/O utilities:
- `create_dataset_manifest()`: Generate file manifest
- `read_csv_chunked()`: Chunked CSV reading
- `RunArtifacts`: Container for run outputs

#### `qratum.oncology.registry.seer_medicare.seer_registry`
SEER parsing:
- `SEERRegistryParser`: Parse SEER case files

#### `qratum.oncology.registry.seer_medicare.medicare_claims`
Claims parsing:
- `MEDPARParser`: Inpatient claims
- `NCHParser`: Carrier/physician claims
- `OUTSAFParser`: Outpatient claims
- `PDEParser`: Part D drug events

#### `qratum.oncology.registry.seer_medicare.cohort`
Cohort building:
- `CohortBuilder`: Build cohorts with criteria
- `CohortStats`: Cohort statistics

#### `qratum.oncology.registry.seer_medicare.timelines`
Timeline construction:
- `TreatmentTimelineBuilder`: Build treatment timelines
- `CodeMappingLibrary`: Treatment code mappings

#### `qratum.oncology.registry.seer_medicare.features`
Feature engineering:
- `FeatureEngineer`: Extract features for QRATUM
- `BaselineFeatures`: Features at index date
- `StateFeatures`: Dynamic state features

## Support

For issues and questions, please file a GitHub issue in the QRATUM repository.

## License

Apache 2.0 - See LICENSE file in repository root.
