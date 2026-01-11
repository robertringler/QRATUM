"""
Medicare Claims Parsing Adapters

Parses Medicare claims files (MEDPAR, NCH, OUTSAF, PDE, etc.)
into normalized ClaimEvent objects.

RESEARCH USE ONLY - Not for clinical diagnosis or treatment decisions.
"""

from __future__ import annotations

import csv
import logging
from datetime import date
from pathlib import Path
from typing import Generator, Optional

from .privacy import PrivacyConfig, SafeLogger, create_claim_id_hash, create_patient_key
from .schema import ClaimEvent, ClaimSetting, CodeSystem

logger = logging.getLogger(__name__)


# Field mappings for different claim types
MEDPAR_FIELD_ALIASES = {
    "patient_id": ["BENE_ID", "PATIENT_ID", "BENEFICIARY_ID", "bene_id"],
    "claim_id": ["CLM_ID", "CLAIM_ID", "MEDPAR_ID", "clm_id"],
    "from_date": ["ADMSN_DT", "CLM_FROM_DT", "FROM_DT", "admsn_dt"],
    "thru_date": ["DSCHRG_DT", "CLM_THRU_DT", "THRU_DT", "dschrg_dt"],
    "drg": ["DRG_CD", "MS_DRG", "drg_cd"],
    "dx_codes": ["DGNS_CD", "ICD_DGNS_CD", "PRNCPAL_DGNS_CD"],
    "px_codes": ["PRCDR_CD", "ICD_PRCDR_CD", "SRGCL_PRCDR_CD"],
    "payment": ["CLM_PMT_AMT", "PMT_AMT", "MDCR_PMT_AMT"],
    "provider": ["PRVDR_NUM", "ORG_NPI_NUM", "provider_id"],
}

NCH_FIELD_ALIASES = {
    "patient_id": ["BENE_ID", "PATIENT_ID", "bene_id"],
    "claim_id": ["CLM_ID", "CARR_CLM_ID", "clm_id"],
    "from_date": ["CLM_FROM_DT", "LINE_1ST_EXPNS_DT", "from_dt"],
    "thru_date": ["CLM_THRU_DT", "LINE_LAST_EXPNS_DT", "thru_dt"],
    "hcpcs": ["HCPCS_CD", "LINE_HCPCS_CD", "hcpcs_cd"],
    "dx_codes": ["ICD_DGNS_CD", "PRNCPAL_DGNS_CD", "LINE_ICD_DGNS_CD"],
    "payment": ["CLM_PMT_AMT", "LINE_NCH_PMT_AMT", "pmt_amt"],
    "provider_specialty": ["PRVDR_SPCLTY", "CARR_LINE_PRVDR_TYPE_CD"],
}

OUTSAF_FIELD_ALIASES = {
    "patient_id": ["BENE_ID", "PATIENT_ID", "bene_id"],
    "claim_id": ["CLM_ID", "clm_id"],
    "from_date": ["CLM_FROM_DT", "from_dt"],
    "thru_date": ["CLM_THRU_DT", "thru_dt"],
    "hcpcs": ["HCPCS_CD", "LINE_HCPCS_CD"],
    "revenue": ["REV_CNTR", "REV_CNTR_CD"],
    "dx_codes": ["ICD_DGNS_CD", "PRNCPAL_DGNS_CD"],
    "payment": ["CLM_PMT_AMT", "REV_CNTR_PMT_AMT_AMT"],
}

PDE_FIELD_ALIASES = {
    "patient_id": ["BENE_ID", "PDE_ID", "bene_id"],
    "event_id": ["PDE_ID", "RX_SRVC_RFRNC_NUM"],
    "service_date": ["SRVC_DT", "RX_SRVC_DT", "FILL_DT"],
    "ndc": ["PROD_SRVC_ID", "NDC", "ndc"],
    "quantity": ["QTY_DSPNSD_NUM", "DAYS_SUPPLY_NUM"],
    "payment": ["PTNT_PAY_AMT", "TOT_RX_CST_AMT", "CVRD_D_PLAN_PD_AMT"],
    "prescriber": ["PRSCRBR_ID", "SRVC_PRVDR_ID"],
}


def _find_field(row: dict[str, str], aliases: list[str]) -> Optional[str]:
    """Find field value using multiple possible column names."""
    for alias in aliases:
        if alias in row:
            return row[alias]
        for key in row:
            if key.upper() == alias.upper():
                return row[key]
    return None


def _parse_date(date_str: Optional[str]) -> Optional[date]:
    """Parse date from various Medicare formats."""
    if not date_str:
        return None

    date_str = date_str.strip()

    try:
        # YYYYMMDD format
        if len(date_str) == 8 and date_str.isdigit():
            return date(int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8]))

        # YYYY-MM-DD format
        if len(date_str) == 10 and "-" in date_str:
            parts = date_str.split("-")
            return date(int(parts[0]), int(parts[1]), int(parts[2]))

        # MM/DD/YYYY format
        if "/" in date_str:
            parts = date_str.split("/")
            if len(parts) == 3:
                return date(int(parts[2]), int(parts[0]), int(parts[1]))

    except (ValueError, IndexError):
        pass

    return None


def _determine_code_system(code: str, context: str = "") -> CodeSystem:
    """Determine code system from code format and context.

    Args:
        code: The code value
        context: Context hint (dx, px, hcpcs, etc.)

    Returns:
        CodeSystem enum value
    """
    if not code:
        return CodeSystem.UNKNOWN

    code = code.strip()

    # NDC is 11 digits
    if len(code) == 11 and code.isdigit():
        return CodeSystem.NDC

    # DRG is typically 3 digits
    if context == "drg" and len(code) <= 4 and code.isdigit():
        return CodeSystem.DRG

    # Revenue center codes are 4 digits
    if context == "revenue" and len(code) == 4 and code.isdigit():
        return CodeSystem.REVENUE

    # HCPCS/CPT codes
    if context == "hcpcs" or (len(code) == 5 and (code[0].isalpha() or code.isdigit())):
        # HCPCS Level II starts with letter
        if code[0].isalpha():
            return CodeSystem.HCPCS
        return CodeSystem.CPT

    # ICD codes
    if context == "dx" or context == "px":
        # ICD-10 diagnosis starts with letter
        if context == "dx":
            if code[0].isalpha():
                return CodeSystem.ICD10_DX
            return CodeSystem.ICD9_DX
        else:
            if code[0].isalpha() or (len(code) == 7 and code[0].isdigit()):
                return CodeSystem.ICD10_PX
            return CodeSystem.ICD9_PX

    return CodeSystem.UNKNOWN


class MedicareClaimsParser:
    """
    Base parser for Medicare claims files.

    Subclasses implement specific claim type parsing.
    """

    setting: ClaimSetting = ClaimSetting.UNKNOWN
    field_aliases: dict[str, list[str]] = {}

    def __init__(
        self,
        privacy_config: Optional[PrivacyConfig] = None,
    ) -> None:
        """Initialize parser.

        Args:
            privacy_config: Privacy configuration
        """
        self.privacy_config = privacy_config or PrivacyConfig()
        self.safe_logger = SafeLogger("medicare_claims", self.privacy_config)

    def parse_file(
        self,
        filepath: Path,
        encoding: str = "utf-8",
        delimiter: str = ",",
    ) -> Generator[ClaimEvent, None, None]:
        """Parse a claims file.

        Args:
            filepath: Path to claims file
            encoding: File encoding
            delimiter: Field delimiter

        Yields:
            ClaimEvent objects
        """
        filepath = Path(filepath)
        self.safe_logger.info(f"Parsing claims file: {filepath.name}")

        count = 0
        errors = 0

        with open(filepath, "r", encoding=encoding) as f:
            reader = csv.DictReader(f, delimiter=delimiter)

            for row in reader:
                try:
                    events = self._parse_row(row, str(filepath.name))
                    for event in events:
                        yield event
                        count += 1
                except Exception as e:
                    errors += 1
                    if errors <= 10:
                        self.safe_logger.warning(f"Error parsing row: {e}")

        self.safe_logger.info_aggregate("Claims parsed", count)
        if errors:
            self.safe_logger.info_aggregate("Parse errors", errors)

    def _parse_row(self, row: dict[str, str], source: str) -> list[ClaimEvent]:
        """Parse a single row. Override in subclasses."""
        return []


class MEDPARParser(MedicareClaimsParser):
    """Parser for MEDPAR (inpatient) claims."""

    setting = ClaimSetting.INPATIENT
    field_aliases = MEDPAR_FIELD_ALIASES

    def _parse_row(self, row: dict[str, str], source: str) -> list[ClaimEvent]:
        """Parse MEDPAR row into ClaimEvents."""
        events = []

        # Get patient key
        patient_id = _find_field(row, self.field_aliases["patient_id"])
        if not patient_id:
            return events

        patient_key = create_patient_key(
            patient_id,
            salt=self.privacy_config.salt,
            algorithm=self.privacy_config.hash_algorithm,
        )

        # Get claim ID hash
        claim_id = _find_field(row, self.field_aliases["claim_id"]) or ""
        claim_id_hash = create_claim_id_hash(claim_id, self.privacy_config.salt)

        # Parse dates
        from_date = _parse_date(_find_field(row, self.field_aliases["from_date"]))
        if not from_date:
            return events

        # Get payment
        payment_str = _find_field(row, self.field_aliases["payment"])
        payment = None
        if payment_str and payment_str.replace(".", "").isdigit():
            payment = float(payment_str)

        # DRG code
        drg = _find_field(row, self.field_aliases["drg"])
        if drg:
            events.append(
                ClaimEvent(
                    patient_key=patient_key,
                    event_date=from_date,
                    setting=self.setting,
                    code_system=CodeSystem.DRG,
                    code=drg.strip(),
                    cost=payment,
                    raw_source=source,
                    claim_id_hash=claim_id_hash,
                )
            )

        # Diagnosis codes
        for i in range(1, 26):  # Up to 25 diagnosis codes
            dx_field = f"ICD_DGNS_CD{i}" if i > 1 else "PRNCPAL_DGNS_CD"
            dx_code = row.get(dx_field) or row.get(f"DGNS_CD{i}")
            if dx_code and dx_code.strip():
                code_system = _determine_code_system(dx_code.strip(), "dx")
                events.append(
                    ClaimEvent(
                        patient_key=patient_key,
                        event_date=from_date,
                        setting=self.setting,
                        code_system=code_system,
                        code=dx_code.strip(),
                        raw_source=source,
                        line_number=i,
                        claim_id_hash=claim_id_hash,
                    )
                )

        # Procedure codes
        for i in range(1, 26):  # Up to 25 procedure codes
            px_field = f"ICD_PRCDR_CD{i}" if i > 1 else "SRGCL_PRCDR_CD1"
            px_code = row.get(px_field) or row.get(f"PRCDR_CD{i}")
            if px_code and px_code.strip():
                code_system = _determine_code_system(px_code.strip(), "px")
                events.append(
                    ClaimEvent(
                        patient_key=patient_key,
                        event_date=from_date,
                        setting=self.setting,
                        code_system=code_system,
                        code=px_code.strip(),
                        raw_source=source,
                        line_number=100 + i,
                        claim_id_hash=claim_id_hash,
                    )
                )

        return events


class NCHParser(MedicareClaimsParser):
    """Parser for NCH (carrier/physician) claims."""

    setting = ClaimSetting.CARRIER
    field_aliases = NCH_FIELD_ALIASES

    def _parse_row(self, row: dict[str, str], source: str) -> list[ClaimEvent]:
        """Parse NCH row into ClaimEvents."""
        events = []

        patient_id = _find_field(row, self.field_aliases["patient_id"])
        if not patient_id:
            return events

        patient_key = create_patient_key(
            patient_id,
            salt=self.privacy_config.salt,
            algorithm=self.privacy_config.hash_algorithm,
        )

        claim_id = _find_field(row, self.field_aliases["claim_id"]) or ""
        claim_id_hash = create_claim_id_hash(claim_id, self.privacy_config.salt)

        from_date = _parse_date(_find_field(row, self.field_aliases["from_date"]))
        if not from_date:
            return events

        payment_str = _find_field(row, self.field_aliases["payment"])
        payment = None
        if payment_str and payment_str.replace(".", "").replace("-", "").isdigit():
            payment = float(payment_str)

        provider_specialty = _find_field(row, self.field_aliases["provider_specialty"])

        # HCPCS/CPT code
        hcpcs = _find_field(row, self.field_aliases["hcpcs"])
        if hcpcs and hcpcs.strip():
            code_system = _determine_code_system(hcpcs.strip(), "hcpcs")
            events.append(
                ClaimEvent(
                    patient_key=patient_key,
                    event_date=from_date,
                    setting=self.setting,
                    code_system=code_system,
                    code=hcpcs.strip(),
                    cost=payment,
                    provider_type=provider_specialty,
                    raw_source=source,
                    claim_id_hash=claim_id_hash,
                )
            )

        return events


class OUTSAFParser(MedicareClaimsParser):
    """Parser for OUTSAF (outpatient) claims."""

    setting = ClaimSetting.OUTPATIENT
    field_aliases = OUTSAF_FIELD_ALIASES

    def _parse_row(self, row: dict[str, str], source: str) -> list[ClaimEvent]:
        """Parse OUTSAF row into ClaimEvents."""
        events = []

        patient_id = _find_field(row, self.field_aliases["patient_id"])
        if not patient_id:
            return events

        patient_key = create_patient_key(
            patient_id,
            salt=self.privacy_config.salt,
            algorithm=self.privacy_config.hash_algorithm,
        )

        claim_id = _find_field(row, self.field_aliases["claim_id"]) or ""
        claim_id_hash = create_claim_id_hash(claim_id, self.privacy_config.salt)

        from_date = _parse_date(_find_field(row, self.field_aliases["from_date"]))
        if not from_date:
            return events

        payment_str = _find_field(row, self.field_aliases["payment"])
        payment = None
        if payment_str and payment_str.replace(".", "").replace("-", "").isdigit():
            payment = float(payment_str)

        # HCPCS code
        hcpcs = _find_field(row, self.field_aliases["hcpcs"])
        if hcpcs and hcpcs.strip():
            code_system = _determine_code_system(hcpcs.strip(), "hcpcs")
            events.append(
                ClaimEvent(
                    patient_key=patient_key,
                    event_date=from_date,
                    setting=self.setting,
                    code_system=code_system,
                    code=hcpcs.strip(),
                    cost=payment,
                    raw_source=source,
                    claim_id_hash=claim_id_hash,
                )
            )

        # Revenue center code
        revenue = _find_field(row, self.field_aliases["revenue"])
        if revenue and revenue.strip():
            events.append(
                ClaimEvent(
                    patient_key=patient_key,
                    event_date=from_date,
                    setting=self.setting,
                    code_system=CodeSystem.REVENUE,
                    code=revenue.strip(),
                    raw_source=source,
                    claim_id_hash=claim_id_hash,
                )
            )

        return events


class PDEParser(MedicareClaimsParser):
    """Parser for PDE (Part D prescription drug events)."""

    setting = ClaimSetting.PDE
    field_aliases = PDE_FIELD_ALIASES

    def _parse_row(self, row: dict[str, str], source: str) -> list[ClaimEvent]:
        """Parse PDE row into ClaimEvents."""
        events = []

        patient_id = _find_field(row, self.field_aliases["patient_id"])
        if not patient_id:
            return events

        patient_key = create_patient_key(
            patient_id,
            salt=self.privacy_config.salt,
            algorithm=self.privacy_config.hash_algorithm,
        )

        event_id = _find_field(row, self.field_aliases["event_id"]) or ""
        event_id_hash = create_claim_id_hash(event_id, self.privacy_config.salt)

        service_date = _parse_date(_find_field(row, self.field_aliases["service_date"]))
        if not service_date:
            return events

        # NDC code
        ndc = _find_field(row, self.field_aliases["ndc"])
        if not ndc or not ndc.strip():
            return events

        # Quantity
        qty_str = _find_field(row, self.field_aliases["quantity"])
        quantity = None
        if qty_str and qty_str.replace(".", "").isdigit():
            quantity = int(float(qty_str))

        # Payment
        payment_str = _find_field(row, self.field_aliases["payment"])
        payment = None
        if payment_str and payment_str.replace(".", "").replace("-", "").isdigit():
            payment = float(payment_str)

        events.append(
            ClaimEvent(
                patient_key=patient_key,
                event_date=service_date,
                setting=self.setting,
                code_system=CodeSystem.NDC,
                code=ndc.strip(),
                quantity=quantity,
                cost=payment,
                raw_source=source,
                claim_id_hash=event_id_hash,
            )
        )

        return events


def get_parser_for_file(filepath: Path) -> Optional[MedicareClaimsParser]:
    """Get appropriate parser based on filename.

    Args:
        filepath: Path to claims file

    Returns:
        Parser instance or None if unknown file type
    """
    filename = filepath.name.upper()

    if "MEDPAR" in filename or "INPATIENT" in filename:
        return MEDPARParser()
    elif "NCH" in filename or "CARRIER" in filename or "BCARRIER" in filename:
        return NCHParser()
    elif "OUTSAF" in filename or "OUTPATIENT" in filename:
        return OUTSAFParser()
    elif "PDE" in filename or "PARTD" in filename or "PART_D" in filename:
        return PDEParser()
    elif "HHA" in filename or "HOME_HEALTH" in filename:
        # HHA uses similar structure to OUTSAF
        parser = OUTSAFParser()
        parser.setting = ClaimSetting.HHA
        return parser
    elif "HOSPICE" in filename:
        parser = OUTSAFParser()
        parser.setting = ClaimSetting.HOSPICE
        return parser
    elif "DME" in filename:
        parser = NCHParser()
        parser.setting = ClaimSetting.DME
        return parser

    return None


def parse_claims_directory(
    directory: Path,
    privacy_config: Optional[PrivacyConfig] = None,
    file_pattern: str = "*.csv",
) -> Generator[ClaimEvent, None, None]:
    """Parse all claims files in a directory.

    Args:
        directory: Directory to scan
        privacy_config: Privacy configuration
        file_pattern: Glob pattern for files

    Yields:
        ClaimEvent objects
    """
    directory = Path(directory)
    safe_logger = SafeLogger("claims_directory", privacy_config)
    safe_logger.info(f"Scanning claims directory: {directory}")

    for filepath in sorted(directory.glob(file_pattern)):
        parser = get_parser_for_file(filepath)
        if parser:
            if privacy_config:
                parser.privacy_config = privacy_config
            yield from parser.parse_file(filepath)
        else:
            safe_logger.warning(f"Unknown file type, skipping: {filepath.name}")
