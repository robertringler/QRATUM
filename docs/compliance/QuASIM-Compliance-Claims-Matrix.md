# QuASIM Compliance Claims Matrix

Purpose: Centralized, evidence-backed claims for audits, RFPs, and customer due diligence. Each claim must be supported by current, verifiable evidence.

How to use:

- Update “Evidence location” with links (doc repo, GRC item, certificate URL).
- Keep “Last verified” current and set status accurately.
- Attach authoritative artifacts (certificates, reports, attestations, CMVP entries, contracts).

Status key:

- Validated = Independently verified and current
- In progress = Work underway, not yet fully verified
- Planned = Not started
- Not applicable = Outside scope

| Program / Standard | Scope (Org/Product/Service) | What “compliant” means in practice | Evidence to keep on file | Evidence owner | Current status | Last verified | Notes / dependencies |
|---|---|---|---|---|---|---|---|
| ISO/IEC 27001:2022 | Organization and ISMS scope | Accredited certification of ISMS within defined scope | ISO cert (number), scope statement, SoA, audit report(s), CB accreditation | GRC / Security | Planned |  | Certification body, validity dates |
| SOC 2 Type II | Defined system (service) | Independent CPA audit over period; report covers selected TSCs | SOC 2 Type II report (redacted + NDA copy), bridge letter, management assertion | GRC / Security | Planned |  | Report period, categories (Security, Availability, etc.) |
| NIST SP 800-171 Rev. 3 | Handling of CUI | Implemented controls, SSP/POA&M, SPRS score, incident handling | SSP, POA&M, SPRS submission receipt, control mappings, MOU/NDA with primes | Compliance / SecEng | Planned |  | Include enclave boundary and hosting details |
| CMMC 2.0 Level 2 | Org/Enclave processing CUI | C3PAO assessment and certificate (unless self-assessed per contract) | Official CMMC certificate, scoping docs, assessment report summary | Compliance | Planned |  | Link to CMMC marketplace entry if available |
| NIST SP 800-53 Rev. 5 | Org/Authorizations | Control implementation evidence; typically via ATO (e.g., FedRAMP/agency) | Control implementations, SAP/SAR, ATO letter (if any), RTM | SecEng / GRC | Planned |  | Not a certification; cite authorizations if applicable |
| DFARS 252.204-7012/7019/7020/7021 | DoD contracting org | SPRS score submitted; incident reporting; flowdowns; CUI handling | SPRS proof, incident response readiness, flowdown clauses, cloud service compliance | Contracts / Compliance | Planned |  | Reference 800-171/CMMC artifacts |
| FIPS 140-3 (CMVP) | Cryptographic modules used | Only specific crypto modules can be validated | CMVP certificate numbers and links, module version mapping, boundary docs | SecEng | Planned |  | List modules actually used by QuASIM |
| NDAA Section 889 | Org supply chain | No procurement/use of covered telecom equipment/services | Signed NDAA 889 attestation, supplier attestations, SCRM records | Procurement / Legal | Planned |  | Include exceptions analysis (none expected) |
| ITAR | Export controls program | Registration (if required), TCPs, access controls, recordkeeping | ITAR registration (if applicable), TCPs, screening logs, training | Legal / Trade | Planned |  | State whether service is permitted for ITAR data |
| EAR | Export controls program | Classifications (ECCN/USML), screening, license management | ECCNs, CCATS (if any), restricted party screening logs, license records | Legal / Trade | Planned |  | Include public classification matrix |
| DO-178C Level A | Program-specific software | Certification only within aircraft/program context | Plan set (PSAC, SDP, SVP, etc.), DER/authority acceptance, compliance reports | Eng / Cert | Planned |  | No general product-wide “Level A” claim |
| SOC 2 Bridge Letter | Defined system | Statement covering gap between audit period and present | Bridge letter, change log, exceptions | GRC | Planned |  | Required when report is >3 months old |
| Privacy (GDPR/CCPA) | Org/service | Lawful basis, DPA, DPIAs, user rights processes | DPA, RoPA, DPIAs, subprocessor list, privacy notices | Legal / Privacy | Planned |  | Not a certification; disclose subprocessors |
| Business Continuity/DR | Service | Tested plans meeting RTO/RPO | BC/DR plans, test results, evidence of backups, failover docs | SRE / SecEng | Planned |  | Map to SOC 2 Availability if applicable |

## FIPS 140-3 module inventory (detail)

Record all crypto in use and validation status.

| Component | Module name and version | CMVP Cert # | Operational environment | Used algorithms | Validation status | Evidence link |
|---|---|---|---|---|---|---|
| [placeholder] | [e.g., OpenSSL FIPS Provider 3.0.x] | [#xxxx] | [OS/CPU] | [SHA-256, AES-GCM…] | [Validated/In process] | [CMVP URL] |

## 800-171/CMMC scoping snapshot

- CUI enclave boundary: [placeholder]
- Hosting: [GovCloud/Commercial/Azure GCC High/etc.]
- SSP/POA&M location: [link]
- SPRS score: [###] as of [date]
- Incident reporting process: [link]

## NDAA 889 attestation snapshot

- Corporate attestation signed by: [name/title], dated [date]
- Supplier screening cadence: [e.g., annual with onboarding checks]
- Covered telecom check: [none used / details]

## Change log

- [date]: Initial matrix created.
