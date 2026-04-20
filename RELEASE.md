# Release Process

This document describes the release process for Sybernix and QuASIM integrations.

## Versioning

We follow [Semantic Versioning 2.0.0](https://semver.org/):

- **MAJOR** version: Incompatible API changes
- **MINOR** version: Backwards-compatible functionality additions
- **PATCH** version: Backwards-compatible bug fixes

### Pre-release Versions

- **Alpha** (x.y.z-alpha.n): Early development, unstable
- **Beta** (x.y.z-beta.n): Feature-complete, testing in progress
- **RC** (x.y.z-rc.n): Release candidate, final testing

## Release Schedule

- **Major releases**: Quarterly
- **Minor releases**: Monthly
- **Patch releases**: As needed for critical fixes
- **Security patches**: Immediately upon fix validation

## Release Checklist

### Pre-release (1-2 weeks before)

- [ ] Create release branch: `release/vX.Y.Z`
- [ ] Update version numbers in:
  - [ ] `quasim/__init__.py`
  - [ ] `pyproject.toml`
  - [ ] `integrations/services/quasim-api/version.py`
  - [ ] Helm chart versions
  - [ ] Container image tags
- [ ] Update CHANGELOG.md with release notes
- [ ] Run full test suite: `make test`
- [ ] Run benchmarks: `make bench`
- [ ] Update documentation
- [ ] Generate SBOM: `syft . -o cyclonedx-json > sbom.json`
- [ ] Run security scans (CodeQL, Trivy, etc.)
- [ ] Review and update dependencies

### Release (Release day)

- [ ] Merge release branch to `main`
- [ ] Create and push git tag: `git tag -a vX.Y.Z -m "Release vX.Y.Z"`
- [ ] Verify CI/CD pipeline completes successfully
- [ ] Verify container images are built and pushed
- [ ] Verify Helm charts are packaged and published
- [ ] Sign container images with cosign
- [ ] Attach SBOM to release
- [ ] Create GitHub release with notes
- [ ] Update documentation site
- [ ] Announce release (if applicable)

### Post-release

- [ ] Monitor for issues and bug reports
- [ ] Update project roadmap
- [ ] Archive release artifacts
- [ ] Merge release branch back to `develop` (if using git-flow)

## Automated Release Workflow

The `.github/workflows/release.yml` workflow automates:

1. **Build**: Multi-arch container images
2. **Test**: Run full test suite
3. **Benchmark**: Execute performance tests
4. **Security**: Generate SBOM, run scans
5. **Sign**: Sign images with cosign
6. **Publish**: Push to GHCR, publish Helm charts
7. **Release**: Create GitHub release with artifacts

### Triggering a Release

Push a semver tag:

```bash
git tag -a v0.2.0 -m "Release v0.2.0"
git push origin v0.2.0
```

## Release Artifacts

Each release includes:

### Container Images

- `ghcr.io/robertringler/quasim-api:vX.Y.Z`
- `ghcr.io/robertringler/quasim-bench:vX.Y.Z`
- Multi-arch: `linux/amd64`, `linux/arm64`

### Helm Charts

- `quasim-api-vX.Y.Z.tgz`
- Includes values, templates, CRDs

### Source Archives

- `sybernix-vX.Y.Z.tar.gz`
- `sybernix-vX.Y.Z.zip`

### Security Artifacts

- `sbom-vX.Y.Z.json` (CycloneDX format)
- Image signatures (cosign)
- Vulnerability scan reports

### Documentation

- API reference (OpenAPI spec)
- Integration cookbooks
- Performance tuning guide
- Example configurations

## Rollback Procedure

If a release has critical issues:

1. **Immediate**: Revert to previous version

   ```bash
   kubectl set image deployment/quasim-api quasim-api=ghcr.io/robertringler/quasim-api:vX.Y.Z-1
   ```

2. **Short-term**: Create hotfix branch from previous tag

   ```bash
   git checkout -b hotfix/vX.Y.Z+1 vX.Y.Z-1
   ```

3. **Long-term**: Fix issue, create patch release
   - Apply fix to hotfix branch
   - Follow release process for patch version
   - Communicate issue and resolution

## Release Notes Template

```markdown
## vX.Y.Z - YYYY-MM-DD

### Features
- **[Component]** Brief description of new feature (#PR)

### Enhancements
- **[Component]** Brief description of enhancement (#PR)

### Bug Fixes
- **[Component]** Brief description of fix (#PR)

### Security
- **[Component]** Brief description of security fix (CVE-YYYY-XXXXX)

### Breaking Changes
- **[Component]** Description of breaking change and migration path

### Deprecations
- **[Component]** Deprecated feature, will be removed in vX.Y.Z

### Dependencies
- Updated dependency X from vA.B to vC.D
- Added dependency Y vE.F for feature Z

### Performance
- **[Benchmark]** X% improvement in throughput
- **[Benchmark]** Y% reduction in latency

### Documentation
- Added guide for [topic]
- Updated examples for [feature]

### Known Issues
- [Issue description] (workaround: [description])

### Contributors
Thank you to all contributors: @user1, @user2
```

## Version Support Policy

- **Latest major version**: Full support (features, bugs, security)
- **Previous major version**: Security and critical bugs only (12 months)
- **Older versions**: No support (upgrade recommended)

## Emergency Security Releases

For critical security vulnerabilities:

1. Develop fix in private security fork
2. Coordinate disclosure with reporter
3. Prepare patch releases for all supported versions
4. Publish CVE and security advisory
5. Release patches simultaneously
6. Send security notifications to users

## Compliance and Attestation

Each release includes:

- **SBOM**: Complete software bill of materials
- **Signatures**: Cryptographic signatures for all artifacts
- **Provenance**: Build provenance attestation (SLSA)
- **License**: License compliance report
- **Export**: ITAR classification (see compliance/EXPORT.md)

## Questions?

Contact the release team or open an issue for:

- Release timeline questions
- Version support inquiries
- Custom release requirements
- Enterprise support needs
