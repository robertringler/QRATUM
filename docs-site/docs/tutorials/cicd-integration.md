# CI/CD Integration

Set up automated testing, building, and deployment pipelines for QRATUM.

## GitHub Actions

QRATUM uses GitHub Actions for CI/CD. Key workflows:

| Workflow | Purpose | Trigger |
|----------|---------|---------|
| `ci.yml` | Main CI pipeline | Push, PR |
| `pr-compliance.yml` | Compliance checks | PR |
| `pr-defense-compliance.yml` | Security scanning | PR |
| `quasim-validation-sweep.yml` | Full validation | Schedule |

## Basic CI Pipeline

Create `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.10', '3.11', '3.12']

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
          cache: 'pip'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -e ".[dev,quantum]"

      - name: Lint with ruff
        run: ruff check .

      - name: Format check
        run: ruff format --check .

      - name: Run tests
        run: pytest tests/ -v --cov=quasim --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v4
        with:
          files: coverage.xml

  build:
    runs-on: ubuntu-latest
    needs: test

    steps:
      - uses: actions/checkout@v4

      - name: Build Docker image
        run: docker build -t qratum:${{ github.sha }} .

      - name: Push to registry
        if: github.ref == 'refs/heads/main'
        run: |
          echo "${{ secrets.DOCKER_PASSWORD }}" | docker login -u "${{ secrets.DOCKER_USERNAME }}" --password-stdin
          docker tag qratum:${{ github.sha }} qratum/qratum:latest
          docker push qratum/qratum:latest
```

## Compliance Pipeline

Create `.github/workflows/compliance.yml`:

```yaml
name: Compliance Checks

on:
  pull_request:
    branches: [main]

jobs:
  compliance:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install bandit pip-audit

      - name: Security scan (Bandit)
        run: bandit -r quasim/ -ll -ii

      - name: Dependency audit
        run: pip-audit

      - name: License check
        run: |
          pip install pip-licenses
          pip-licenses --format=csv > licenses.csv
          # Fail if GPL dependencies found
          if grep -i "GPL" licenses.csv; then exit 1; fi

      - name: SBOM generation
        run: |
          pip install syft
          syft . -o spdx-json > sbom.spdx.json

      - name: Upload SBOM
        uses: actions/upload-artifact@v4
        with:
          name: sbom
          path: sbom.spdx.json
```

## Deployment Pipeline

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy

on:
  push:
    tags:
      - 'v*'

jobs:
  deploy:
    runs-on: ubuntu-latest
    environment: production

    steps:
      - uses: actions/checkout@v4

      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: us-west-2

      - name: Login to Amazon ECR
        id: login-ecr
        uses: aws-actions/amazon-ecr-login@v2

      - name: Build and push
        env:
          ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
          IMAGE_TAG: ${{ github.ref_name }}
        run: |
          docker build -t $ECR_REGISTRY/qratum:$IMAGE_TAG .
          docker push $ECR_REGISTRY/qratum:$IMAGE_TAG

      - name: Update kubeconfig
        run: aws eks update-kubeconfig --name qratum-prod --region us-west-2

      - name: Deploy to Kubernetes
        run: |
          kubectl set image deployment/qratum-api \
            api=${{ steps.login-ecr.outputs.registry }}/qratum:${{ github.ref_name }} \
            -n qratum

      - name: Verify deployment
        run: kubectl rollout status deployment/qratum-api -n qratum --timeout=300s
```

## GitLab CI/CD

If using GitLab, create `.gitlab-ci.yml`:

```yaml
stages:
  - test
  - build
  - deploy

variables:
  PIP_CACHE_DIR: "$CI_PROJECT_DIR/.pip"

cache:
  paths:
    - .pip/

test:
  stage: test
  image: python:3.11
  script:
    - pip install -r requirements.txt
    - pip install -e ".[dev,quantum]"
    - ruff check .
    - pytest tests/ -v --cov=quasim
  coverage: '/TOTAL.*\s+(\d+%)/'

build:
  stage: build
  image: docker:24
  services:
    - docker:24-dind
  script:
    - docker build -t $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA .
    - docker push $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
  only:
    - main

deploy:
  stage: deploy
  image: bitnami/kubectl:latest
  script:
    - kubectl set image deployment/qratum-api api=$CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
  only:
    - tags
  environment:
    name: production
```

## Jenkins Pipeline

If using Jenkins, create `Jenkinsfile`:

```groovy
pipeline {
    agent any
    
    environment {
        DOCKER_IMAGE = 'qratum/qratum'
    }
    
    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }
        
        stage('Test') {
            agent {
                docker {
                    image 'python:3.11'
                }
            }
            steps {
                sh 'pip install -r requirements.txt'
                sh 'pip install -e ".[dev,quantum]"'
                sh 'ruff check .'
                sh 'pytest tests/ -v --cov=quasim'
            }
        }
        
        stage('Build') {
            steps {
                script {
                    docker.build("${DOCKER_IMAGE}:${BUILD_NUMBER}")
                }
            }
        }
        
        stage('Deploy') {
            when {
                branch 'main'
            }
            steps {
                script {
                    docker.withRegistry('', 'docker-hub-creds') {
                        docker.image("${DOCKER_IMAGE}:${BUILD_NUMBER}").push('latest')
                    }
                }
            }
        }
    }
    
    post {
        always {
            cleanWs()
        }
    }
}
```

## Local Testing

Test CI pipelines locally with [act](https://github.com/nektos/act):

```bash
# Install act
brew install act  # macOS
# or: curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash

# Run all workflows
act

# Run specific workflow
act -W .github/workflows/ci.yml

# Run specific job
act -j test
```

## Best Practices

### 1. Cache Dependencies

```yaml
- uses: actions/cache@v4
  with:
    path: ~/.cache/pip
    key: ${{ runner.os }}-pip-${{ hashFiles('requirements.txt') }}
```

### 2. Matrix Testing

```yaml
strategy:
  matrix:
    python-version: ['3.10', '3.11', '3.12']
    os: [ubuntu-latest, macos-latest]
```

### 3. Parallel Jobs

```yaml
jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - run: ruff check .

  test:
    runs-on: ubuntu-latest
    steps:
      - run: pytest tests/

  # Runs in parallel with lint and test
```

### 4. Required Status Checks

Configure branch protection in GitHub:

1. Settings → Branches → Branch protection rules
2. Require status checks: `test`, `lint`, `compliance`
3. Require pull request reviews

## Next Steps

- [Kubernetes Deployment](kubernetes-deployment.md) - Production deployment
- [Compliance](../compliance/index.md) - Security requirements
- [Troubleshooting](../advanced/troubleshooting.md) - Common issues
