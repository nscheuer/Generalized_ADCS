# Contributing to Generalized ADCS

Thank you for your interest in **Generalized ADCS**. This project is a research-grade framework for satellite attitude determination and control (ADCS), and contributions from students, researchers, and industry engineers are welcome.

## Code of Conduct

By participating in this project, you agree to follow the standards in [CODE_OF_CONDUCT.md](https://github.com/nscheuer/Generalized_ADCS/blob/main/CODE_OF_CONDUCT.md).

If you experience or witness unacceptable behavior, report it to the maintainers at **nscheuer@mit.edu**.

If this project is not currently using a conduct email, open a GitHub issue and title it `Code of Conduct report` (avoid including sensitive personal details in public reports).

## Ways to Contribute

There are many ways you can help improve the project, including:

- Developing new controllers (LQR, MPC, adaptive control, etc.)
- Adding new sensor models (star trackers, sun sensors, gyros, magnetometers)
- Implementing new estimators (EKF, UKF, Particle Filters)
- Creating new simulation scenarios or benchmark cases
- Expanding testing and validation coverage
- Improving or adding documentation
- Fixing bugs and optimizing performance
- Enhancing visualization and animation tools

## Places to Start

If you are new to the project, these are good first contributions:

- Review and improve documentation for clarity, setup, and examples
- Triage open issues (reproduce, label, and suggest scope)
- Review open pull requests for test coverage and code clarity
- Run and improve existing example scenarios in [examples](https://github.com/nscheuer/Generalized_ADCS/tree/main/examples)

## Development Setup

Before contributing, follow the installation instructions to ensure a compatible development environment:

- [Installation Guide](https://nscheuer.github.io/Generalized_ADCS/installation/index.html)

For contributors using a fork-based workflow:

1. Fork the repository on GitHub.
2. Clone your fork:

```bash
git clone https://github.com/<your-username>/Generalized_ADCS.git
cd Generalized_ADCS
```

3. Add the main repository as `upstream`:

```bash
git remote add upstream https://github.com/<upstream-owner>/Generalized_ADCS.git
git remote -v
```

4. Create a feature branch:

```bash
git checkout -b feature/<short-description>
```

5. Keep your branch up to date with upstream `main`:

```bash
git fetch upstream
git checkout main
git pull upstream main
git checkout feature/<short-description>
git rebase main
```

6. Push your branch to your fork:

```bash
git push -u origin feature/<short-description>
```

## Reporting Bugs

### Check first

Before filing a new bug report, search existing issues and pull requests to avoid duplicates.

### Bug report template

When opening an issue, include the following:

- Clear title that summarizes the problem
- Environment details (OS, Python version, dependency versions, commit hash if relevant)
- Minimal reproduction steps (numbered and deterministic)
- Expected behavior
- Observed behavior
- Error logs or stack traces
- Example code or configuration needed to reproduce
- Screenshots or plots (if relevant for simulation/visualization issues)
- Whether this is a regression and the last known good version

## Reporting Security Vulnerabilities

Do not report security vulnerabilities through public issues.

Use one of these private channels:
- Maintainer contact: **nscheuer@mit.edu**
- add more?

Include impact, affected components, reproduction details, and any suggested mitigation.

## Questions and Community

For usage questions, design discussion, and collaboration: GitHub Discussions

## Testing and Documentation

For new features or significant changes, please ensure that:

- Relevant tests are added or updated
- Existing tests pass locally before opening a pull request
- Sphinx documentation builds without errors
- User-facing behavior changes are documented

Helpful guides:

- [Sphinx Documentation](https://nscheuer.github.io/Generalized_ADCS/contributing/documentation.html)
- [Testing Documentation](https://nscheuer.github.io/Generalized_ADCS/contributing/testing.html)

## License Terms for Contributions

By submitting code, documentation, or other contributions to this repository, you agree that your contribution is provided under the same license as the project: the MIT License (see [LICENSE](https://github.com/nscheuer/Generalized_ADCS/blob/main/LICENSE)).

## Becoming a Maintainer

Long-term contributors may be invited to become maintainers based on:

- Consistent, high-quality contributions over time
- Constructive and respectful code reviews
- Reliability in triaging issues and supporting releases
- Alignment with project scope and standards

Maintainer access is granted at the discretion of existing maintainers.

### Thank you for your contributions!