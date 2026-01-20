# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security issue, please report it responsibly.

### Reporting Process

1. **Do not** open a public GitHub issue for security vulnerabilities
2. Email security concerns to: `security@example.com` (or open a private security advisory on GitHub)
3. Include the following in your report:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

### What to Expect

- **Acknowledgment**: Within 48 hours of your report
- **Initial Assessment**: Within 7 days
- **Resolution Timeline**: Depends on severity (critical: 7 days, high: 30 days, medium: 90 days)

### Coordinated Disclosure

We follow coordinated disclosure practices:
- We will work with you to understand and resolve the issue
- We will credit you in the security advisory (unless you prefer anonymity)
- We request a 90-day disclosure window before public disclosure

## ICS/OT Security Considerations

This software is designed for industrial control system (ICS) and operational technology (OT) environments. Please consider the following:

### Safety-Critical Notes

1. **Read-Only Operations**: The OPC UA connector enforces read-only access by default. Do not modify this behavior in production environments without proper safety analysis.

2. **Network Segmentation**: Deploy the swarm agents within properly segmented networks following IEC 62443 guidelines.

3. **Authentication**: Always configure proper authentication for:
   - OPC UA server connections
   - MQTT broker access
   - Federated learning server endpoints
   - API access

4. **Data Sensitivity**: Tag names and process data may contain sensitive operational information. Ensure proper access controls and encryption in transit.

### Deployment Recommendations

- Use TLS/SSL for all network communications
- Deploy behind a reverse proxy with proper authentication
- Enable audit logging for all mapping decisions
- Review consensus results before committing to production AAS registries
- Regularly update dependencies to patch known vulnerabilities

## Security Features

This project includes several security-conscious design decisions:

- **Minimal Permissions**: OPC UA connections request only read permissions
- **Input Validation**: All API inputs are validated via Pydantic models
- **Structured Logging**: Security events are logged with correlation IDs
- **No Hardcoded Secrets**: All sensitive configuration via environment variables

## Scope

The following are **in scope** for security reports:
- Remote code execution
- Authentication/authorization bypass
- SQL/command injection
- Cross-site scripting (XSS) in UI components
- Sensitive data exposure
- Denial of service vulnerabilities

The following are **out of scope**:
- Issues requiring physical access
- Social engineering attacks
- Issues in dependencies (report to upstream maintainers)
- Theoretical vulnerabilities without proof of concept
