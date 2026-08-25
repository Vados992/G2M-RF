# Security and genomic privacy

Do not commit real participant genomes, phenotypes, consent forms, identifiers, access tokens or derived re-identification data.

Whole-genome data are inherently identifying. Production deployments should use encrypted storage, least-privilege access, immutable audit logs, isolated compute, pseudonymous participant IDs and institution-approved retention/deletion rules.

The repository `.gitignore` blocks common generated datasets, but this is not a substitute for data-governance controls.
