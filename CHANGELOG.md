# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- AnatEM data files
- Original PubTator training data

### Fixed

- Tagging script yielded incorrect spans when text contained double spaces. Now trailing space in sentences is preserved

### Removed

- Unused data files
- Placeholders for local models
- Original scripts from the AIONER repo 

## v0.1.1 - 2024-11-12

NER models implemented directly from HF using the AIONER scheme.

### Added

- Processed training data.
- Training and execution scripts for the new models.

### Changed

- Visualisation example now follows the more straightforward implementation.

### Fixed

- Test evaluation script reverted to original from AIONER repo for better comparison.

### Deprecated

- Models trained following the original AIONER scripts will be soon removed.

## v0.0.1 - 2024-08-30

- Updated documentation to adhere to semantic versioning and keep a changelog.

## v0.0.0

Initial version of the Named Entity Recognition module.

