# Architecture Diagrams

This directory contains PlantUML source files for AuDRA-Rad architecture diagrams.

## Diagram Files

### C4 Model Diagrams

1. **context.puml** - System Context Diagram (C4 Level 1)
   - Shows AuDRA-Rad in its operational environment
   - External actors: Clinicians, Patients
   - External systems: RIS, EHR, LLM services

2. **container.puml** - Container Diagram (C4 Level 2)
   - Major runtime components
   - Technology stack
   - Communication patterns

3. **sequence_processing.puml** - Report Processing Sequence
   - Critical workflow showing end-to-end processing
   - Tool execution flow
   - Timing and interactions

4. **deployment_aws.puml** - AWS Deployment Architecture
   - Cloud infrastructure topology
   - EKS cluster layout
   - External service integrations

## Rendering PlantUML Diagrams

### Option 1: Online PlantUML Server

Visit http://www.plantuml.com/plantuml/uml/ and paste the contents of any `.puml` file.

### Option 2: VS Code Extension

1. Install the "PlantUML" extension by jebbs
2. Open any `.puml` file
3. Press `Alt+D` to preview

### Option 3: Command Line

```bash
# Install PlantUML
brew install plantuml  # macOS
# or
sudo apt-get install plantuml  # Linux

# Generate PNG
plantuml context.puml

# Generate SVG (vector, recommended for docs)
plantuml -tsvg context.puml

# Generate all diagrams
plantuml *.puml
```

### Option 4: Docker

```bash
docker run --rm -v $(pwd):/data plantuml/plantuml:latest \
  -tsvg /data/context.puml
```

## Diagram Standards

### Color Coding

- **Light Blue (#D4E4F7):** User-facing components (Web UI, Clinician workspace)
- **Light Red (#FFE5E5):** API Gateway, external interfaces
- **Light Yellow (#FFF4D6):** Core agent orchestration
- **Light Orange (#FFE8CC):** Parsing, task generation
- **Light Green (#E0F2E9):** Guideline system, knowledge retrieval
- **Light Purple (#E8E0F2):** LLM services, AI inference
- **Light Pink (#FFE8E8):** EHR integration, external systems

### Font and Styling

- **Title:** 16pt, bold
- **Component labels:** 13pt
- **Descriptions:** 10pt, italic
- **Arrows:** 2px width, labeled with protocol/format

## Exporting for Documentation

For inclusion in README.md or technical docs:

1. **Generate SVG** (preferred for web):
   ```bash
   plantuml -tsvg *.puml
   ```

2. **Generate PNG** (for presentations):
   ```bash
   plantuml -tpng *.puml
   ```

3. **Embed in Markdown**:
   ```markdown
   ![Context Diagram](docs/diagrams/context.svg)
   ```

## Diagram Maintenance

- **Review Frequency:** Quarterly or when major architecture changes occur
- **Update Trigger:** New services added, deployment topology changes, major refactoring
- **Version Control:** All diagrams are versioned in Git
- **Validation:** Ensure diagrams match actual implementation (cross-check with codebase)

## Additional Resources

- [C4 Model Documentation](https://c4model.com/)
- [PlantUML Official Site](https://plantuml.com/)
- [PlantUML C4 Library](https://github.com/plantuml-stdlib/C4-PlantUML)
- [AuDRA-Rad System Design Doc](../SYSTEM_DESIGN.md)
- [AuDRA-Rad Architecture Doc](../ARCHITECTURE.md)
