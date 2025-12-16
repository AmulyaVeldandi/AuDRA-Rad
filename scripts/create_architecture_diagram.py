#!/usr/bin/env python3
"""Generate an updated architecture diagram for AuDRA-Rad."""

from PIL import Image, ImageDraw, ImageFont
import os

# Diagram dimensions
WIDTH = 1400
HEIGHT = 1000
PADDING = 40

# Colors
BG_COLOR = "#F8F9FA"
CLINICIAN_COLOR = "#D4E4F7"  # Light blue
API_COLOR = "#FFE5E5"  # Light red
AGENT_COLOR = "#FFF4D6"  # Light yellow
PARSER_COLOR = "#FFE8CC"  # Light orange
GUIDELINE_COLOR = "#E0F2E9"  # Light green
LLM_COLOR = "#E8E0F2"  # Light purple
VECTOR_COLOR = "#E0F2E9"  # Light green
EHR_COLOR = "#FFE8E8"  # Light pink
OBSERVABILITY_COLOR = "#E8F4F8"  # Light cyan
FRONTEND_COLOR = "#E8E8F8"  # Light indigo
TEXT_COLOR = "#2C3E50"
BORDER_COLOR = "#95A5A6"
ARROW_COLOR = "#34495E"

def create_rounded_rectangle(draw, coords, radius, fill, outline):
    """Draw a rounded rectangle."""
    x1, y1, x2, y2 = coords

    # Draw rectangles
    draw.rectangle([x1 + radius, y1, x2 - radius, y2], fill=fill, outline=outline)
    draw.rectangle([x1, y1 + radius, x2, y2 - radius], fill=fill, outline=outline)

    # Draw corners
    draw.pieslice([x1, y1, x1 + 2*radius, y1 + 2*radius], 180, 270, fill=fill, outline=outline)
    draw.pieslice([x2 - 2*radius, y1, x2, y1 + 2*radius], 270, 360, fill=fill, outline=outline)
    draw.pieslice([x1, y2 - 2*radius, x1 + 2*radius, y2], 90, 180, fill=fill, outline=outline)
    draw.pieslice([x2 - 2*radius, y2 - 2*radius, x2, y2], 0, 90, fill=fill, outline=outline)

def draw_arrow(draw, start, end, color=ARROW_COLOR, width=2):
    """Draw an arrow from start to end."""
    x1, y1 = start
    x2, y2 = end

    # Draw line
    draw.line([x1, y1, x2, y2], fill=color, width=width)

    # Draw arrowhead
    import math
    angle = math.atan2(y2 - y1, x2 - x1)
    arrow_length = 10
    arrow_angle = math.pi / 6

    point1_x = x2 - arrow_length * math.cos(angle - arrow_angle)
    point1_y = y2 - arrow_length * math.sin(angle - arrow_angle)
    point2_x = x2 - arrow_length * math.cos(angle + arrow_angle)
    point2_y = y2 - arrow_length * math.sin(angle + arrow_angle)

    draw.polygon([x2, y2, point1_x, point1_y, point2_x, point2_y], fill=color)

def draw_text_box(draw, coords, text, bg_color, font, small_font=None):
    """Draw a box with text inside."""
    x1, y1, x2, y2 = coords
    create_rounded_rectangle(draw, coords, 8, bg_color, BORDER_COLOR)

    # Parse text for title and subtitle
    lines = text.split('\n')
    title = lines[0]
    subtitles = lines[1:] if len(lines) > 1 else []

    # Draw title
    bbox = draw.textbbox((0, 0), title, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    text_x = x1 + (x2 - x1 - text_width) // 2
    text_y = y1 + 10
    draw.text((text_x, text_y), title, fill=TEXT_COLOR, font=font)

    # Draw subtitles
    if subtitles and small_font:
        current_y = text_y + text_height + 5
        for subtitle in subtitles:
            bbox = draw.textbbox((0, 0), subtitle, font=small_font)
            text_width = bbox[2] - bbox[0]
            text_x = x1 + (x2 - x1 - text_width) // 2
            draw.text((text_x, current_y), subtitle, fill=TEXT_COLOR, font=small_font)
            current_y += (bbox[3] - bbox[1]) + 3

# Create image
img = Image.new('RGB', (WIDTH, HEIGHT), BG_COLOR)
draw = ImageDraw.Draw(img)

# Try to use a nice font, fall back to default if not available
try:
    title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 13)
    small_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 10)
except:
    title_font = ImageFont.load_default()
    font = ImageFont.load_default()
    small_font = ImageFont.load_default()

# Draw title
draw.text((WIDTH // 2 - 200, 20), "AuDRA-Rad System Architecture", fill=TEXT_COLOR, font=title_font)

# Layer 1: User Interface (Top)
draw_text_box(draw, (50, 60, 230, 120), "Clinician Workspace\nWeb Interface\nReport Upload", CLINICIAN_COLOR, font, small_font)
draw_text_box(draw, (250, 60, 430, 120), "React Frontend\nReport Viewer\nFindings & Tasks", FRONTEND_COLOR, font, small_font)

# Layer 2: API Gateway
draw_text_box(draw, (50, 150, 430, 210), "FastAPI REST API\nRate Limiting | Authentication\n/process-report | /batch-process | /health", API_COLOR, font, small_font)

# Layer 3: Agent Orchestrator (Center)
draw_text_box(draw, (50, 240, 430, 330), "AuDRA Agent (ReAct)\nSession Manager | State Tracking\nDecision Trace | Safety Validation", AGENT_COLOR, font, small_font)

# Layer 4: Processing Components (Left side)
# Parsers
draw_text_box(draw, (50, 360, 210, 440), "Parsers & Context\nFHIR Parser\nReport Parser\nNarrative Extraction", PARSER_COLOR, font, small_font)

# Guideline System
draw_text_box(draw, (220, 360, 380, 440), "Guideline System\nRetriever | Matcher\nIndexer", GUIDELINE_COLOR, font, small_font)

# Task Generator
draw_text_box(draw, (390, 360, 550, 440), "Task Generator\nFHIR Builder\nServiceRequest\nOrder Creation", PARSER_COLOR, font, small_font)

# Layer 5: External Services (Right side - split into columns)
# LLM Services
draw_text_box(draw, (580, 60, 740, 140), "LLM Services\nNVIDIA NIM\nOllama (Local)\nNemotron", LLM_COLOR, font, small_font)

# Embedding Services
draw_text_box(draw, (760, 60, 920, 140), "Embedding Services\nNVIDIA NIM\nOllama (Local)\nnomic-embed-text", LLM_COLOR, font, small_font)

# Vector Stores
draw_text_box(draw, (940, 60, 1100, 140), "Vector Stores\nOpenSearch\nSimple Vector Store\nGuideline Chunks", VECTOR_COLOR, font, small_font)

# EHR Integration
draw_text_box(draw, (1120, 60, 1280, 140), "EHR Integration\nFHIR Client\nOrder Submission\nTask Tracking", EHR_COLOR, font, small_font)

# Observability Layer (Bottom)
draw_text_box(draw, (580, 360, 830, 440), "Observability\nStructured Logging\nMetrics & Traces\nDecision Audit", OBSERVABILITY_COLOR, font, small_font)

# Data Layer (Bottom right)
draw_text_box(draw, (850, 360, 1010, 440), "Guidelines Data\nFleischner 2017\nACR Liver/Lung\nEvidence-based", GUIDELINE_COLOR, font, small_font)

# Safety & Validation
draw_text_box(draw, (1030, 360, 1190, 440), "Safety & Validation\nRule-based Guards\nConflict Detection\nHuman Review Flags", PARSER_COLOR, font, small_font)

# External Systems
draw_text_box(draw, (1210, 360, 1350, 440), "External Services\nHospital EHR (FHIR)\nRadiology RIS\nInbox", EHR_COLOR, font, small_font)

# Draw arrows showing data flow
# User to API
draw_arrow(draw, (140, 120), (140, 150))
draw_arrow(draw, (340, 120), (340, 150))

# API to Agent
draw_arrow(draw, (240, 210), (240, 240))

# Agent to Processing Components
draw_arrow(draw, (130, 330), (130, 360))
draw_arrow(draw, (300, 330), (300, 360))
draw_arrow(draw, (470, 330), (470, 360))

# Agent to External Services (right side flows)
# To LLM
draw_arrow(draw, (430, 270), (580, 100), color="#9B59B6")
# To Embeddings
draw_arrow(draw, (430, 285), (760, 100), color="#9B59B6")
# To Vector Store
draw_arrow(draw, (430, 300), (940, 100), color="#27AE60")
# To EHR
draw_arrow(draw, (430, 315), (1120, 100), color="#E74C3C")

# Guideline System to Vector Store
draw_arrow(draw, (380, 390), (940, 140), color="#27AE60")

# Task Generator to EHR
draw_arrow(draw, (550, 400), (1120, 140), color="#E74C3C")

# Agent to Observability
draw_arrow(draw, (430, 285), (580, 360), color="#3498DB")

# Guidelines Data to Guideline System
draw_arrow(draw, (850, 390), (380, 390), color="#27AE60")

# Safety to Agent (feedback loop)
draw_arrow(draw, (1030, 360), (430, 315), color="#E67E22")

# EHR to External Systems
draw_arrow(draw, (1280, 100), (1280, 360), color="#E74C3C")

# Add legend
legend_y = 470
draw.text((50, legend_y), "Data Flow Legend:", fill=TEXT_COLOR, font=font)
draw.line([50, legend_y + 25, 100, legend_y + 25], fill="#9B59B6", width=2)
draw.text((110, legend_y + 18), "LLM/Embedding Calls", fill=TEXT_COLOR, font=small_font)
draw.line([280, legend_y + 25, 330, legend_y + 25], fill="#27AE60", width=2)
draw.text((340, legend_y + 18), "Guideline Retrieval", fill=TEXT_COLOR, font=small_font)
draw.line([510, legend_y + 25, 560, legend_y + 25], fill="#E74C3C", width=2)
draw.text((570, legend_y + 18), "EHR Integration", fill=TEXT_COLOR, font=small_font)
draw.line([720, legend_y + 25, 770, legend_y + 25], fill="#3498DB", width=2)
draw.text((780, legend_y + 18), "Observability", fill=TEXT_COLOR, font=small_font)
draw.line([920, legend_y + 25, 970, legend_y + 25], fill="#E67E22", width=2)
draw.text((980, legend_y + 18), "Safety Validation", fill=TEXT_COLOR, font=small_font)

# Add processing flow description
flow_y = 520
draw.text((50, flow_y), "Processing Flow:", fill=TEXT_COLOR, font=font)
draw.text((50, flow_y + 25), "1. Clinician uploads radiology report via web interface", fill=TEXT_COLOR, font=small_font)
draw.text((50, flow_y + 45), "2. API validates & routes to AuDRA Agent (ReAct orchestrator)", fill=TEXT_COLOR, font=small_font)
draw.text((50, flow_y + 65), "3. Agent executes: Parse → Retrieve Guidelines → Match Recommendation → Validate Safety → Generate Task", fill=TEXT_COLOR, font=small_font)
draw.text((50, flow_y + 85), "4. LLM (cloud/local) performs reasoning; Vector store retrieves relevant guidelines", fill=TEXT_COLOR, font=small_font)
draw.text((50, flow_y + 105), "5. Safety validation checks for conflicts & flags high-risk cases for human review", fill=TEXT_COLOR, font=small_font)
draw.text((50, flow_y + 125), "6. FHIR-compliant tasks/orders pushed to EHR; All decisions logged for audit", fill=TEXT_COLOR, font=small_font)

# Deployment modes
deploy_y = 670
draw.text((50, deploy_y), "Deployment Modes:", fill=TEXT_COLOR, font=font)
draw.text((50, deploy_y + 25), "• Cloud: NVIDIA NIM (LLM + Embeddings) + OpenSearch", fill=TEXT_COLOR, font=small_font)
draw.text((50, deploy_y + 45), "• Local: Ollama (Llama 3.1 8B + nomic-embed-text) + Simple Vector Store", fill=TEXT_COLOR, font=small_font)
draw.text((50, deploy_y + 65), "• Hybrid: Mix of cloud LLM with local embeddings/storage", fill=TEXT_COLOR, font=small_font)

# Key features
features_y = 760
draw.text((50, features_y), "Key Features:", fill=TEXT_COLOR, font=font)
draw.text((50, features_y + 25), "✓ Dual LLM backend support (cloud & local)", fill=TEXT_COLOR, font=small_font)
draw.text((50, features_y + 45), "✓ FHIR-compliant integration", fill=TEXT_COLOR, font=small_font)
draw.text((50, features_y + 65), "✓ Evidence-based guideline matching (Fleischner, ACR)", fill=TEXT_COLOR, font=small_font)

draw.text((400, features_y + 25), "✓ Safety validation & human review flags", fill=TEXT_COLOR, font=small_font)
draw.text((400, features_y + 45), "✓ Complete decision tracing & audit logs", fill=TEXT_COLOR, font=small_font)
draw.text((400, features_y + 65), "✓ Automated follow-up order generation", fill=TEXT_COLOR, font=small_font)

draw.text((750, features_y + 25), "✓ Real-time processing (3-5 seconds)", fill=TEXT_COLOR, font=small_font)
draw.text((750, features_y + 45), "✓ Batch processing support", fill=TEXT_COLOR, font=small_font)
draw.text((750, features_y + 65), "✓ Interactive web dashboard", fill=TEXT_COLOR, font=small_font)

# Save the image
output_path = '/Users/amulyaveldandi/Desktop/AuDRA-Rad/assets/architecture_diagram.png'
img.save(output_path)
print(f"Architecture diagram saved to {output_path}")
