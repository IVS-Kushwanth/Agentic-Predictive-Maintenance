"""
Script to generate the 4 DOCX documents requested in Part 14:
- Complete_Academic_Project_Report.docx
- Complete_Technical_Implementation_Manual.docx
- End_User_Instruction_Manual.docx
- Project_Phase_Wise_Reports.docx
"""

import os
import re
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
DOC_DIR = os.path.join(ROOT_DIR, "documentation")

def parse_markdown_to_docx(doc, md_text):
    """Simple parser converting markdown headers, bullet points, tables, and paragraphs to docx."""
    lines = md_text.splitlines()
    in_code_block = False
    code_lines = []

    for line in lines:
        stripped = line.strip()

        # Code block handling
        if stripped.startswith("```"):
            if in_code_block:
                # End of code block
                in_code_block = False
                p = doc.add_paragraph("\n".join(code_lines))
                p.paragraph_format.left_indent = Inches(0.4)
                for run in p.runs:
                    run.font.name = "Consolas"
                    run.font.size = Pt(8.5)
                    run.font.color.rgb = RGBColor(40, 40, 40)
                code_lines = []
            else:
                in_code_block = True
                code_lines = []
            continue

        if in_code_block:
            code_lines.append(line)
            continue

        # Blank line
        if not stripped:
            continue

        # Horizontal rule
        if stripped.startswith("---") or stripped.startswith("***"):
            continue

        # Headings
        if stripped.startswith("# "):
            h = doc.add_heading(stripped[2:], level=1)
            h.paragraph_format.space_before = Pt(14)
            h.paragraph_format.space_after = Pt(6)
        elif stripped.startswith("## "):
            h = doc.add_heading(stripped[3:], level=2)
            h.paragraph_format.space_before = Pt(10)
            h.paragraph_format.space_after = Pt(4)
        elif stripped.startswith("### "):
            h = doc.add_heading(stripped[4:], level=3)
            h.paragraph_format.space_before = Pt(8)
            h.paragraph_format.space_after = Pt(2)
        elif stripped.startswith("#### "):
            h = doc.add_heading(stripped[5:], level=4)
        elif stripped.startswith("- ") or stripped.startswith("* "):
            # Bullet point
            clean_text = stripped[2:]
            clean_text = re.sub(r"\*\*(.*?)\*\*", r"\1", clean_text)
            clean_text = re.sub(r"\*(.*?)\*", r"\1", clean_text)
            clean_text = re.sub(r"`(.*?)`", r"\1", clean_text)
            clean_text = re.sub(r"\$(.*?)\$", r"\1", clean_text)
            doc.add_paragraph(clean_text, style="List Bullet")
        elif re.match(r"^\d+\.\s+", stripped):
            # Numbered list
            clean_text = re.sub(r"^\d+\.\s+", "", stripped)
            clean_text = re.sub(r"\*\*(.*?)\*\*", r"\1", clean_text)
            clean_text = re.sub(r"\*(.*?)\*", r"\1", clean_text)
            clean_text = re.sub(r"`(.*?)`", r"\1", clean_text)
            clean_text = re.sub(r"\$(.*?)\$", r"\1", clean_text)
            doc.add_paragraph(clean_text, style="List Number")
        elif stripped.startswith("|") and stripped.endswith("|"):
            # Table row (simplified text representation)
            if re.match(r"^\|[\s\-:]+\|$", stripped):
                continue
            cells = [c.strip() for c in stripped.split("|")[1:-1]]
            clean_row = " | ".join(cells)
            clean_row = re.sub(r"\*\*(.*?)\*\*", r"\1", clean_row)
            p = doc.add_paragraph(clean_row)
            p.paragraph_format.left_indent = Inches(0.2)
            for run in p.runs:
                run.font.size = Pt(9)
        else:
            # Normal paragraph
            clean_text = stripped
            clean_text = re.sub(r"\*\*(.*?)\*\*", r"\1", clean_text)
            clean_text = re.sub(r"\*(.*?)\*", r"\1", clean_text)
            clean_text = re.sub(r"`(.*?)`", r"\1", clean_text)
            clean_text = re.sub(r"\$(.*?)\$", r"\1", clean_text)
            p = doc.add_paragraph(clean_text)
            p.paragraph_format.space_after = Pt(4)

def build_docx(target_filename, source_files, doc_title):
    doc = Document()
    
    # Title Page
    title_p = doc.add_paragraph()
    title_p.paragraph_format.space_before = Pt(72)
    title_p.paragraph_format.space_after = Pt(18)
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_title = title_p.add_run(doc_title)
    run_title.font.size = Pt(24)
    run_title.font.bold = True
    run_title.font.color.rgb = RGBColor(15, 23, 42)

    sub_p = doc.add_paragraph()
    sub_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub_p.paragraph_format.space_after = Pt(144)
    run_sub = sub_p.add_run("Agentic Predictive Maintenance Framework\nAn Explainable Multivariate Time-Series Framework for Early Machine Failure Detection and Maintenance Decision Support")
    run_sub.font.size = Pt(12)
    run_sub.font.color.rgb = RGBColor(71, 85, 105)

    doc.add_page_break()

    for sf in source_files:
        if os.path.exists(sf):
            with open(sf, "r", encoding="utf-8") as f:
                content = f.read()
            parse_markdown_to_docx(doc, content)
            doc.add_page_break()

    out_path = os.path.join(DOC_DIR, target_filename)
    doc.save(out_path)
    print(f"Generated DOCX: {out_path} ({os.path.getsize(out_path)} bytes)")

def main():
    # 1. Complete_Academic_Project_Report.docx
    academic_files = [
        os.path.join(DOC_DIR, "05_Research_Documentation", "Complete_Academic_Report.md"),
        os.path.join(DOC_DIR, "05_Research_Documentation", "Literature_Review.md"),
        os.path.join(DOC_DIR, "05_Research_Documentation", "Experimental_Results.md"),
        os.path.join(DOC_DIR, "05_Research_Documentation", "References.md"),
        os.path.join(DOC_DIR, "05_Research_Documentation", "Limitations_and_Future_Work.md"),
    ]
    build_docx("Complete_Academic_Project_Report.docx", academic_files, "Academic Project Report & Dissertation")

    # 2. Complete_Technical_Implementation_Manual.docx
    technical_files = [
        os.path.join(DOC_DIR, "01_Project_Overview", "Project_Overview.md"),
        os.path.join(DOC_DIR, "01_Project_Overview", "System_Requirements.md"),
        os.path.join(DOC_DIR, "01_Project_Overview", "Technology_Stack.md"),
        os.path.join(DOC_DIR, "01_Project_Overview", "Overall_Architecture.md"),
        os.path.join(DOC_DIR, "04_Technical_Reference", "Mathematical_Foundations.md"),
        os.path.join(DOC_DIR, "04_Technical_Reference", "Algorithm_Reference.md"),
        os.path.join(DOC_DIR, "04_Technical_Reference", "Source_Code_Reference.md"),
        os.path.join(DOC_DIR, "04_Technical_Reference", "API_Reference.md"),
        os.path.join(DOC_DIR, "04_Technical_Reference", "Database_Reference.md"),
        os.path.join(DOC_DIR, "04_Technical_Reference", "Project_Glossary.md"),
        os.path.join(DOC_DIR, "06_Architecture_Diagrams", "All_Architecture_Diagrams.md"),
    ]
    build_docx("Complete_Technical_Implementation_Manual.docx", technical_files, "Complete Technical Implementation Manual")

    # 3. End_User_Instruction_Manual.docx
    user_files = [
        os.path.join(DOC_DIR, "03_User_Instruction_Manual", "Installation_Guide.md"),
        os.path.join(DOC_DIR, "03_User_Instruction_Manual", "Configuration_Guide.md"),
        os.path.join(DOC_DIR, "03_User_Instruction_Manual", "Application_User_Manual.md"),
        os.path.join(DOC_DIR, "03_User_Instruction_Manual", "Troubleshooting_Guide.md"),
    ]
    build_docx("End_User_Instruction_Manual.docx", user_files, "End-User Instruction & Operations Manual")

    # 4. Project_Phase_Wise_Reports.docx
    phase_files = [
        os.path.join(DOC_DIR, "02_Phase_Wise_Technical_Reports", f"Phase_{i:02d}_{suffix}.md")
        for i, suffix in [
            (1, "Data_Preprocessing"),
            (2, "Machine_Learning"),
            (3, "Deep_Learning"),
            (4, "Anomaly_Detection"),
            (5, "Explainable_AI"),
            (6, "Agentic_AI_LangGraph"),
            (7, "Maintenance_Recommendation"),
            (8, "Dashboard_Integration"),
            (9, "Testing_Validation"),
        ]
    ]
    build_docx("Project_Phase_Wise_Reports.docx", phase_files, "Comprehensive Phase-Wise Implementation Reports (Phases 1-9)")

if __name__ == "__main__":
    main()
