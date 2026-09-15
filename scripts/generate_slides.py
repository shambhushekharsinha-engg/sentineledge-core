"""
generate_slides.py - Creates a beautiful PDF slide deck for SentinelEdge
Uses ReportLab for professional PDF generation with custom styling.
"""

from reportlab.lib.pagesizes import landscape, A4
from reportlab.lib import colors
from reportlab.lib.units import inch, cm
from reportlab.platypus import SimpleDocTemplate, Spacer, Table, TableStyle, Paragraph, Image as RLImage
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.pdfgen import canvas
from reportlab.platypus import PageBreak
import os

# ── Constants ─────────────────────────────────────────────────────────────────
W, H = landscape(A4)
COVER_IMG = r"C:\Users\shamb\.gemini\antigravity\brain\a82891e3-a3d0-406b-b1ed-ebd627ad619b\sentineledge_cover_1789485574945.jpg"
OUT_PATH  = r"C:\sentineledge-core\SentinelEdge_Presentation.pdf"

# ── Colour Palette ─────────────────────────────────────────────────────────────
NAVY       = colors.HexColor("#0A0E1A")
BLUE       = colors.HexColor("#0068B5")
ELEC_BLUE  = colors.HexColor("#00B4FF")
ACCENT     = colors.HexColor("#4ADE80")
WHITE      = colors.white
LIGHT_GRAY = colors.HexColor("#CBD5E1")
DARK_GRAY  = colors.HexColor("#1E293B")
CARD_BG    = colors.HexColor("#111827")

# ── Slide Builder ─────────────────────────────────────────────────────────────
class SlideDeck:
    def __init__(self, path):
        self.c = canvas.Canvas(path, pagesize=landscape(A4))
        self.c.setTitle("SentinelEdge – Intel Physical AI Challenge")
        self.c.setAuthor("SentinelEdge Team")
        self.c.setSubject("Bimanual VLA Manipulation on Intel Core Ultra")

    def save(self):
        self.c.save()
        print(f"Saved: {OUT_PATH}")

    # ── drawing primitives ────────────────────────────────────────────────────
    def bg(self, col=NAVY):
        self.c.setFillColor(col)
        self.c.rect(0, 0, W, H, fill=1, stroke=0)

    def card(self, x, y, w, h, col=DARK_GRAY, radius=8):
        self.c.setFillColor(col)
        self.c.roundRect(x, y, w, h, radius, fill=1, stroke=0)

    def hline(self, y, col=ELEC_BLUE, lw=1.5):
        self.c.setStrokeColor(col)
        self.c.setLineWidth(lw)
        self.c.line(0.5*inch, y, W - 0.5*inch, y)

    def txt(self, text, x, y, size=12, col=WHITE, bold=False, align="left"):
        font = "Helvetica-Bold" if bold else "Helvetica"
        self.c.setFont(font, size)
        self.c.setFillColor(col)
        if align == "center":
            self.c.drawCentredString(x, y, text)
        elif align == "right":
            self.c.drawRightString(x, y, text)
        else:
            self.c.drawString(x, y, text)

    def badge(self, x, y, text, bg=BLUE, fg=WHITE, size=8):
        tw = self.c.stringWidth(text, "Helvetica-Bold", size) + 10
        self.c.setFillColor(bg)
        self.c.roundRect(x, y - 3, tw, 14, 4, fill=1, stroke=0)
        self.c.setFont("Helvetica-Bold", size)
        self.c.setFillColor(fg)
        self.c.drawString(x + 5, y + 1, text)
        return tw + 4

    def slide_num(self, n, total=9):
        self.txt(f"{n} / {total}", W - 0.6*inch, 0.25*inch,
                 size=8, col=LIGHT_GRAY, align="right")

    def footer(self, text="Intel Physical AI Challenge  |  lablab.ai 2026"):
        self.c.setFillColor(colors.HexColor("#1E293B"))
        self.c.rect(0, 0, W, 0.45*inch, fill=1, stroke=0)
        self.txt(text, W/2, 0.15*inch, size=8, col=LIGHT_GRAY, align="center")

    def next_slide(self):
        self.c.showPage()

    # ── bullet helper ─────────────────────────────────────────────────────────
    def bullets(self, items, x, y, size=11, gap=18, col=WHITE, dot_col=ELEC_BLUE):
        for item in items:
            self.c.setFillColor(dot_col)
            self.c.circle(x + 4, y + 3, 3, fill=1, stroke=0)
            self.c.setFont("Helvetica", size)
            self.c.setFillColor(col)
            self.c.drawString(x + 12, y, item)
            y -= gap
        return y

    # ═══════════════════════════════════════════════════════════════════════════
    # SLIDE 1 – Cover
    # ═══════════════════════════════════════════════════════════════════════════
    def slide_cover(self):
        if os.path.exists(COVER_IMG):
            self.c.drawImage(COVER_IMG, 0, 0, W, H, preserveAspectRatio=False)
        else:
            self.bg()
        # dark overlay for text legibility
        self.c.setFillColor(colors.HexColor("#00000066"))
        self.c.rect(0, 0, W, H, fill=1, stroke=0)

        self.txt("SentinelEdge", W/2, H - 1.4*inch, size=52, bold=True, col=WHITE, align="center")
        self.txt("Bimanual VLA Manipulation with Multi-Modal Reasoning", W/2, H - 2.1*inch,
                 size=18, col=ELEC_BLUE, align="center")
        self.hline(H - 2.4*inch, lw=2)

        # badges row
        badges = ["Intel Core Ultra NPU", "OpenVINO INT8", "MuJoCo Physics",
                  "Speechmatics ASR", "39 Tests Passing", "GitHub Actions CI"]
        bw_total = sum(self.c.stringWidth(b, "Helvetica-Bold", 9) + 18 for b in badges) + 10*(len(badges)-1)
        bx = (W - bw_total) / 2
        for b in badges:
            bw = self.badge(bx, H - 2.9*inch, b, size=9)
            bx += bw + 6

        self.txt("Intel Physical AI Challenge  |  lablab.ai 2026",
                 W/2, 0.9*inch, size=11, col=LIGHT_GRAY, align="center")
        self.txt("github.com/shambhushekharsinha-engg/sentineledge-core",
                 W/2, 0.55*inch, size=10, col=ELEC_BLUE, align="center")
        self.next_slide()

    # ═══════════════════════════════════════════════════════════════════════════
    # SLIDE 2 – The Problem
    # ═══════════════════════════════════════════════════════════════════════════
    def slide_problem(self):
        self.bg()
        self.txt("THE PROBLEM", 0.55*inch, H - 0.8*inch, size=9, col=ELEC_BLUE, bold=True)
        self.txt("3 Hard Bottlenecks in Physical AI", 0.5*inch, H - 1.2*inch, size=28, bold=True)
        self.hline(H - 1.4*inch)

        problems = [
            ("01", "Multi-Modal Grounding",
             "VLA Transformers that interpret language + vision are too large for edge hardware."),
            ("02", "Bimanual Coordination",
             "Synchronizing two arms for hand-offs, drawer ops, and collision avoidance is exponentially complex."),
            ("03", "Edge Inference Latency",
             "Heavy VLA policies need cloud GPUs — making real-time 30Hz closed-loop control impossible."),
        ]
        card_w = (W - 1.5*inch) / 3 - 0.15*inch
        cx = 0.5*inch
        for num, title, desc in problems:
            self.card(cx, 1.0*inch, card_w, H - 2.8*inch, col=DARK_GRAY)
            self.c.setFillColor(BLUE)
            self.c.circle(cx + 0.45*inch, H - 2.0*inch, 18, fill=1, stroke=0)
            self.txt(num, cx + 0.45*inch, H - 2.07*inch, size=13, bold=True, align="center")
            self.txt(title, cx + 0.2*inch, H - 2.65*inch, size=13, bold=True, col=ELEC_BLUE)
            # word-wrap desc
            words = desc.split()
            line, lines = "", []
            for w in words:
                if self.c.stringWidth(line + w, "Helvetica", 9.5) < card_w - 0.3*inch:
                    line += w + " "
                else:
                    lines.append(line.strip()); line = w + " "
            lines.append(line.strip())
            dy = H - 2.95*inch
            for l in lines:
                self.txt(l, cx + 0.15*inch, dy, size=9.5, col=LIGHT_GRAY)
                dy -= 14
            cx += card_w + 0.15*inch

        self.footer(); self.slide_num(2); self.next_slide()

    # ═══════════════════════════════════════════════════════════════════════════
    # SLIDE 3 – Our Solution
    # ═══════════════════════════════════════════════════════════════════════════
    def slide_solution(self):
        self.bg()
        self.txt("OUR SOLUTION", 0.55*inch, H - 0.8*inch, size=9, col=ACCENT, bold=True)
        self.txt("End-to-End Perception-to-Action Pipeline", 0.5*inch, H - 1.2*inch, size=28, bold=True)
        self.hline(H - 1.4*inch)

        pts = [
            ("Dual SO-101 Arms", "MuJoCo bimanual coordination: set table, open drawer, hand-off, dodge obstacles"),
            ("Temporal VLA Transformer", "CNN vision + Sentence-Transformer language + action history fused via 4-layer Cross-Modal Transformer"),
            ("Intel OpenVINO INT8", "NNCF Post-Training Quantization → 85 FPS on Core Ultra NPU (2.8x real-time headroom)"),
            ("Speechmatics Voice-to-Action", "Live mic → Batch ASR → semantic embedding → robot execution → TTS voice confirmation"),
            ("6-Axis Domain Randomization", "Mass, friction, placement, lighting, shape scale, floor texture vary across 10 seeds"),
            ("Multi-Stage FSM Reward", "INIT → APPROACHING → GRASPING → LIFTING → PLACING → DONE (shaped rewards)"),
        ]
        cy = H - 1.8*inch
        for i, (title, desc) in enumerate(pts):
            col = BLUE if i % 2 == 0 else colors.HexColor("#1D4ED8")
            self.card(0.5*inch, cy - 0.42*inch, W - 1.0*inch, 0.47*inch, col=col, radius=6)
            self.txt(f"  {title}", 0.65*inch, cy - 0.08*inch, size=11, bold=True, col=WHITE)
            self.txt(desc, 0.65*inch, cy - 0.28*inch, size=9, col=LIGHT_GRAY)
            cy -= 0.55*inch

        self.footer(); self.slide_num(3); self.next_slide()

    # ═══════════════════════════════════════════════════════════════════════════
    # SLIDE 4 – Architecture
    # ═══════════════════════════════════════════════════════════════════════════
    def slide_architecture(self):
        self.bg()
        self.txt("SYSTEM ARCHITECTURE", 0.55*inch, H - 0.8*inch, size=9, col=ELEC_BLUE, bold=True)
        self.txt("Voice-to-Action Pipeline", 0.5*inch, H - 1.2*inch, size=28, bold=True)
        self.hline(H - 1.4*inch)

        # Pipeline boxes
        pipeline = [
            ("Voice Command\n/ Text", BLUE),
            ("Speechmatics\nASR", colors.HexColor("#1ABC9C")),
            ("Sentence\nTransformers", colors.HexColor("#8B5CF6")),
            ("Cross-Modal\nTransformer", colors.HexColor("#F59E0B")),
            ("Bimanual\nAction Heads", colors.HexColor("#EF4444")),
            ("EMA Filter\n+ MuJoCo", colors.HexColor("#10B981")),
        ]

        bw = (W - 1.2*inch) / len(pipeline) - 0.1*inch
        bx = 0.55*inch
        by = H / 2 - 0.5*inch
        bh = 1.1*inch
        for i, (label, col) in enumerate(pipeline):
            self.card(bx, by, bw, bh, col=col, radius=10)
            lines = label.split("\n")
            for j, line in enumerate(lines):
                self.txt(line, bx + bw/2, by + bh/2 + 8 - j*16,
                         size=10, bold=True, col=WHITE, align="center")
            if i < len(pipeline) - 1:
                ax = bx + bw + 2
                self.c.setFillColor(ELEC_BLUE)
                self.c.setStrokeColor(ELEC_BLUE)
                self.c.setLineWidth(1.5)
                self.c.line(ax, by + bh/2, ax + 0.08*inch, by + bh/2)
                p = self.c.beginPath()
                p.moveTo(ax + 0.08*inch, by + bh/2 + 4)
                p.lineTo(ax + 0.08*inch, by + bh/2 - 4)
                p.lineTo(ax + 0.13*inch, by + bh/2)
                p.close()
                self.c.drawPath(p, fill=1, stroke=0)
            bx += bw + 0.12*inch

        # Bottom info cards
        info = [
            ("Vision Encoder", "4-stage CNN + AdaptiveAvgPool", BLUE),
            ("History Encoder", "64-d action buffer, 4-step memory", colors.HexColor("#7C3AED")),
            ("OpenVINO NPU", "INT8 PTQ via NNCF → 85 FPS", colors.HexColor("#DC2626")),
            ("ROS 2 Node", "Production robot deployment ready", colors.HexColor("#065F46")),
        ]
        cw = (W - 1.2*inch) / 4 - 0.1*inch
        cx = 0.55*inch
        for title, desc, col in info:
            self.card(cx, 0.65*inch, cw, 1.1*inch, col=col, radius=6)
            self.txt(title, cx + cw/2, 1.55*inch, size=10, bold=True, col=WHITE, align="center")
            self.txt(desc, cx + cw/2, 1.25*inch, size=8, col=LIGHT_GRAY, align="center")
            cx += cw + 0.12*inch

        self.footer(); self.slide_num(4); self.next_slide()

    # ═══════════════════════════════════════════════════════════════════════════
    # SLIDE 5 – Task Suite
    # ═══════════════════════════════════════════════════════════════════════════
    def slide_tasks(self):
        self.bg()
        self.txt("TASK SUITE", 0.55*inch, H - 0.8*inch, size=9, col=ACCENT, bold=True)
        self.txt("4 Bimanual Task Variations", 0.5*inch, H - 1.2*inch, size=28, bold=True)
        self.hline(H - 1.4*inch)

        tasks = [
            ("Set Table", "Place plate and cup in correct positions", "Bimanual coordination, IK solving", "#3B82F6"),
            ("Retrieve Spoon", "Open drawer, retrieve spoon, place beside plate", "Drawer manipulation + sequencing", "#8B5CF6"),
            ("Cross-Arm Hand-off", "Pick cup with Arm A, pass to Arm B, place on table", "Inter-arm object transfer", "#EF4444"),
            ("Collision Avoidance", "Place fork while avoiding sinusoidal obstacle", "Dynamic obstacle reasoning", "#F59E0B"),
        ]
        tw = (W - 1.2*inch) / 2 - 0.1*inch
        positions = [(0.5*inch, H/2), (0.5*inch + tw + 0.12*inch, H/2),
                     (0.5*inch, 0.65*inch), (0.5*inch + tw + 0.12*inch, 0.65*inch)]
        th = H/2 - 0.85*inch

        for i, (title, desc, challenge, col) in enumerate(tasks):
            tx, ty = positions[i]
            self.card(tx, ty, tw, th, col=DARK_GRAY, radius=8)
            self.c.setFillColor(colors.HexColor(col))
            self.c.roundRect(tx, ty + th - 0.35*inch, tw, 0.35*inch, 8, fill=1, stroke=0)
            self.txt(title, tx + tw/2, ty + th - 0.22*inch, size=13, bold=True, col=WHITE, align="center")
            self.txt(desc, tx + 0.12*inch, ty + th - 0.65*inch, size=9.5, col=WHITE)
            self.txt("Challenge:", tx + 0.12*inch, ty + th - 0.9*inch, size=8.5, col=ELEC_BLUE, bold=True)
            self.txt(challenge, tx + 0.12*inch, ty + th - 1.1*inch, size=8.5, col=LIGHT_GRAY)

        self.footer(); self.slide_num(5); self.next_slide()

    # ═══════════════════════════════════════════════════════════════════════════
    # SLIDE 6 – Intel Optimization
    # ═══════════════════════════════════════════════════════════════════════════
    def slide_intel(self):
        self.bg()
        self.txt("INTEL OPTIMIZATION", 0.55*inch, H - 0.8*inch, size=9, col=BLUE, bold=True)
        self.txt("OpenVINO INT8 on Intel Core Ultra NPU", 0.5*inch, H - 1.2*inch, size=28, bold=True)
        self.hline(H - 1.4*inch)

        # Big metric
        self.card(0.5*inch, H/2 - 0.2*inch, W/2 - 0.65*inch, H/2 - 1.1*inch, col=DARK_GRAY)
        self.txt("85.34 FPS", W/4 - 0.1*inch, H - 2.3*inch, size=52, bold=True, col=ELEC_BLUE, align="center")
        self.txt("Intel Core Ultra NPU  |  INT8 PTQ via NNCF", W/4 - 0.1*inch,
                 H - 2.85*inch, size=11, col=LIGHT_GRAY, align="center")
        self.txt("11.72 ms avg latency  |  2.8x real-time headroom",
                 W/4 - 0.1*inch, H - 3.15*inch, size=10, col=ACCENT, align="center")

        # Right side steps
        steps = [
            ("Step 1", "torch.jit.script() - Stable model export"),
            ("Step 2", "ov.convert_model() - OpenVINO IR format"),
            ("Step 3", "nncf.quantize() - INT8 Post-Training Quantization"),
            ("Step 4", "NPU auto-routing via PERFORMANCE_HINT=THROUGHPUT"),
            ("Step 5", "100-iteration benchmark + p50/p95/p99 profiling"),
        ]
        rx = W/2 + 0.2*inch
        ry = H - 1.8*inch
        self.card(rx - 0.15*inch, 0.6*inch, W/2 - 0.55*inch, H - 2.25*inch, col=DARK_GRAY)
        for num, text in steps:
            self.badge(rx, ry + 2, num, bg=BLUE, size=8)
            self.txt(text, rx + 0.75*inch, ry, size=9.5, col=WHITE)
            ry -= 0.55*inch

        self.footer(); self.slide_num(6); self.next_slide()

    # ═══════════════════════════════════════════════════════════════════════════
    # SLIDE 7 – Technical Quality
    # ═══════════════════════════════════════════════════════════════════════════
    def slide_quality(self):
        self.bg()
        self.txt("TECHNICAL QUALITY", 0.55*inch, H - 0.8*inch, size=9, col=ACCENT, bold=True)
        self.txt("Production-Grade Engineering Standards", 0.5*inch, H - 1.2*inch, size=28, bold=True)
        self.hline(H - 1.4*inch)

        metrics = [
            ("39", "pytest Tests", "4 files: env, policy, reward FSM, metrics", BLUE),
            ("6-Axis", "Randomization", "Mass, friction, pose, lighting, shape, bg", colors.HexColor("#7C3AED")),
            ("10", "Seeds Demo", "Full stitched video + GIF grid in README", colors.HexColor("#D97706")),
            ("INT8", "Quantization", "NNCF PTQ · 4x size reduction · NPU routing", colors.HexColor("#DC2626")),
        ]
        mw = (W - 1.2*inch) / 4 - 0.1*inch
        mx = 0.5*inch
        for val, label, desc, col in metrics:
            self.card(mx, H/2 - 0.1*inch, mw, H/2 - 1.0*inch, col=col, radius=10)
            self.txt(val, mx + mw/2, H - 2.2*inch, size=36, bold=True, col=WHITE, align="center")
            self.txt(label, mx + mw/2, H - 2.75*inch, size=11, bold=True, col=WHITE, align="center")
            self.txt(desc, mx + mw/2, H - 3.1*inch, size=8, col=LIGHT_GRAY, align="center")
            mx += mw + 0.12*inch

        items = [
            "Conda + pip + Docker + Makefile — 4 install paths",
            "GitHub Actions CI with full pytest suite on every push",
            "HuggingFace Model Card (MODEL_CARD.md) with BibTeX",
            "Ablation Study Jupyter Notebook (randomization, precision, device, history)",
            "EMA kinematic smoother + Foxglove telemetry dashboard for Sim2Real",
            "ROS 2 deployment node for production physical robot",
        ]
        self.bullets(items, 0.55*inch, H/2 - 0.55*inch, size=10, gap=16)

        self.footer(); self.slide_num(7); self.next_slide()

    # ═══════════════════════════════════════════════════════════════════════════
    # SLIDE 8 – Scoring Rubric
    # ═══════════════════════════════════════════════════════════════════════════
    def slide_rubric(self):
        self.bg()
        self.txt("SCORING RUBRIC", 0.55*inch, H - 0.8*inch, size=9, col=ELEC_BLUE, bold=True)
        self.txt("Full Coverage Across All Criteria", 0.5*inch, H - 1.2*inch, size=28, bold=True)
        self.hline(H - 1.4*inch)

        rows = [
            ("Category", "Max Pts", "Coverage", True),
            ("Bimanual Task Completion", "25", "4 tasks: set table, drawer, hand-off, collision avoidance", False),
            ("Robustness & Generalization", "15", "6-axis domain randomization × 10 seeds", False),
            ("VLA / Multi-Modal Reasoning", "20", "Real embeddings + Temporal ACT + Cross-Modal Transformer", False),
            ("OpenVINO & Intel Core Ultra", "20", "INT8 PTQ + NPU auto-routing + p50/p95/p99 profiling", False),
            ("Technical Quality", "10", "CI + Docker + Conda + Makefile + 39 tests + YAML config", False),
            ("Innovation & Demo", "5", "Voice-to-Action + TTS + EMA + Foxglove + ablation notebook", False),
            ("Speechmatics BONUS", "+Bonus", "Full Batch ASR API + microphone Gradio UI", False),
        ]

        rw = [2.8*inch, 0.7*inch, W - 4.3*inch]
        rx = 0.5*inch
        ry = H - 1.75*inch
        rh = 0.38*inch

        for row in rows:
            cat, pts, cov, header = row
            if header:
                self.card(rx, ry - rh + 5, sum(rw) + 0.1*inch, rh, col=BLUE, radius=4)
                fcol = WHITE
            else:
                bg = DARK_GRAY if rows.index(row) % 2 == 0 else CARD_BG
                self.card(rx, ry - rh + 5, sum(rw) + 0.1*inch, rh, col=bg, radius=0)
                fcol = LIGHT_GRAY
            self.txt(cat, rx + 0.1*inch, ry - 0.1*inch, size=10 if header else 9.5,
                     bold=header, col=WHITE if header else fcol)
            self.c.setFillColor(ACCENT if "Bonus" in pts else (WHITE if header else ELEC_BLUE))
            self.c.setFont("Helvetica-Bold" if header else "Helvetica-Bold", 11 if "25" in pts else 9)
            self.c.drawCentredString(rx + rw[0] + rw[1]/2, ry - 0.1*inch, pts)
            self.txt(cov, rx + rw[0] + rw[1] + 0.1*inch, ry - 0.1*inch,
                     size=9 if not header else 10, bold=header, col=WHITE if header else fcol)
            ry -= rh

        self.footer(); self.slide_num(8); self.next_slide()

    # ═══════════════════════════════════════════════════════════════════════════
    # SLIDE 9 – Call to Action / Links
    # ═══════════════════════════════════════════════════════════════════════════
    def slide_cta(self):
        self.bg()
        # gradient overlay
        self.c.setFillColor(colors.HexColor("#050D1A"))
        self.c.rect(0, H/2, W, H/2, fill=1, stroke=0)

        self.txt("READY TO SUBMIT", W/2, H - 1.0*inch, size=11, col=ELEC_BLUE, bold=True, align="center")
        self.txt("SentinelEdge", W/2, H - 1.7*inch, size=52, bold=True, col=WHITE, align="center")
        self.txt("The Physical AI System That Speaks, Sees, Thinks, and Acts.", W/2,
                 H - 2.3*inch, size=16, col=LIGHT_GRAY, align="center")
        self.hline(H - 2.6*inch)

        links = [
            ("GitHub Repository", "github.com/shambhushekharsinha-engg/sentineledge-core", BLUE),
            ("Demo Video", "youtube.com/watch?v=ugT-6m7i8ls", colors.HexColor("#EF4444")),
            ("lablab.ai Portal", "lablab.ai/event/intel-physical-ai-challenge", colors.HexColor("#10B981")),
        ]
        lw = (W - 1.5*inch) / 3 - 0.15*inch
        lx = 0.6*inch
        for label, url, col in links:
            self.card(lx, H/2 - 0.5*inch, lw, 1.3*inch, col=col, radius=10)
            self.txt(label, lx + lw/2, H/2 + 0.62*inch, size=12, bold=True, col=WHITE, align="center")
            self.c.setFont("Helvetica", 8)
            self.c.setFillColor(LIGHT_GRAY)
            self.c.drawCentredString(lx + lw/2, H/2 + 0.38*inch, url)
            lx += lw + 0.15*inch

        stats = [
            ("39", "pytest Tests"), ("10", "Demo Seeds"), ("6", "Domain Axes"),
            ("85", "FPS on NPU"), ("4", "Task Types"), ("100%", "Committed"),
        ]
        sw = (W - 1.2*inch) / 6 - 0.08*inch
        sx = 0.5*inch
        for val, lbl in stats:
            self.card(sx, 0.65*inch, sw, 1.2*inch, col=DARK_GRAY, radius=8)
            self.txt(val, sx + sw/2, 1.55*inch, size=22, bold=True, col=ELEC_BLUE, align="center")
            self.txt(lbl, sx + sw/2, 1.15*inch, size=8, col=LIGHT_GRAY, align="center")
            sx += sw + 0.1*inch

        self.footer("Intel Physical AI Challenge  |  Built with OpenVINO + MuJoCo + Speechmatics  |  lablab.ai 2026")
        self.slide_num(9)
        self.next_slide()


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    deck = SlideDeck(OUT_PATH)
    deck.slide_cover()
    deck.slide_problem()
    deck.slide_solution()
    deck.slide_architecture()
    deck.slide_tasks()
    deck.slide_intel()
    deck.slide_quality()
    deck.slide_rubric()
    deck.slide_cta()
    deck.save()
    print(f"\nPresentation saved: {OUT_PATH}")
    print("9 slides | Landscape A4 | Intel Physical AI Challenge")
