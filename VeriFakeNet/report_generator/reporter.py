import re
from fpdf import FPDF
import os

class ReportGenerator:
    def __init__(self, output_dir="outputs/reports"):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

    def _clean_text(self, text):
        if not isinstance(text, str):
            text = str(text)
        # Map common non-latin1 characters to ASCII equivalents
        text = (text
            .replace('\u2014', '-').replace('\u2013', '-')
            .replace('\u2018', "'").replace('\u2019', "'")
            .replace('\u201c', '"').replace('\u201d', '"')
            .replace('\u2022', '-').replace('\u2192', '->')
            .replace('\u00d7', 'x').replace('\u2713', '[OK]')
            .replace('\u2714', '[OK]').replace('\u2718', '[X]')
        )
        # Strip remaining non-latin1 characters
        return text.encode('latin-1', 'replace').decode('latin-1')

    def _strip_html(self, text):
        """Remove HTML tags and decode common HTML entities."""
        if not isinstance(text, str):
            text = str(text)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = (text
            .replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
            .replace('&nbsp;', ' ').replace('&#39;', "'").replace('&quot;', '"')
        )
        # Collapse whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return self._clean_text(text)

    def _section_header(self, pdf, number, title):
        pdf.set_font("Arial", 'B', 13)
        pdf.set_fill_color(20, 30, 55)
        pdf.set_text_color(200, 220, 255)
        pdf.cell(0, 10, txt=self._clean_text(f"{number}. {title}"), ln=True, fill=True)
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Arial", size=11)
        pdf.ln(2)

    def _row(self, pdf, label, value, bold_label=True):
        """Render a key-value row, handling long values with word wrap."""
        label_w = 52  # mm
        # Remaining width = page width minus both margins minus label column
        value_w = pdf.w - pdf.l_margin - pdf.r_margin - label_w
        x_start = pdf.get_x()
        y_start = pdf.get_y()

        pdf.set_font("Arial", 'B' if bold_label else '', 11)
        pdf.cell(label_w, 8, txt=self._clean_text(label), border=0, ln=0)
        pdf.set_font("Arial", size=11)
        pdf.multi_cell(value_w, 8, txt=self._clean_text(str(value)), border=0)
        # Ensure we don't double-advance if multi_cell already moved to next line


    def generate_report(self, filename, data):
        """
        Generates a comprehensive PDF forensic report.

        Args:
            filename (str): Output PDF filename.
            data (dict): All analysis results from the pipeline.
        """
        pdf = FPDF()
        pdf.add_page()

        # ── Header ─────────────────────────────────────────────────────────────
        pdf.set_fill_color(10, 20, 45)
        pdf.rect(0, 0, 210, 28, 'F')
        pdf.set_font("Arial", 'B', 18)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(0, 14, txt="VeriFakeNet", ln=False, align='C')
        pdf.ln(10)
        pdf.set_font("Arial", size=10)
        pdf.set_text_color(180, 200, 240)
        pdf.cell(0, 8, txt="AI-Powered Media Forensics & Deepfake Detection Report", ln=True, align='C')
        pdf.set_text_color(0, 0, 0)
        pdf.ln(6)

        # ── 1. Executive Summary ───────────────────────────────────────────────
        self._section_header(pdf, 1, "Executive Summary")
        pred           = data.get('prediction', 'Unknown')
        conf           = float(data.get('confidence', 0.0))
        trust          = data.get('trust_score', 0)
        interpretation = data.get('interpretation', '')
        filename_orig  = data.get('filename', 'N/A')
        ai_verdict     = data.get('ai_explanation_verdict', '')
        ai_source      = data.get('ai_explanation_source', 'rule_based')

        self._row(pdf, "File:",          filename_orig)
        self._row(pdf, "Prediction:",    f"{pred}  ({conf:.1f}% model confidence)")
        self._row(pdf, "Trust Score:",   f"{trust}/100 — {interpretation}")
        self._row(pdf, "AI Verdict:",    ai_verdict or f"{pred} with {conf:.1f}% confidence")
        self._row(pdf, "Explanation by:",
                  "Groq LLaMA-3.3-70B" if ai_source == 'groq' else "Rule-Based Analysis")
        pdf.ln(4)

        # ── 2. Trust Score Breakdown ───────────────────────────────────────────
        self._section_header(pdf, 2, "Trust Score Breakdown")
        breakdown = data.get('breakdown', {})
        score_items = [
            ("Deepfake Model (40%)", breakdown.get('deepfake_score', 0)),
            ("Metadata Forensics (20%)", breakdown.get('metadata_score', 0)),
            ("ELA Analysis (20%)", breakdown.get('ela_score', 0)),
            ("Hash Integrity (20%)", breakdown.get('hash_score', 0)),
        ]
        for label, val in score_items:
            bar_width = int(val * 1.4)  # scale to ~140 max px
            pdf.set_font("Arial", size=10)
            pdf.cell(70, 7, txt=self._clean_text(label))
            pdf.set_font("Arial", 'B', 10)
            pdf.cell(20, 7, txt=f"{val:.0f}/100")
            pdf.ln(7)
        pdf.ln(4)

        # ── 3. Face Attributes ─────────────────────────────────────────────────
        self._section_header(pdf, 3, "Face Attributes Analysis")
        attrs = data.get('attributes', {})
        if attrs:
            attr_map = [
                ('gender', 'Gender'),
                ('face_shape', 'Face Shape'),
                ('hair_texture', 'Hair Texture'),
                ('hair_color', 'Hair Color'),
                ('skin_tone', 'Skin Tone'),
            ]
            for key, label in attr_map:
                av = attrs.get(key, {})
                if isinstance(av, dict):
                    p = av.get('prediction', 'N/A')
                    c = av.get('confidence', 0.0)
                    self._row(pdf, f"{label}:", f"{p}  ({float(c):.1f}% confidence)")
        else:
            pdf.cell(0, 8, txt="No face detected for attribute analysis.", ln=True)
        pdf.ln(4)

        # ── 4. Metadata Forensics ──────────────────────────────────────────────
        self._section_header(pdf, 4, "Metadata Forensics")
        meta_flags = data.get('metadata_flags', [])
        if meta_flags:
            for flag in meta_flags:
                pdf.set_font("Arial", size=10)
                pdf.multi_cell(0, 7, txt=self._clean_text(f"  - {flag}"))
        else:
            pdf.cell(0, 8, txt="No metadata flags generated.", ln=True)
        pdf.ln(4)

        # ── 5. Error Level Analysis (ELA) ─────────────────────────────────────
        self._section_header(pdf, 5, "Error Level Analysis (ELA)")
        ela_avg = float(data.get('ela_avg', 0.0))
        ela_max = float(data.get('ela_max', 0.0))
        ela_verdict = (
            "SEVERE anomalies — strong evidence of manipulation"
            if ela_avg > 12 else
            "MODERATE anomalies — possible localized editing"
            if ela_avg > 5 else
            "NORMAL compression — consistent with authentic media"
        )
        self._row(pdf, "Average Error:", f"{ela_avg:.2f}  ({ela_verdict})")
        self._row(pdf, "Maximum Error:", f"{ela_max:.2f}")
        pdf.ln(4)

        # ── 6. Perceptual Hash Verification ───────────────────────────────────
        self._section_header(pdf, 6, "Perceptual Hash Verification")
        hash_msg  = data.get('hash_message', 'N/A')
        hash_score= data.get('hash_score', 0)
        self._row(pdf, "Integrity Score:", f"{hash_score}/100")
        self._row(pdf, "Result:",          hash_msg)
        hash_vals = data.get('hash_values', {})
        if hash_vals and 'error' not in hash_vals:
            for htype, hval in hash_vals.items():
                self._row(pdf, f"{htype}:", str(hval)[:60])
        pdf.ln(4)

        # ── 7. AI Forensic Explanation ────────────────────────────────────────
        self._section_header(pdf, 7, "AI Forensic Explanation")
        ai_source  = data.get('ai_explanation_source', 'rule_based')
        ai_text    = data.get('ai_explanation', '')
        ai_verdict = data.get('ai_explanation_verdict', '')

        # Source badge line
        source_label = "Groq LLaMA-3.3-70B (AI-Generated)" if ai_source == 'groq' else "Rule-Based Forensic Analysis"
        pdf.set_font("Arial", 'I', 9)
        pdf.set_text_color(80, 120, 180)
        pdf.cell(0, 6, txt=f"  Source: {source_label}", ln=True)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(2)

        pdf.set_font("Arial", 'B', 10)
        pdf.cell(0, 7, txt=self._clean_text(f"Verdict: {ai_verdict}"), ln=True)
        pdf.ln(2)

        if ai_text:
            # Strip any HTML tags that may be in the raw text
            clean_explanation = self._strip_html(ai_text)
            pdf.set_font("Arial", size=10)
            pdf.set_fill_color(240, 245, 255)
            pdf.multi_cell(0, 7, txt=clean_explanation, border=1, fill=True)
        else:
            pdf.cell(0, 8, txt="AI explanation not available for this analysis.", ln=True)
        pdf.ln(4)

        # ── Footer ─────────────────────────────────────────────────────────────
        pdf.set_font("Arial", 'I', 8)
        pdf.set_text_color(120, 120, 120)
        pdf.cell(0, 6,
                 txt="VeriFakeNet v1.0 | B.Tech Final Year Project | EfficientNet-B3 + BiLSTM + Groq AI",
                 ln=True, align='C')

        # Save PDF
        output_path = os.path.join(self.output_dir, filename)
        pdf.output(output_path)
        return output_path
