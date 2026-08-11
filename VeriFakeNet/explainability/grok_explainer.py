"""
explainability/grok_explainer.py
=================================
AI explainability for VeriFakeNet using Groq API.
API key (gsk_...) is read from .env / GROK_API_KEY env var.

Groq provides fast inference on open-source models (llama-3.3-70b-versatile etc.)
Falls back to rich rule-based explanation if Groq is unavailable.
"""

import os
import json
import pathlib

# Load .env from project root
try:
    from dotenv import load_dotenv
    _env_path = pathlib.Path(__file__).parent.parent / '.env'
    load_dotenv(dotenv_path=_env_path, override=True)
except ImportError:
    pass


class GrokExplainer:
    """
    Generates Groq-powered forensic explanations for VeriFakeNet analysis results.
    Falls back to rich rule-based explanation if Groq is unavailable.
    """

    def __init__(self, api_key: str = ""):
        # Deliberately read the key lazily on each explain() call
        # so that .env changes are picked up without restarting
        self._forced_key = api_key

    def _get_client(self):
        """Lazy client creation — re-reads env var every call."""
        api_key = self._forced_key or os.environ.get("GROK_API_KEY", "").strip()
        if not api_key:
            return None
        try:
            from groq import Groq
            return Groq(api_key=api_key)
        except ImportError:
            try:
                # Fallback: use openai-compatible client pointed at Groq
                from openai import OpenAI
                return OpenAI(
                    api_key=api_key,
                    base_url="https://api.groq.com/openai/v1"
                )
            except ImportError:
                return None
        except Exception:
            return None

    def explain(self, results: dict, media_type: str = "image") -> dict:
        """
        Generate a forensic AI explanation from analysis results.
        Returns dict with keys: 'text', 'source', 'verdict'
        """
        context = self._build_context(results, media_type)
        client  = self._get_client()

        if client:
            try:
                return self._groq_explain(client, context, media_type)
            except Exception as e:
                print(f"[GrokExplainer] Groq API error: {e} — using rule-based")

        return self._rule_based_explain(context, media_type)

    # ─── CONTEXT BUILDER ───────────────────────────────────────────────────────

    def _build_context(self, r: dict, media_type: str) -> dict:
        attrs = r.get('attributes', {}) or {}
        reg   = r.get('region_result', {}) or {}
        return {
            'media_type':        media_type,
            'prediction':        r.get('prediction', 'Unknown'),
            'confidence':        round(float(r.get('confidence', 0)), 1),
            'real_probability':  round(float(r.get('real_probability', 0)), 1),
            'fake_probability':  round(float(r.get('fake_probability', 0)), 1),
            'trust_score':       round(float(r.get('trust_score', 0)), 1),
            'interpretation':    r.get('interpretation', 'Unknown'),
            'ela_avg':           round(float(r.get('ela_avg', 0)), 2),
            'ela_max':           round(float(r.get('ela_max', 0)), 2),
            'metadata_score':    round(float(r.get('metadata_score', 100)), 1),
            'metadata_flags':    r.get('metadata_flags', []),
            'hash_score':        round(float(r.get('hash_score', 100)), 1),
            'hash_message':      r.get('hash_message', ''),
            'edited_detected':   reg.get('edited_detected', False),
            'edited_area_pct':   round(float(reg.get('edited_area_percentage', 0)), 1),
            'suspicious_regions':reg.get('suspicious_regions', []),
            'gender':            attrs.get('gender', {}).get('prediction', 'N/A'),
            'face_shape':        attrs.get('face_shape', {}).get('prediction', 'N/A'),
            'hair_texture':      attrs.get('hair_texture', {}).get('prediction', 'N/A'),
            'hair_color':        attrs.get('hair_color', {}).get('prediction', 'N/A'),
            'skin_tone':         attrs.get('skin_tone', {}).get('prediction', 'N/A'),
            'total_frames':      r.get('total_frames', 0),
            'fake_frame_count':  r.get('fake_frame_count', 0),
            'real_frame_count':  r.get('real_frame_count', 0),
        }

    # ─── GROQ API EXPLANATION ─────────────────────────────────────────────────

    def _groq_explain(self, client, ctx: dict, media_type: str) -> dict:
        # Use a valid Groq model — override any invalid xAI model names from .env
        env_model = os.environ.get("GROK_MODEL", "llama-3.3-70b-versatile")
        # Map any invalid/xAI model names to valid Groq models
        _GROQ_MODEL_MAP = {
            'grok-3-mini': 'llama-3.3-70b-versatile',
            'grok-3':      'llama-3.3-70b-versatile',
            'grok-2':      'llama-3.1-8b-instant',
        }
        model = _GROQ_MODEL_MAP.get(env_model, env_model)

        system_prompt = (
            "You are VeriFakeNet's expert forensic AI analyst. "
            "Given structured deepfake detection results, write a clear, professional "
            "forensic explanation in exactly 4-6 sentences. "
            "Interpret the numbers meaningfully — don't just repeat them. "
            "Cover: verdict, key evidence, reliability, and what face attributes suggest. "
            "Plain English only. No bullet points, no markdown."
        )

        user_prompt = (
            f"Explain these {media_type} forensic analysis results:\n"
            f"{json.dumps(ctx, indent=2)}"
        )

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=400,
            temperature=0.3,
        )

        raw_text = response.choices[0].message.content.strip()
        verdict  = self._make_verdict(ctx)

        return {
            'text':   self._wrap_html(raw_text, ctx, source='groq'),
            'source': 'groq',
            'verdict': verdict,
            'raw':    raw_text,
        }

    # ─── RULE-BASED FALLBACK ──────────────────────────────────────────────────

    def _rule_based_explain(self, ctx: dict, media_type: str) -> dict:
        pred       = ctx['prediction']
        conf       = ctx['confidence']
        trust      = ctx['trust_score']
        ela        = ctx['ela_avg']
        meta_score = ctx['metadata_score']
        edited_pct = ctx['edited_area_pct']
        regions    = ', '.join(ctx['suspicious_regions']) or 'the face region'
        mflags     = '; '.join(ctx['metadata_flags']) or 'No anomalies detected.'
        pred_color = '#f87171' if 'fake' in pred.lower() else '#34d399'

        if 'fake' in pred.lower():
            det = (f"The {media_type} is classified as "
                   f"<strong style='color:{pred_color};'>FAKE / EDITED</strong> "
                   f"with <strong>{conf}%</strong> model confidence.")
        elif 'real' in pred.lower():
            det = (f"The {media_type} appears "
                   f"<strong style='color:{pred_color};'>AUTHENTIC</strong> "
                   f"with <strong>{conf}%</strong> model confidence.")
        else:
            det = "The detection result was <strong>inconclusive</strong> — face detection may have failed."

        ela_s = (
            f"ELA detected <strong>severe compression artifacts</strong> (avg={ela:.1f}), "
            f"strongly indicative of post-processing."
            if ela > 12 else
            f"ELA detected <strong>moderate compression inconsistencies</strong> (avg={ela:.1f}), "
            f"suggesting possible localized editing."
            if ela > 5 else
            f"ELA shows <strong>normal compression patterns</strong> (avg={ela:.1f}), "
            f"consistent with authentic media."
        )

        gc_s = (
            f"Grad-CAM identified manipulations in <strong>{regions}</strong> "
            f"covering approximately {edited_pct:.1f}% of the face area."
            if ctx['edited_detected'] else
            "Grad-CAM found no significant spatial anomalies across the face regions."
        )

        meta_s = (
            f"Metadata forensics raised <strong>critical flags</strong>: {mflags}"
            if meta_score < 70 else
            f"Metadata shows <strong>minor inconsistencies</strong>: {mflags}"
            if meta_score < 90 else
            "Metadata analysis returned <strong>no suspicious flags</strong>."
        )

        vid_s = ""
        if media_type == "video" and ctx['total_frames'] > 0:
            fake_f = ctx['fake_frame_count']
            real_f = ctx['real_frame_count']
            total  = ctx['total_frames']
            vid_s  = (
                f" Frame analysis processed <strong>{total} frames</strong>: "
                f"<span style='color:#f87171;'>{fake_f} fake</span> / "
                f"<span style='color:#34d399;'>{real_f} real</span>."
            )

        trust_s = (
            f"Overall Trust Score: <strong style='color:#34d399;'>{trust}/100</strong> — "
            f"evidence collectively supports the classification."
            if trust >= 70 else
            f"Overall Trust Score: <strong style='color:#f59e0b;'>{trust}/100</strong> — "
            f"moderate confidence; manual review is recommended."
            if trust >= 40 else
            f"Overall Trust Score: <strong style='color:#f87171;'>{trust}/100</strong> — "
            f"strong indicators of manipulation across multiple forensic signals."
        )

        full = f"{det}<br><br>{ela_s}<br>{gc_s}<br>{meta_s}{vid_s}<br><br>{trust_s}"
        return {
            'text':   self._wrap_html(full, ctx, source='rule_based'),
            'source': 'rule_based',
            'verdict': self._make_verdict(ctx),
        }

    # ─── HELPERS ─────────────────────────────────────────────────────────────

    def _make_verdict(self, ctx: dict) -> str:
        pred  = ctx['prediction']
        conf  = ctx['confidence']
        trust = ctx['trust_score']
        if 'fake' in pred.lower():
            return f"FAKE detected ({conf}% confidence, trust {trust}/100)"
        elif 'real' in pred.lower():
            return f"AUTHENTIC ({conf}% confidence, trust {trust}/100)"
        return f"Inconclusive (trust {trust}/100)"

    def _wrap_html(self, text: str, ctx: dict, source: str) -> str:
        badge_color = '#1a4a3a' if source == 'groq' else '#1e2a3f'
        badge_text_color = '#34d399' if source == 'groq' else '#60a5fa'
        badge_label = 'Groq AI' if source == 'groq' else 'Rule-Based'
        return f"""
        <div class="ai-explain-box">
            <div style="font-size:0.7rem;font-weight:700;letter-spacing:0.1em;
                        color:#7ea8c9;margin-bottom:10px;display:flex;align-items:center;gap:8px;">
                AI FORENSIC EXPLANATION
                <span style="font-size:0.65rem;background:{badge_color};color:{badge_text_color};
                    padding:2px 8px;border-radius:10px;font-weight:600;">{badge_label}</span>
            </div>
            <div style="line-height:1.8;font-size:0.88rem;color:#cbd5e1;">
                {text}
            </div>
        </div>
        """
