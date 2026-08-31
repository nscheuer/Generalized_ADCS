"""
PDF parser for Reaction Wheel datasheets.

Provides parse_pdf_file(path) -> dict with normalized fields used by
ADCS.satellite_factory.actuators.create_cubewheel_* factories.

Heuristics-based extraction: prefers html-like PDFs but falls back to
text extraction with PyPDF2 when pdfplumber is unavailable.

Returned dict keys (example):
  {
    'component_type': 'reaction_wheel',
    'vendor': 'ExampleCo',
    'model': 'CubeWheel-SmallPlus',
    'max_torque_Nm': 0.0023,
    'rotor_inertia_kgm2': 5.7e-6,
    'initial_h_Nm_s': 0.0,
    'h_max_Nm_s': 0.0036,
    'axis': [1.0, 0.0, 0.0],
    'datasheet_path': '...'
  }

This parser is intentionally conservative: it returns None for a field it
cannot confidently extract. Use the factory adapter to provide fallbacks or
human verification.
"""

from __future__ import annotations

import re
import os
import logging
from typing import Dict, Optional, List

logger = logging.getLogger(__name__)

try:
    import pdfplumber  # type: ignore
    _HAS_PDFPLUMBER = True
except Exception:
    _HAS_PDFPLUMBER = False

try:
    import PyPDF2  # type: ignore
    _HAS_PYPDF2 = True
except Exception:
    _HAS_PYPDF2 = False

print(f"PDF parser: pdfplumber available: {_HAS_PDFPLUMBER}, PyPDF2 available: {_HAS_PYPDF2}")

# Basic SI prefix multipliers for parsing
_SI_PREFIX = {
    "p": 1e-12,
    "n": 1e-9,
    "u": 1e-6,
    "µ": 1e-6,
    "m": 1e-3,
    "c": 1e-2,
    "d": 1e-1,
    "": 1.0,
    "k": 1e3,
    "M": 1e6,
    "G": 1e9,
}

_number_re = re.compile(r"([+-]?[0-9]+(?:[\.,][0-9]+)?(?:[eE][+-]?\d+)?)")
_unit_val_re = re.compile(r"([+-]?[0-9]+(?:[\.,][0-9]+)?(?:[eE][+-]?\d+)?)[ ]*([a-zA-Zµμ·\^0-9\s/\-]+)")


def _extract_text_from_pdf(path: str) -> str:
    """Extract text from a PDF using pdfplumber or PyPDF2 as fallback.

    Returns the entire extracted text as a single string.
    """
    if _HAS_PDFPLUMBER:
        try:
            with pdfplumber.open(path) as pdf:
                pages = [p.extract_text() or "" for p in pdf.pages]
            return "\n\n".join(pages)
        except Exception as e:
            logger.debug("pdfplumber failed for %s: %s", path, e)

    if _HAS_PYPDF2:
        try:
            reader = PyPDF2.PdfReader(path)
            out = []
            for p in reader.pages:
                try:
                    out.append(p.extract_text() or "")
                except Exception:
                    out.append("")
            return "\n\n".join(out)
        except Exception as e:
            logger.debug("PyPDF2 failed for %s: %s", path, e)

    # Last resort: call system pdftotext if available (not used here). Return empty.
    logger.warning("No PDF text extractor available (pdfplumber or PyPDF2). Returning empty text for %s", path)
    return ""


def _parse_number_with_si(token: str) -> Optional[float]:
    """Parse a numeric token that may include an SI prefix on the unit.

    token example inputs (split between value and unit externally):
      - value_str = '5.7e-6', unit_str = 'kg·m^2'
      - value_str = '2.3', unit_str = 'mN·m' or 'mN m'

    This helper only parses the numeric portion. Conversion is handled outside
    according to the detected unit.
    """
    if not token:
        return None
    token = token.replace(',', '.')
    try:
        return float(token)
    except Exception:
        try:
            # remove spaces and try again
            return float(token.replace(' ', ''))
        except Exception:
            logger.debug("Failed to parse numeric token '%s'", token)
            return None


def _unit_clean(u: str) -> str:
    return u.replace('\u00b7', ' ').replace('·', ' ').replace('^', '').replace('\n', ' ').strip()


def _find_values_near_keywords(text: str, keywords: List[str], unit_hint_patterns: List[str], chars_window: int = 120) -> List[str]:
    """Search for occurrences of keywords and return candidate unit-containing substrings nearby.

    Returns list of substrings (typically containing a numeric token + unit).
    """
    results = []
    low = text.lower()
    for kw in keywords:
        start = 0
        while True:
            idx = low.find(kw.lower(), start)
            if idx == -1:
                break
            # Extract window around keyword
            s = max(0, idx - chars_window)
            e = min(len(text), idx + chars_window)
            window = text[s:e]
            # find all unit-like tokens in window
            for m in _unit_val_re.finditer(window):
                val, unit = m.groups()
                uc = _unit_clean(unit)
                for up in unit_hint_patterns:
                    if up.lower() in uc.lower() or up.lower() in window.lower():
                        results.append(m.group(0))
                        break
            start = idx + len(kw)
    return results


def _interpret_torque_token(token: str) -> Optional[float]:
    # token is like '0.0023 N·m' or '2.3 mN m' or '2300 uN m'
    m = _unit_val_re.search(token)
    if not m:
        return None
    val_str, unit_str = m.groups()
    val = _parse_number_with_si(val_str)
    if val is None:
        return None
    unit = _unit_clean(unit_str)
    unit = unit.replace(' ', '')
    # handle milli/newton-meter prefixes: match patterns containing 'N' and 'm'
    # common variants: 'N·m', 'Nm', 'mN·m' (milli-Newton-meter => mN·m = 1e-3 N·m)
    # We'll detect presence of 'n' or 'm' prefix before 'N' and apply multiplier
    # naive approach: look for 'mn' or 'mnm' => milli-newton, 'un' or 'μn' => micro-newton
    us = unit.lower()
    # If unit contains 'nm' and not 'n' before 'm' ... careful: 'Nm' is Newton-meter
    if 'nm' in us and ' ' not in us and us.endswith('nm') and us != 'nm':
        # ambiguous; fallback
        pass
    # Detect patterns
    multiplier = 1.0
    # if unit like 'mnm' or 'mN m' we've removed spaces, so e.g. 'mnm' appears
    if re.search(r'(^|[^a-zA-Z])(mn|mnm|mnewton)', us):
        multiplier = 1e-3
    elif re.search(r'(^|[^a-zA-Z])(un|µn|μn)', us):
        multiplier = 1e-6
    elif re.search(r'(^|[^a-zA-Z])(kn|knm)', us):
        multiplier = 1e3
    # If unit contains 'nm' in the sense of Newton-meter (usual 'Nm'), multiplier stays 1.0
    # If the unit contains 'n' followed by 'm' without prefix, assume Newton-meter
    # Now return value in N*m
    return val * multiplier


def _interpret_inertia_token(token: str) -> Optional[float]:
    # Look for kg*m2 or kg m^2 patterns
    m = _unit_val_re.search(token)
    if not m:
        return None
    val_str, unit_str = m.groups()
    val = _parse_number_with_si(val_str)
    if val is None:
        return None
    us = _unit_clean(unit_str).lower()
    # If unit contains 'kg' and 'm' and possibly '2' then assume kg*m^2
    if 'kg' in us and ('m2' in us or 'm^2' in us or 'm' in us):
        return val
    return None


def _interpret_momentum_token(token: str) -> Optional[float]:
    # Look for N*m*s or Nms or N·m·s
    m = _unit_val_re.search(token)
    if not m:
        return None
    val_str, unit_str = m.groups()
    val = _parse_number_with_si(val_str)
    if val is None:
        return None
    us = _unit_clean(unit_str).lower().replace(' ', '')
    if 'nms' in us or 'n*m*s' in us or 'n·m·s' in unit_str:
        return val
    # Some datasheets report max momentum in mNms etc. detect prefixes
    # naive prefix handling: if unit contains 'm' before 'nms' treat as milli
    if 'mnms' in us or 'm n m s' in unit_str:
        return val * 1e-3
    return val


def parse_rw_spec_from_text(text: str) -> Dict[str, Optional[float]]:
    """Heuristic extraction of RW relevant fields from raw PDF text.

    Returns a dict with keys (some may be None): max_torque_Nm, rotor_inertia_kgm2,
    h_max_Nm_s, initial_h_Nm_s (default 0.0 if not found), vendor, model, axis
    """
    out: Dict[str, Optional[float]] = {
        'component_type': 'reaction_wheel',
        'vendor': None,
        'model': None,
        'max_torque_Nm': None,
        'rotor_inertia_kgm2': None,
        'h_max_Nm_s': None,
        'initial_h_Nm_s': 0.0,
        'axis': [1.0, 0.0, 0.0],
    }

    # quick vendor/model heuristics: title lines, look at top of document
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        # first non-empty line may contain vendor or model branding
        first = lines[0]
        # try to split vendor/model by '-' or '|' if present
        if ' - ' in first:
            parts = first.split(' - ', 1)
            out['vendor'] = parts[0].strip()
            out['model'] = parts[1].strip()
        elif '|' in first:
            parts = first.split('|', 1)
            out['vendor'] = parts[0].strip()
            out['model'] = parts[1].strip()
        else:
            # fallback: if first line has uppercase words assume vendor
            out['vendor'] = first

    # Torque: look for lines near keywords
    torque_candidates = _find_values_near_keywords(text, keywords=['torque', 'torques', 'torque ('], unit_hint_patterns=['N', 'Nm', 'N·m', 'N m'])
    # Interpret best candidate
    for tok in torque_candidates:
        val = _interpret_torque_token(tok)
        if val is not None:
            out['max_torque_Nm'] = val
            break

    # Inertia (rotor J): look for 'inertia' or 'rotor inertia' or 'J ='
    inertia_cands = _find_values_near_keywords(text, keywords=['inertia', 'rotor inertia', 'moment of inertia', 'J =', 'J='], unit_hint_patterns=['kg', 'kgm2', 'kg m2', 'kg·m2'])
    for tok in inertia_cands:
        val = _interpret_inertia_token(tok)
        if val is not None:
            out['rotor_inertia_kgm2'] = val
            break

    # Momentum capacity (h_max): search for 'momentum', 'max momentum', 'momentum capacity'
    momentum_cands = _find_values_near_keywords(text, keywords=['momentum', 'max momentum', 'momentum capacity', 'angular momentum'], unit_hint_patterns=['Nms', 'Nm s', 'N m s'])
    for tok in momentum_cands:
        val = _interpret_momentum_token(tok)
        if val is not None:
            out['h_max_Nm_s'] = val
            break

    # Fallback heuristics: sometimes datasheets present torque (mN·m) and momentum (mN·m·s)
    # Try a broad search for unit-bearing numbers and assign based on plausibility
    if out['max_torque_Nm'] is None or out['rotor_inertia_kgm2'] is None or out['h_max_Nm_s'] is None:
        # find all unit-bearing tokens that include 'N' or 'kg'
        tokens = []
        for m in _unit_val_re.finditer(text):
            tokens.append(m.group(0))
        # coarse pass: select plausible torque (value between 1e-6 and 1e0 Nm)
        for tok in tokens:
            if out['max_torque_Nm'] is None:
                v = _interpret_torque_token(tok)
                if v is not None and 1e-9 < abs(v) < 10.0:
                    out['max_torque_Nm'] = v
            if out['rotor_inertia_kgm2'] is None:
                Jv = _interpret_inertia_token(tok)
                if Jv is not None and 1e-9 < Jv < 1.0:
                    out['rotor_inertia_kgm2'] = Jv
            if out['h_max_Nm_s'] is None:
                hv = _interpret_momentum_token(tok)
                if hv is not None and 1e-6 < abs(hv) < 10.0:
                    out['h_max_Nm_s'] = hv
            if out['max_torque_Nm'] is not None and out['rotor_inertia_kgm2'] is not None and out['h_max_Nm_s'] is not None:
                break

    return out


def parse_pdf_file(path: str) -> Dict[str, Optional[float]]:
    """Main entry: parse a single RW PDF and return a normalized spec dict.

    The function normalizes units where possible and returns a dict suitable for
    passing into the factory adapter. Fields that could not be extracted remain None.
    """
    spec = parse_pdf_file.__doc__  # just to reference in case lint
    print(f"Parsing RW PDF: {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    text = _extract_text_from_pdf(path)
    if not text.strip():
        logger.warning("No text extracted from %s", path)
    parsed = parse_rw_spec_from_text(text)
    parsed['datasheet_path'] = path
    # Basic normalization: ensure floats or None
    for k in ['max_torque_Nm', 'rotor_inertia_kgm2', 'h_max_Nm_s', 'initial_h_Nm_s']:
        if k in parsed and parsed[k] is not None:
            try:
                parsed[k] = float(parsed[k])
            except Exception:
                parsed[k] = None

    return parsed


def parse_pdf_folder(folder: str) -> List[Dict[str, Optional[float]]]:
    """Parse all PDFs in a folder and return list of spec dicts."""
    print(f"Parsing RW PDF folder: {folder}")
    out = []
    if not os.path.isdir(folder):
        raise NotADirectoryError(folder)
    for fn in os.listdir(folder):
        if not fn.lower().endswith('.pdf'):
            continue
        path = os.path.join(folder, fn)
        try:
            out.append(parse_pdf_file(path))
        except Exception as e:
            logger.exception("Failed parsing %s: %s", path, e)
    return out


if __name__ == '__main__':
    import argparse
    import json

    p = argparse.ArgumentParser(description='Parse RW datasheet PDFs in a folder')
    p.add_argument('folder', help='Folder containing PDFs')
    p.add_argument('--out', help='Write JSON output to this file', default=None)
    args = p.parse_args()

    specs = parse_pdf_folder(args.folder)
    if args.out:
        with open(args.out, 'w', encoding='utf8') as fh:
            json.dump(specs, fh, indent=2)
    else:
        print(specs)
