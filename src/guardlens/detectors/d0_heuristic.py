"""D0: transparent heuristic baseline.

Hand-written regex rules, grouped by attack category, each with a hand-set weight.
Score = 1 - prod(1 - w) over the rules that fire (several weak signals add up, one strong
signal is enough). Rules are generic markers of each category, not copied from any dataset.
"""

import re
from dataclasses import dataclass
from typing import Any

from guardlens.detectors.base import NO_CATEGORY, Detector, DetectorOutput, Direction

RULES_VERSION = "d0-rules-v1"
FLAGS = re.IGNORECASE | re.MULTILINE


@dataclass(frozen=True)
class Rule:
    name: str
    category: str
    weight: float
    pattern: re.Pattern[str]


def _rule(name: str, category: str, weight: float, regex: str) -> Rule:
    return Rule(name, category, weight, re.compile(regex, FLAGS))


_PRIOR = r"(previous|prior|above|earlier|preceding|initial|original|all|any|your)"
_INSTR = r"(instructions?|rules|directions|directives|guidelines|prompts?|context)"

RULES: list[Rule] = [
    # --- instruction override: tries to cancel or replace the app's instructions
    _rule("ignore_prior", "instruction_override", 0.8,
          rf"\b(ignore|disregard|forget|override|bypass|skip)\b.{{0,30}}\b{_PRIOR}\b.{{0,20}}\b{_INSTR}\b"),
    _rule("new_instructions", "instruction_override", 0.5,
          r"\b(new|updated|real|actual)\s+(instructions?|rules|task)\s*:"),
    _rule("from_now_on", "instruction_override", 0.3,
          r"\bfrom now on\b.{0,40}\b(you|your)\b"),
    _rule("ignore_prior_de", "instruction_override", 0.8,
          r"\b(ignorier\w*|vergiss\w*)\b.{0,30}\b(vorherige\w*|bisherige\w*|alle\w*|obige\w*)\b.{0,20}\b(anweisung\w*|befehl\w*|regeln)\b"),
    _rule("ignore_prior_fr", "instruction_override", 0.8,
          r"\b(ignore[rz]?|oublie[rz]?)\b.{0,30}\b(les|toutes?|tes|vos)\b.{0,20}\b(instructions?|consignes|règles)\b"),
    _rule("ignore_prior_es", "instruction_override", 0.8,
          r"\b(ignora\w*|olvida\w*)\b.{0,30}\b(las|todas|tus|sus)\b.{0,20}\b(instrucciones|reglas|órdenes)\b"),
    # --- prompt / secret extraction: tries to read the hidden prompt or a secret
    _rule("reveal_prompt", "prompt_extraction", 0.7,
          r"\b(reveal|show|print|repeat|output|display|leak|tell me|give me|write out)\b.{0,40}\b(system prompt|initial prompt|hidden prompt|your (instructions|prompt|rules)|pre-?prompt)\b"),
    _rule("ask_prompt", "prompt_extraction", 0.5,
          r"\bwhat\s+(is|are|were)\s+your\s+(system prompt|instructions|initial prompt|rules)\b"),
    _rule("secret_word", "prompt_extraction", 0.4,
          r"\b(password|passphrase|secret (word|key|code))\b"),
    _rule("spell_out", "prompt_extraction", 0.3,
          r"\b(spell|letter by letter|one letter at a time|each letter|backwards|reverse order)\b"),
    # --- persona / jailbreak framing: role-play set up to remove the model's limits
    _rule("unrestricted_persona", "persona_jailbreak", 0.7,
          r"\b(no|without|free (of|from)|zero)\s+(any\s+)?(restrictions|limits|limitations|filters|censorship|rules|boundaries|ethic\w*|moral\w*)\b"),
    _rule("unfiltered_adj", "persona_jailbreak", 0.5,
          r"\b(unrestricted|unfiltered|uncensored|unlimited|amoral|unethical)\b"),
    _rule("dan_mode", "persona_jailbreak", 0.8,
          r"\b(DAN|do anything now|developer mode|jailbreak(ed)?|god mode)\b"),
    _rule("stay_in_character", "persona_jailbreak", 0.4,
          r"\b(stay in character|never break character|you must never refuse|you (can|will) not refuse)\b"),
    _rule("act_as", "persona_jailbreak", 0.15,
          r"\b(act as|pretend (to be|you are)|you are now|roleplay as|simulate)\b"),
    # --- fake delimiters: imitates system / chat-template markup inside user text
    _rule("fake_role_line", "fake_delimiter", 0.5,
          r"^\s*(system|assistant|developer)\s*:"),
    _rule("chat_template_tokens", "fake_delimiter", 0.7,
          r"(<\|?(system|im_start|im_end|endoftext)\|?>|\[/?(INST|SYS)\]|<</?SYS>>)"),
    _rule("markdown_header_role", "fake_delimiter", 0.4,
          r"^\s*#{2,}\s*(system|instruction|admin)\b"),
    # --- obfuscation: hides the payload from simple filters
    _rule("base64_blob", "obfuscation", 0.4, r"[A-Za-z0-9+/]{48,}={0,2}"),
    _rule("decode_request", "obfuscation", 0.4,
          r"\b(decode|base64|rot13|hex(adecimal)?|caesar cipher)\b.{0,30}\b(this|following|message|text|string)\b"),
    _rule("zero_width_chars", "obfuscation", 0.5, r"[​‌‍⁠﻿]"),
]


class HeuristicDetector(Detector):
    name = "d0"

    def __init__(self, rules: list[Rule] | None = None) -> None:
        self.rules = RULES if rules is None else rules

    def score(self, text: str, direction: Direction = "input") -> DetectorOutput:
        fired = [r for r in self.rules if r.pattern.search(text)]
        if not fired:
            return DetectorOutput(score=0.0, category=NO_CATEGORY, reason="no rule fired")
        keep = 1.0
        for r in fired:
            keep *= 1.0 - r.weight
        strongest = max(fired, key=lambda r: r.weight)
        return DetectorOutput(
            score=round(1.0 - keep, 6),
            category=strongest.category,
            reason="rules: " + ", ".join(r.name for r in fired),
        )

    def config(self) -> dict[str, Any]:
        return {"name": self.name, "rules_version": RULES_VERSION, "n_rules": len(self.rules)}
