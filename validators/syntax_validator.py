"""Stage 1: Syntax validator — Suricata -T rule syntax checking.

Invokes the Suricata binary to validate rule syntax. Captures and parses
error output for structured feedback generation.
"""

import os
import re
import subprocess
import tempfile
from typing import Tuple, Optional
from pathlib import Path

from ..utils.suricata_utils import SuricataRule
from ..utils.logging import get_logger


class SyntaxValidator:
    """Validate Suricata rule syntax by invoking suricata -T.

    Args:
        suricata_binary: Path to the suricata executable (default: 'suricata').
        timeout: Maximum time in seconds for the validation command.
        temp_dir: Directory for temporary rule files.
    """

    def __init__(
        self,
        suricata_binary: str = "suricata",
        timeout: int = 30,
        temp_dir: str = None,
    ):
        self.suricata_binary = suricata_binary
        self.timeout = timeout
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self.logger = get_logger("pcap2rule.syntax")

    def check(self, rule: SuricataRule) -> Tuple[bool, str]:
        """Validate rule syntax using suricata -T.

        Args:
            rule: SuricataRule to validate.

        Returns:
            (is_valid, feedback_message) tuple. is_valid=True if syntax is correct.
        """
        if rule is None:
            return False, "[SYNTAX ERROR] Null rule — rule parsing failed entirely."

        rule_str = rule.to_string()

        # Quick local validation first
        local_errors = self._local_check(rule_str)
        if local_errors:
            return False, "\n".join(local_errors)

        # Invoke suricata -T
        suricata_ok, suricata_output = self._suricata_check(rule_str)
        if suricata_ok:
            return True, ""

        # Parse suricata error output into structured feedback
        feedback = self._parse_suricata_error(suricata_output, rule_str)
        return False, feedback

    def _local_check(self, rule_str: str) -> list:
        """Perform local syntax checks without invoking suricata."""
        errors = []

        # Check parentheses balance
        if rule_str.count('(') != rule_str.count(')'):
            errors.append(
                "[SYNTAX ERROR] Unbalanced parentheses: "
                f"{rule_str.count('(')} open, {rule_str.count(')')} close."
            )

        # Check quote balance
        if rule_str.count('"') % 2 != 0:
            errors.append(
                "[SYNTAX ERROR] Unbalanced double quotes — check content/pcre strings."
            )

        # Check required fields
        if 'msg:' not in rule_str:
            errors.append(
                "[SYNTAX ERROR] Missing required 'msg' field in rule options."
            )
        if 'sid:' not in rule_str:
            errors.append(
                "[SYNTAX ERROR] Missing required 'sid' field in rule options."
            )

        # Check for common content string issues
        content_match = re.findall(r'content:"([^"]*)"', rule_str)
        for i, c in enumerate(content_match):
            if len(c) == 0:
                errors.append(
                    f"[SYNTAX WARNING] content:{i+1} is empty — "
                    "this matches every packet."
                )

        # Check semicolons between options
        options_part = rule_str.split('(', 1)[-1].rsplit(')', 1)[0] if '(' in rule_str else ''
        for opt in options_part.split(';"'):
            opt = opt.strip()
            if opt and not opt.endswith(';') and not opt.endswith('"'):
                errors.append(
                    f"[SYNTAX WARNING] Possible missing semicolon near: '{opt[:50]}'"
                )

        return errors

    def _suricata_check(self, rule_str: str) -> Tuple[bool, str]:
        """Invoke suricata -T on a temporary rule file.

        Returns:
            (success, output) tuple.
        """
        # Write rule to temp file
        rule_file = os.path.join(self.temp_dir, f"_pcap2rule_check_{os.getpid()}.rules")
        try:
            with open(rule_file, 'w') as f:
                f.write(rule_str + "\n")

            cmd = [
                self.suricata_binary, '-T',
                '-l', self.temp_dir,
                '-r', rule_file,
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )

            output = result.stdout + result.stderr
            success = result.returncode == 0

            return success, output.strip()

        except subprocess.TimeoutExpired:
            return False, "[SYNTAX ERROR] Suricata validation timed out."
        except FileNotFoundError:
            self.logger.warning(
                f"Suricata binary '{self.suricata_binary}' not found. "
                "Skipping external syntax check."
            )
            return True, ""  # Can't verify, assume OK
        finally:
            try:
                os.remove(rule_file)
            except OSError:
                pass

    def _parse_suricata_error(self, output: str, rule_str: str) -> str:
        """Parse suricata error output into structured, actionable feedback."""
        lines = output.split('\n')
        feedback_lines = ["[SYNTAX ERROR] Suricata validation failed:"]

        error_patterns = [
            (r"(Error:.*)", "Error"),
            (r"(error:.*)", "Error"),
            (r"(Warning:.*)", "Warning"),
            (r"line (\d+)", "Line"),
            (r"unknown keyword", "UnknownKeyword"),
            (r"unexpected token", "UnexpectedToken"),
            (r"not a valid", "Invalid"),
        ]

        for line in lines:
            for pattern, _ in error_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    feedback_lines.append(f"  {line.strip()}")
                    break

        if len(feedback_lines) == 1:
            feedback_lines.append(f"  Raw output: {output[:300]}")

        # Add suggestion based on error type
        output_lower = output.lower()
        if 'content' in output_lower:
            feedback_lines.append(
                "Suggestion: Check content string formatting. Use content:\"...\""
            )
        if 'pcre' in output_lower:
            feedback_lines.append(
                "Suggestion: Check PCRE regex syntax. Ensure pattern is properly escaped."
            )

        return "\n".join(feedback_lines)
