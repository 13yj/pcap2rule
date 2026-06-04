#!/usr/bin/env python
"""Smoke tests for Pcap2Rule — tests core functionality that works CPU-only.

Run from project PARENT directory:
    python pcap2rule/tests/smoke_test.py
"""
import sys, os, json, tempfile
import numpy as np

# Ensure we're not in the pcap2rule directory (to avoid shadowing issues)
if os.path.basename(os.getcwd()) == 'pcap2rule':
    os.chdir('..')

TESTS_PASSED = 0
TESTS_FAILED = 0

def check(name, condition, detail=""):
    global TESTS_PASSED, TESTS_FAILED
    if condition:
        TESTS_PASSED += 1
        print(f"  [OK] {name}")
    else:
        TESTS_FAILED += 1
        print(f"  [FAIL] {name}: {detail}")

def test_config():
    print("\n--- Config ---")
    from pcap2rule.utils.config import Config, load_config
    cfg = Config({'a': {'b': 1, 'c': 'hello'}, 'd': [1, 2, 3]})
    check("Config attribute access", cfg.a.b == 1)
    check("Config list access", cfg.d == [1, 2, 3])

def test_seed():
    print("\n--- Seed ---")
    from pcap2rule.utils.seed import set_seed, get_seeds
    set_seed(42)
    check("set_seed runs", True)
    seeds = get_seeds()
    check("get_seeds returns list", isinstance(seeds, list) and len(seeds) >= 5)

def test_suricata_utils():
    print("\n--- Suricata Utils ---")
    from pcap2rule.utils.suricata_utils import (
        SuricataRule, validate_rule_syntax, parse_suricata_rule
    )

    rule = SuricataRule(
        action='alert', protocol='http',
        src_ip='$EXTERNAL_NET', src_port='any',
        dst_ip='$HOME_NET', dst_port='any',
        msg='"SQL Injection Attempt"',
        content=['"UNION SELECT"'],
        sid=100001, rev=1,
    )
    rule_str = rule.to_string()
    check("Rule to_string non-empty", len(rule_str) > 0, rule_str)
    check("Rule contains action", 'alert' in rule_str)
    check("Rule contains content", 'UNION SELECT' in rule_str)

    rule.generate_sid(rule_str)
    check("SID generation", rule.sid >= 100000)

    errors = validate_rule_syntax(rule)
    check("Local syntax check", len(errors) == 0, str(errors))

    parsed = parse_suricata_rule(rule_str)
    check("Rule parsing", parsed is not None and parsed.action == 'alert')
    check("Parsed content match", 'UNION SELECT' in parsed.content[0])

def test_pcap_utils():
    print("\n--- PCAP Utils ---")
    from pcap2rule.utils.pcap_utils import compute_pcap_hash, is_pcap_file
    check("pcap_utils imports", True)

def test_logging():
    print("\n--- Logging ---")
    with tempfile.TemporaryDirectory() as d:
        from pcap2rule.utils.logging import setup_logging, get_logger
        setup_logging(d, level='DEBUG')
        logger = get_logger("test")
        logger.info("test message")
        check("Logger works", True)

def test_metrics():
    print("\n--- Evaluation Metrics ---")
    from pcap2rule.evaluation.metrics import (
        syntactic_validity, detection_coverage,
        false_positive_rate, f1_score, compute_all_metrics
    )
    from pcap2rule.utils.suricata_utils import parse_suricata_rule

    # SV: create SuricataRule objects
    r1 = parse_suricata_rule('alert http any any -> any any (msg:"test"; sid:1; rev:1;)')
    r2 = parse_suricata_rule('alert tcp any any -> any any (msg:"test2"; sid:2; rev:1;)')
    sv = syntactic_validity([r1, r2])
    check("Syntactic validity (all valid)", sv == 100.0, f"got {sv}")

    # SV: one invalid (None = invalid)
    sv2 = syntactic_validity([None, r2])
    check("Syntactic validity (partial)", sv2 == 50.0, f"got {sv2}")

    # DC
    dc = detection_coverage([r1, r2], [True, True, False, True])
    check("Detection coverage", dc == 75.0, f"got {dc}")

    # FPR
    fpr = false_positive_rate([r1, r2], [2, 1], 100)
    check("False positive rate", abs(fpr - 1.5) < 0.1, f"got {fpr}")

    # F1 (from precision/recall)
    f1 = f1_score(precision=0.9, recall=0.85)
    check("F1 score (from p/r)", abs(f1 - 0.8743) < 0.01, f"got {f1}")

    # compute_all
    all_m = compute_all_metrics(
        rules=[r1, r2],
        generalized_rules=[r1, r2],
        detection_results=[True, True],
        variant_results=[[True, False], [True, True]],
        benign_matches=[0, 1],
        benign_total=100,
    )
    check("compute_all_metrics returns dict", isinstance(all_m, dict))
    check("SV in all_metrics", 'SV' in all_m)
    check("DC in all_metrics", 'DC' in all_m)

def test_statistical():
    print("\n--- Statistical ---")
    from pcap2rule.evaluation.statistical import (
        bootstrap_confidence_interval, mcnemar_test,
        compute_mean_std_across_seeds,
    )

    vals = [0.85, 0.87, 0.83, 0.89, 0.86, 0.88, 0.84, 0.87, 0.85, 0.86]
    mean, (lo, hi) = bootstrap_confidence_interval(vals, n_bootstrap=500)
    check("Bootstrap CI: mean in range", lo <= mean <= hi, f"{mean} [{lo}, {hi}]")

    a = [True, True, False, True, True, True, False, True]
    b = [True, False, False, True, True, True, True, True]
    chi2, p = mcnemar_test(a, b)
    check("McNemar returns chi2", chi2 >= 0, f"chi2={chi2}")
    check("McNemar returns p-value", 0 <= p <= 1, f"p={p}")

    per_seed = [{'SV': 92, 'DC': 87}, {'SV': 93, 'DC': 86}, {'SV': 91, 'DC': 88}]
    ms = compute_mean_std_across_seeds(per_seed)
    check("Mean-std across seeds", abs(ms['SV'][0] - 92.0) < 0.1)

def test_human_eval():
    print("\n--- Human Evaluation ---")
    from pcap2rule.evaluation.human_eval import (
        compute_fleiss_kappa, compute_approval_rates,
    )

    # 5 rules, 3 raters, each rates 0-2
    ratings = [
        [2, 2, 2],  # All agree: deployable
        [1, 1, 2],  # Mostly agree: needs tuning
        [0, 0, 1],  # Mostly agree: reject
        [2, 1, 2],  # Partially agree
        [1, 2, 2],  # Partially agree
    ]
    kappa = compute_fleiss_kappa(ratings)
    check("Fleiss kappa in [-1, 1]", -1 <= kappa <= 1, f"kappa={kappa}")

    rates = compute_approval_rates(ratings, approval_threshold=1)
    check("Approval rates >= 0", all(v >= 0 for v in rates.values()))
    check("Approval rates <= 100", all(v <= 100 for v in rates.values()))

def test_ablation():
    print("\n--- Ablation ---")
    from pcap2rule.evaluation.ablation import ABLATION_CONFIGS
    check("8 ablation configs", len(ABLATION_CONFIGS) == 8)
    check("full config present", 'full' in ABLATION_CONFIGS)
    check("full all True", all(v is True for v in ABLATION_CONFIGS['full'].values()
                               if isinstance(v, bool)))

def test_robustness():
    print("\n--- Robustness ---")
    from pcap2rule.evaluation.robustness import (
        perturb_payload_obfuscation, perturb_timing,
        evaluate_robustness,
    )
    text = "GET /search?q=UNION SELECT * FROM users WHERE 1=1 HTTP/1.1"
    perturbed = perturb_payload_obfuscation(text, seed=42)
    check("Payload perturbation produces output", len(perturbed) > 0)

    flow = np.random.randn(4, 38)
    t_perturbed = perturb_timing(flow, seed=42)
    check("Timing perturbation shape", t_perturbed.shape == (4, 38))

    results = evaluate_robustness(87.0, {'PO': 72.0, 'TP': 80.0, 'FF': 76.0, 'PM': 69.0})
    check("Robustness eval returns 4 entries", len(results) == 4)

def test_protocol_parser():
    print("\n--- Protocol Parser ---")
    from pcap2rule.pcap_processor.protocol_parser import (
        parse_http_request_line, parse_http_headers, normalize_payload,
        extract_malicious_tokens,
    )
    method, uri, version = parse_http_request_line("GET /index.html HTTP/1.1")
    check("HTTP request method", method == 'GET')
    check("HTTP request URI", uri == '/index.html')

    headers = parse_http_headers("GET / HTTP/1.1\r\nHost: example.com\r\nUser-Agent: Mozilla\r\n")
    check("HTTP headers parse", headers.get('Host') == 'example.com')

    tokens = extract_malicious_tokens("SELECT * FROM users WHERE 1=1 OR 'a'='a'")
    check("Malicious token extraction", len(tokens) > 0)

def test_sqli_patterns():
    print("\n--- SQLi Patterns ---")
    from pcap2rule.generalization.sqli_patterns import (
        SQLI_PATTERNS, get_default_sqli_pcre, expand_sqli_patterns,
    )
    check("SQLi patterns dict", len(SQLI_PATTERNS) >= 4)
    pcre = get_default_sqli_pcre()
    check("Default SQLi PCRE non-empty", len(pcre) > 0)
    expanded = expand_sqli_patterns("UNION SELECT * FROM users")
    check("Expand SQLi patterns", isinstance(expanded, list))

def test_feedback_formatter():
    print("\n--- Feedback Formatter ---")
    from pcap2rule.validators.feedback_formatter import (
        format_syntax_feedback, format_semantic_feedback,
        combine_feedback,
    )
    fb = format_syntax_feedback(["Unmatched quotes", "Missing closing quote in content"])
    check("Syntax feedback non-empty", len(fb) > 0)

    fb2 = format_semantic_feedback(0.45, 0.6, ["Content mismatch detected"])
    check("Semantic feedback contains score", '0.45' in fb2 or '45' in fb2.lower() or 'score' in fb2.lower())

    combined = combine_feedback({
        'syntax': "syntax err",
        'semantic': "semantic err",
        'performance': "performance err",
    })
    check("Combined feedback non-empty", len(combined) > 0)

def test_visualization():
    print("\n--- Visualization ---")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    with tempfile.TemporaryDirectory() as d:
        # Sensitivity curves
        from pcap2rule.visualization.sensitivity_curves import (
            SensitivityResult, plot_sensitivity_curves,
        )
        rng = np.random.RandomState(42)
        results = {}
        for name in ['k (RAG)', 'alpha', 'tau']:
            results[name] = SensitivityResult(
                param_name=name, param_values=[1, 3, 5],
                sv_mean=[90, 91, 92], sv_std=[1, 1, 1],
                dc_mean=[80, 85, 87], dc_std=[2, 2, 2],
                vc_mean=[65, 70, 73], vc_std=[3, 3, 3],
                fpr_mean=[3, 3.5, 4], fpr_std=[0.5, 0.5, 0.5],
            )
        fig = plot_sensitivity_curves(results)
        path = os.path.join(d, 'sensitivity.png')
        fig.savefig(path, dpi=72)
        plt.close(fig)
        check("Sensitivity figure saved", os.path.exists(path))

        # Confusion matrix
        from pcap2rule.visualization.confusion_matrix import (
            plot_confusion_matrix, compute_confusion_matrix_data,
        )
        labels = ['Benign', 'SQLi', 'XSS']
        y_true = labels * 10
        y_pred = labels * 10
        cm, lbs = compute_confusion_matrix_data(y_true, y_pred)
        fig = plot_confusion_matrix(cm, lbs)
        path = os.path.join(d, 'cm.png')
        fig.savefig(path, dpi=72)
        plt.close(fig)
        check("Confusion matrix saved", os.path.exists(path))


if __name__ == '__main__':
    print("=" * 60)
    print("Pcap2Rule Smoke Tests")
    print("=" * 60)

    tests = [
        test_config, test_seed, test_suricata_utils, test_pcap_utils,
        test_logging, test_metrics, test_statistical, test_human_eval,
        test_ablation, test_robustness, test_protocol_parser,
        test_sqli_patterns, test_feedback_formatter, test_visualization,
    ]
    for t in tests:
        try:
            t()
        except Exception as e:
            import traceback
            traceback.print_exc()
            TESTS_FAILED += 1
            print(f"  [FAIL] {t.__name__}: {e}")

    print(f"\n{'=' * 60}")
    print(f"Results: {TESTS_PASSED} passed, {TESTS_FAILED} failed")
    sys.exit(0 if TESTS_FAILED == 0 else 1)
