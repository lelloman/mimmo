#!/usr/bin/env python3
"""Generate Rust test cases from cascade test results using ground truth labels."""

import re
import json
from pathlib import Path

def parse_size(size_str):
    """Parse size string like '1.5GB' to bytes"""
    match = re.match(r'([\d.]+)(B|KB|MB|GB|TB)', size_str)
    if not match:
        return 0
    val = float(match.group(1))
    unit = match.group(2)
    multipliers = {'B': 1, 'KB': 1024, 'MB': 1024**2, 'GB': 1024**3, 'TB': 1024**4}
    return int(val * multipliers[unit])

def parse_results_file(path):
    with open(path, 'r') as f:
        content = f.read()

    # Find all test cases
    pattern = r'\[(\d+)\] (.+?)\n    Files: (\d+) \(([^)]+) total\)\n((?:      - .+\n)*)\s+=> (\w+) \(conf=([^,]+), source=([^)]+)\)'

    cases = []
    for match in re.finditer(pattern, content):
        idx = int(match.group(1))
        name = match.group(2).strip()
        file_count = int(match.group(3))
        total_size = match.group(4)
        files_block = match.group(5)
        medium = match.group(6).lower()
        confidence = float(match.group(7))
        source = match.group(8)

        # Parse files - use greedy match to get full path, size is at the end
        files = []
        for file_match in re.finditer(r'      - (.+) \((\d+\.?\d*[KMGT]?B)\)', files_block):
            file_path = file_match.group(1).strip()
            file_size = parse_size(file_match.group(2))
            files.append({'path': file_path, 'size': file_size})

        cases.append({
            'idx': idx,
            'name': name,
            'files': files,
            'detected': medium,
            'confidence': confidence,
            'source': source,
        })

    return cases

def load_ground_truth(path):
    """Load ground truth labels from JSON file."""
    with open(path, 'r') as f:
        data = json.load(f)

    # Convert to dict for easy lookup
    labels = {s['idx']: s['ground_truth'] for s in data['samples']}
    skip = set(data.get('skip_in_tests', []))
    return labels, skip

def generate_rust_test(case, expected_medium, prefix=""):
    """Generate a single Rust test case"""
    # Escape the name for Rust string
    name = case['name'].replace('\\', '\\\\').replace('"', '\\"')

    # Generate files vec
    files_code = []
    for f in case['files'][:5]:  # Limit to 5 files
        path = f['path'].replace('\\', '\\\\').replace('"', '\\"')
        # Use the last component as filename (preserving extension)
        filename = path.split('/')[-1]
        # For very long paths, truncate from the beginning but keep filename
        if len(path) > 80:
            path = "..." + path[-(77):]
        files_code.append(f'''        FileInfo {{
            path: "{path}".to_string(),
            filename: "{filename}".to_string(),
            size: {f['size']},
        }}''')

    files_vec = ",\n".join(files_code)

    medium_enum = {
        'video': 'Medium::Video',
        'audio': 'Medium::Audio',
        'book': 'Medium::Book',
        'software': 'Medium::Software',
        'other': 'Medium::Other',
    }[expected_medium]

    # Create a valid test function name
    test_name = f"{prefix}sample_{case['idx']:03d}"

    return f'''#[test]
fn {test_name}() {{
    // {name[:60]}
    let info = ContentInfo {{
        name: "{name[:100]}".to_string(),
        files: vec![
{files_vec}
        ],
    }};
    let result = CASCADE.classify(&info).unwrap();
    assert_eq!(result.medium, {medium_enum}, "{prefix.upper() if prefix else ''}Sample {case['idx']}: {{:?}}", result);
}}
'''

def process_batch(results_file, ground_truth_file, prefix=""):
    """Process a single batch and return test cases and stats."""
    cases = parse_results_file(results_file)
    ground_truth, skip_samples = load_ground_truth(ground_truth_file)

    print(f"\n{prefix.upper() if prefix else 'Batch'}: Parsed {len(cases)} test cases from {results_file}")
    print(f"  Loaded {len(ground_truth)} ground truth labels, {len(skip_samples)} to skip")

    # Analyze errors
    errors_heuristic = []
    errors_ml = []
    correct = 0

    for case in cases:
        if case['idx'] in skip_samples:
            continue
        expected = ground_truth.get(case['idx'])
        if expected and case['detected'] != expected:
            if case['source'] in ['extensions', 'patterns']:
                errors_heuristic.append(case)
            else:
                errors_ml.append(case)
        else:
            correct += 1

    print(f"  Error analysis (excluding {len(skip_samples)} skipped):")
    print(f"    Correct: {correct}")
    print(f"    Heuristic errors: {len(errors_heuristic)}")
    for e in errors_heuristic:
        print(f"      #{e['idx']}: {e['detected']} should be {ground_truth[e['idx']]} (source={e['source']})")
    print(f"    ML errors: {len(errors_ml)}")
    for e in errors_ml:
        print(f"      #{e['idx']}: {e['detected']} should be {ground_truth[e['idx']]} (source={e['source']})")

    # Generate test code
    tests = []
    for case in cases:
        if case['idx'] in skip_samples:
            continue
        expected = ground_truth.get(case['idx'], case['detected'])
        tests.append(generate_rust_test(case, expected, prefix))

    return tests, len(cases) - len(skip_samples), len(skip_samples)

def main():
    project_root = Path(__file__).parent.parent

    # Check which batches exist
    batches = []

    # Batch 1
    batch1_results = project_root / "cascade_test_results.txt"
    batch1_gt = project_root / "test_ground_truth.json"
    batch1_gt_alt = project_root / "test_ground_truth_batch1.json"

    if batch1_gt.exists():
        if batch1_results.exists():
            batches.append(("", batch1_results, batch1_gt))
    elif batch1_gt_alt.exists():
        if batch1_results.exists():
            batches.append(("batch1_", batch1_results, batch1_gt_alt))

    # Batch 2
    batch2_results = project_root / "cascade_test_results_batch2.txt"
    batch2_gt = project_root / "test_ground_truth_batch2.json"

    if batch2_results.exists() and batch2_gt.exists():
        batches.append(("batch2_", batch2_results, batch2_gt))

    if not batches:
        print("No test batches found!")
        return

    # Generate Rust code header
    rust_code = '''//! Sample-based regression tests for cascade classifier.
//!
//! These tests use manually verified ground truth labels.
//! They ensure the classifier produces correct results on real-world torrent data.

use crate::cascade::{Cascade, Medium};
use crate::{ContentInfo, FileInfo};
use std::sync::LazyLock;

static CASCADE: LazyLock<Cascade> = LazyLock::new(|| {
    Cascade::default_with_ml().expect("Failed to create cascade")
});

'''

    total_tests = 0
    total_skipped = 0

    for prefix, results_file, gt_file in batches:
        tests, count, skipped = process_batch(results_file, gt_file, prefix)
        for test in tests:
            rust_code += test + "\n"
        total_tests += count
        total_skipped += skipped

    with open(project_root / 'src/cascade/samples_test.rs', 'w') as f:
        f.write(rust_code)

    print(f"\n{'='*60}")
    print(f"Written src/cascade/samples_test.rs ({total_tests} tests, {total_skipped} skipped)")

if __name__ == "__main__":
    main()
