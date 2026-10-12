#!/usr/bin/env python
"""Convert LabelMe filled polygons to binary masks using explicit labels and patient groups."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from icg_pipeline.data.labelme import convert_labelme_dataset, make_manifest_template


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Private source directory of LabelMe JSONs")
    parser.add_argument("--template", help="Write an unassigned annotation,split,group_id CSV and exit")
    parser.add_argument("--manifest", help="Completed CSV; group_id identifies the patient, across all procedures")
    parser.add_argument("--output", help="New private dataset directory, separate from input")
    parser.add_argument("--foreground-labels", nargs="+", help="Explicit labels to union into foreground")
    parser.add_argument("--background-labels", nargs="*", default=[], help="Other labels to treat as background")
    parser.add_argument("--exclude-labels", nargs="*", default=[], help="Exclude the entire sample if it contains any of these labels")
    parser.add_argument("--train-only", action="store_true", help="Single-patient diagnostic: only train rows, no validation/test split")
    args = parser.parse_args()
    if args.template:
        if args.manifest or args.output or args.foreground_labels or args.background_labels or args.train_only:
            parser.error("--template accepts only --input and optional --exclude-labels")
        print(json.dumps({"annotations": make_manifest_template(args.input, args.template, args.exclude_labels), "splits_assigned": False}))
    else:
        if not args.manifest or not args.output or not args.foreground_labels:
            parser.error("Conversion requires --manifest, --output and --foreground-labels")
        print(json.dumps(convert_labelme_dataset(args.input, args.output, args.manifest,
                         args.foreground_labels, args.background_labels,
                         train_only=args.train_only, exclude_labels=args.exclude_labels), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
