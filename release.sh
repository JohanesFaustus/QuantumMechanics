#!/bin/bash
TAG="v$(date +%Y.%m.%d)A5"
PDF="Main.pdf"

gh release create "$TAG" "$PDF" \
  --title "Quantum Mechanics A5 $TAG" \
  --notes "Auto-generated release on $(date)"
