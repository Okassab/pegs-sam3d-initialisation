#!/usr/bin/env bash
# Re-run once on every new RunPod pod to restore git + GitHub SSH setup.
set -e
git config --global user.name  "Omar Kassab"
git config --global user.email "okassab2003@gmail.com"
git config --global init.defaultBranch main
git config --global core.sshCommand "ssh -i /workspace/.ssh/github_ed25519 -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new"
echo "git configured, using key /workspace/.ssh/github_ed25519"
