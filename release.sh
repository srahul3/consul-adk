#!/bin/bash
set -e

# prompt if all the changes are committed. It is ok if they are not pushed
if [[ -n $(git status --porcelain) ]]; then
  read -p "You have uncommitted changes. Do you want to continue? (y/n) " yn
  case $yn in
    [Yy]* ) echo "Continuing...";;
    [Nn]* ) echo "Please commit your changes before releasing."; exit 1;;
    * ) echo "Please answer yes or no."; exit 1;;
  esac
fi

make setup-dev
source .venv/bin/activate

# Prompt for new version
read -p "Enter new semver version (e.g. 1.2.3): " NEW_VERSION

# Run tests
make test

# Update version in pyproject.toml
sed -i '' "s/^version = \".*\"/version = \"$NEW_VERSION\"/" pyproject.toml

git add pyproject.toml
git commit -m "chore: release v$NEW_VERSION"
git push origin main

git tag "$NEW_VERSION"
git push origin "$NEW_VERSION"

echo "Release v$NEW_VERSION created and pushed to main."

