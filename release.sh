#!/bin/bash
set -e

make clean

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

# read the current version from pyproject.toml
CURRENT_VERSION=$(grep -oP '(?<=^version = ")[^"]*' pyproject.toml)
echo "Current version is $CURRENT_VERSION"
# Prompt for new version
read -p "Enter new semver version (e.g. 1.2.3): " NEW_VERSION

# Validate semver format and also confirm with user
if [[ ! $NEW_VERSION =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "Error: Version must be in semver format (e.g. 1.2.3)"
  exit 1
fi
read -p "You entered version $NEW_VERSION. Is this correct? (y/n) " yn

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

# create a new release on GitHub
gh release create "$NEW_VERSION" --title "v$NEW_VERSION" --notes "Release v$NEW_VERSION"

