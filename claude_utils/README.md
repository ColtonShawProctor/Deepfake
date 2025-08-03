# Claude API Diagnostic Tools

This directory contains diagnostic scripts for troubleshooting Claude API key issues across different platforms.

## Scripts

### 1. `claude_api_diagnostic.py`
**General Claude API Key Diagnostic**
- Checks environment variables
- Validates API key format
- Tests API key with actual API call
- Checks file configurations (.env, .claude.json, shell profiles)
- Provides fix suggestions

**Usage:**
```bash
python3 claude_api_diagnostic.py
```

### 2. `claude_code_diagnostic.py`
**VS Code Extension Diagnostic**
- Checks VS Code installation
- Validates VS Code settings
- Checks extension installation
- Diagnoses Claude Code extension issues

**Usage:**
```bash
python3 claude_code_diagnostic.py
```

### 3. `cursor_claude_diagnostic.py`
**Cursor IDE Diagnostic**
- Checks Cursor installation
- Validates Cursor settings
- Diagnoses Claude Code in Cursor issues
- Provides Cursor-specific fix suggestions

**Usage:**
```bash
python3 cursor_claude_diagnostic.py
```

### 4. `cursor_settings.json`
**Cursor Settings Template**
- Template for Cursor settings with API key
- Can be copied to `~/Library/Application Support/Cursor/User/settings.json`

## Requirements

- Python 3.6+
- `requests` library (for API testing)
- `pathlib` (built-in)

## Installation

```bash
# Install required packages
pip install requests

# Make scripts executable
chmod +x *.py
```

## Quick Fix Commands

### For Cursor:
```bash
# Add API key to Cursor settings
cp cursor_settings.json ~/Library/Application\ Support/Cursor/User/settings.json

# Restart Cursor after making changes
```

### For Environment Variables:
```bash
# Set API key in environment
export ANTHROPIC_API_KEY="your-key-here"

# Add to shell profile permanently
echo 'export ANTHROPIC_API_KEY="your-key-here"' >> ~/.zshrc
```

## Troubleshooting

1. **Run the appropriate diagnostic script**
2. **Follow the fix suggestions provided**
3. **Restart the application after making changes**
4. **Run the diagnostic again to verify fixes**

## Notes

- These scripts are designed to be non-destructive
- They create backups when modifying settings
- All API tests use minimal requests to avoid charges
- Scripts provide detailed error messages and fix suggestions 