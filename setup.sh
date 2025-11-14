#!/bin/bash

echo "Setting up FAQ Support Chatbot..."

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "Error: Python 3.8+ is required. Found version $python_version"
    exit 1
fi
echo "Python version $python_version - OK"

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create .env file from .env.example
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo ".env file created successfully"
else
    echo ".env file already exists, skipping..."
fi

echo ""
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Edit .env file and add your API key:"
echo "   - For OpenRouter: Get key from https://openrouter.ai/keys"
echo "   - For OpenAI: Get key from https://platform.openai.com/api-keys"
echo "3. Update OPENAI_BASE_URL in .env if using OpenAI instead of OpenRouter"
echo "4. Build the index: python src/build_index.py"
echo "5. Run queries: python src/query.py \"your question here\""
