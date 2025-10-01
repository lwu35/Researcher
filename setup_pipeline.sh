#!/bin/bash

echo "=========================================="
echo "🚀 Automated Research Pipeline Setup"
echo "=========================================="
echo ""

# Check if we're in the Researcher directory
if [ ! -f "automated_research_pipeline.py" ]; then
    echo "❌ Error: Please run this script from the Researcher directory"
    exit 1
fi

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements_pipeline.txt

if [ $? -ne 0 ]; then
    echo "⚠️  Some packages failed to install. Trying individually..."
    pip install ai_researcher
    pip install openai
    pip install scholarly
    pip install requests
    pip install bibtexparser
fi

echo ""
echo "✅ Dependencies installed!"
echo ""

# Check for OpenAI API key
echo "🔑 Checking for OpenAI API key..."
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  OpenAI API key not found in environment"
    echo ""
    echo "Please set your OpenAI API key:"
    echo "  export OPENAI_API_KEY='your-key-here'"
    echo ""
    echo "Or add to your ~/.bashrc or ~/.zshrc:"
    echo "  echo 'export OPENAI_API_KEY=\"your-key-here\"' >> ~/.bashrc"
    echo ""
    echo "Get your key from: https://platform.openai.com/api-keys"
    echo ""
else
    echo "✅ OpenAI API key found"
fi

# Check GPU availability
echo ""
echo "🖥️  Checking GPU availability..."
python3 -c "import torch; print('✅ GPU available:', torch.cuda.is_available()); print('   GPU count:', torch.cuda.device_count()) if torch.cuda.is_available() else print('⚠️  No GPU detected - generation will be very slow')" 2>/dev/null

if [ $? -ne 0 ]; then
    echo "⚠️  PyTorch not installed or GPU check failed"
fi

# Create output directory
echo ""
echo "📁 Creating output directory..."
mkdir -p output
echo "✅ Output directory created"

# Check if research topic file exists
echo ""
echo "📝 Checking research topic file..."
if [ ! -f "my_research_topic.md" ]; then
    echo "⚠️  my_research_topic.md not found"
    echo "   Please edit my_research_topic.md with your research idea"
else
    echo "✅ Research topic file found"
fi

echo ""
echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Set OpenAI API key (if not done):"
echo "     export OPENAI_API_KEY='your-key-here'"
echo ""
echo "  2. Edit your research topic:"
echo "     nano my_research_topic.md"
echo ""
echo "  3. Run the pipeline:"
echo "     python automated_research_pipeline.py"
echo ""
echo "  4. Review results in output/ directory"
echo ""
echo "For detailed instructions, see PIPELINE_README.md"
echo "" 