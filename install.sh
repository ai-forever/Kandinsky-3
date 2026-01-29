set -e

# Step 1: Check Python version
echo "[1/4] Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"
echo ""

# Step 2: Install dependencies from requirements.txt
echo "[2/4] Installing dependencies from requirements.txt..."
if [ -f "requirements.txt" ]; then
    pip install --upgrade pip setuptools wheel
    pip install -r requirements.txt
    echo "Dependencies installed successfully"
else
    echo "requirements.txt not found"
    exit 1
fi
echo ""

# Step 3: Install the package using setup.py
echo "[3/4] Installing kandinsky3 package using setup.py..."
if [ -f "setup.py" ]; then
    pip install -e .
    echo "Package installed successfully"
else
    echo "setup.py not found"
    exit 1
fi
echo ""

# Step 4: Test import
echo "[4/4] Testing kandinsky3 import..."
python3 test_import.py
if [ $? -eq 0 ]; then
    echo ""
    echo "Installation completed successfully"
    exit 0
else
    echo ""
    echo "Installation failed during import test"
    exit 1
fi
