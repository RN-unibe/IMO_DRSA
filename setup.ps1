# ./setup.ps1

python3 -m venv .venv

.\.venv\Scripts\Activate.ps1

python -m ensurepip --upgrade
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
