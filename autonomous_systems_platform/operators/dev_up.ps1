param()
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r services\backend\requirements.txt
$env:FLASK_APP = "services\backend\app.py"
python services\backend\app.py
