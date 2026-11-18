# Customer Churn Prediction

A Flask application for individual and batch churn predictions using the project's saved Random Forest model. It includes account roles, per-user prediction history, administration tools, login auditing, CSV export, and PDF reports.

## Requirements

- Use the project’s Python 3.14 virtual environment from `.venv` for all local runs. The app and saved model artifacts are regenerated under the current interpreter to avoid scikit-learn pickle compatibility warnings.
- `cleaned_dataset.csv`, `churn_model.pkl`, and `encoders.pkl` must remain in the project root. The optional `label_encoder.pkl` is not used by the application.
- A legacy validation inconsistency remains in the old prediction flow: `predict_customer()` rejects `Tenure > 120`, even though some real rows in `cleaned_dataset.csv` contain `Tenure = 122`. This will be replaced by schema-driven validation in the multi-company upgrade.

## Setup and Run (Windows PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
$secret = py -c "import secrets; print(secrets.token_hex(32))"
$adminCode = "0808"
@("SECRET_KEY=$secret", "ADMIN_REGISTRATION_CODE=$adminCode") | Set-Content .env
flask --app app run --debug
```

Open `http://127.0.0.1:5000`. Choose **Sign in as User** or **Sign in as Admin** on the login page, and select the corresponding account type when registering. Admin registration requires the server-side `ADMIN_REGISTRATION_CODE` (development default: `0808`). Change it to a private value before deploying. Keep `.env` private. To run with `python app.py` instead of the Flask CLI, set both values in the current PowerShell session first:

```powershell
$env:SECRET_KEY = (py -c "import secrets; print(secrets.token_hex(32))")
$env:ADMIN_REGISTRATION_CODE = "0808"
python app.py
```

The SQLite database is created automatically at `instance/churn.sqlite`. New accounts are regular Users by default; only the separate admin registration flow with a valid secret code creates an administrator. Existing accounts and their roles are preserved; administrators can also manage roles from **Admin → Users**. The historical database migration script can be run when needed:
```powershell
python scripts/migrate_v3.py
```

## Usage

- **Registration:** Registration validates names, email addresses, and password strength. Accounts can sign in immediately after creation.
- **Sign In:** User and Admin sign-in modes are separate. A login only succeeds when the account role matches the selected mode.
- **Workspace Predictions:** Select a company workspace from the dashboard. Predictions use only that company's active dataset and trained model; numeric values must stay within observed dataset ranges and categories must match dataset values.
- **Train Models:** Before predicting for the first time, train active company datasets with `python scripts/train_company_models.py` from the activated project environment.
- **Bulk Prediction:** Upload a CSV using the company-specific template on the Bulk Prediction page (up to 500 rows). Results are directly downloaded as CSV.
- **Prediction History & Export:** Browse, filter, and export prediction history as CSV. Individual prediction PDFs can be generated directly.
- **Admin Management:** Admins can manage accounts and roles, grant or revoke user access to active company workspaces from **Users → Manage access**, review workspace-wide predictions and audit logs, and inspect the model card. Users can select and predict only for companies granted to their account; administrators automatically see all active companies.
- **Risk Thresholds:** Prediction-risk bands are Low below 30%, Medium from 30% to below 60%, and High at 60% or higher.

## Tests

```powershell
pytest
```

The tests cover account registration, role-specific sign-in, login lockout/auditing, export generation, formula injection escaping, admin access controls, and prediction persistence.

## Project Structure

```text
ABCD/
├── app.py                         # Flask entry point
├── config.py                      # Environment and session configuration
├── churn_app/
│   ├── __init__.py                 # Application factory
│   ├── extensions.py               # SQLAlchemy, Flask-Login, CSRF
│   ├── forms.py                    # WTForms validation
│   ├── models.py                   # Current multi-company data model
│   ├── security.py                 # Admin authorization
│   ├── services/                  # Schema, data-quality, training, prediction logic
│   └── blueprints/
│       ├── auth/                   # Registration, login, logout
│       ├── main/                   # Dashboard, prediction, history, profile
│       └── admin/                  # Admin dashboard and company management
├── scripts/                       # Dataset generation and migration helpers
├── data/                          # Synthetic company datasets and metadata
├── templates/                     # Shared, user, admin, and error pages
├── static/                        # Styles and bulk CSV template
├── tests/                         # Pytest suite
├── legacy/                        # Archived single-company scripts kept for reference
├── cleaned_dataset.csv            # Legacy dataset retained for compatibility
├── churn_model.pkl                # Legacy saved model artifact
├── encoders.pkl                  # Legacy encoder artifact
├── README.md
├── requirements.txt
├── run_demo.ps1
└── .env.example
```

## Screenshots

Capture desktop and mobile screenshots into `docs/screenshots/` when preparing a report or demonstration:

- `register.png` — registration and password-strength meter
- `user-dashboard.png` — personal KPIs, charts, and prediction result
- `history.png` — filters, details dialog, and prediction table
- `admin-dashboard.png` — system KPIs and activity charts
- `admin-users.png` — user and role management