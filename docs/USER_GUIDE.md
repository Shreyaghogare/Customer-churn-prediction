# User guide

## Initial setup

1. Open a terminal in the project root.
2. Create a Python 3.11 or 3.12 virtual environment.
3. Install the requirements with `pip install -r requirements.txt`.
4. If migrating from an existing database, run `python scripts/migrate_v3.py`.
5. Generate the demo datasets with `python scripts/generate_datasets.py`.
6. Seed the company catalog with `python scripts/seed_companies.py`.
7. Configure `.env` with `SECRET_KEY` and `ADMIN_REGISTRATION_CODE` (development default: `0808`; replace before deployment).
8. Start the app with `flask --app app run --debug`.

## Account Registration

- Choose **Create Account as User** or **Create Account as Admin** at `/register`. Admin accounts are created only after the server verifies the secret admin code.
- Choose the matching **Sign in as User** or **Sign in as Admin** mode at `/login`. The account role must match the selected mode.

## Demo companies

- TeleConnect
- SecureBank
- ShopEase
- StreamHub
- SafeGuard

These company records are stored in the database and each one references its own dataset file.

## How the platform works

- Select a company from the app when the company-aware UI is available.
- Upload or generate a CSV that follows the company schema.
- Train or retrain a model from the dataset.
- Run prediction calls through the schema-aware service layer.
- Export prediction history as CSV.
- Bulk predict and download the scored results.
- Administrators assign user workspace permissions in **Admin → Users → Manage access**. A user can select and make predictions only in an active company workspace assigned to that account. Administrators automatically have access to every active company.

## Notes

The project uses a service-oriented design to keep business logic separate from the route handlers, which makes it easier to extend to more companies and new datasets.
