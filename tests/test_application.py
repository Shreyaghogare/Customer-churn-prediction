import io
from datetime import timedelta

import pytest
from werkzeug.security import generate_password_hash

from churn_app.blueprints.main import routes as main_routes
from churn_app.extensions import db
from churn_app.models import (
    AdminActivityLog,
    Company,
    CompanyDataset,
    LoginLog,
    Prediction,
    TrainedModel,
    User,
    UserCompanyAccess,
    utc_now,
)
from churn_app.services import build_data_quality_report, infer_schema, train_company_model


def add_user(app, *, name="Test User", email="user@example.com", role="user"):
    with app.app_context():
        user = User(
            name=name,
            email=email,
            password_hash=generate_password_hash("StrongPass1!"),
            role=role,
        )
        db.session.add(user)
        db.session.commit()
        return user.id


def authenticate(client, user_id):
    with client.session_transaction() as session:
        session["_user_id"] = str(user_id)
        session["_fresh"] = True


def test_registration_requires_explicit_admin_mode_and_secret_code(app, client):
    user_registration_page = client.get("/register")
    admin_registration_page = client.get("/register?mode=admin")
    assert b'id="admin_code"' not in user_registration_page.data
    assert b'id="admin_code"' in admin_registration_page.data
    assert b"0808" not in admin_registration_page.data

    weak = client.post("/register", data={
        "name": "First Admin",
        "email": "FIRST@example.com",
        "password": "weakpass",
        "confirm_password": "weakpass",
    })
    assert weak.status_code == 200
    assert b"uppercase" in weak.data

    response = client.post("/register", data={
        "name": "First Admin",
        "email": "FIRST@example.com",
        "password": "StrongPass1!",
        "confirm_password": "StrongPass1!",
        "account_type": "admin",
        "admin_code": "wrong-code",
    })
    assert response.status_code == 200
    assert b"Invalid admin secret code. Please enter the correct code." in response.data
    with app.app_context():
        assert User.query.count() == 0

    response = client.post("/register", data={
        "name": "First User",
        "email": "FIRST@example.com",
        "password": "StrongPass1!",
        "confirm_password": "StrongPass1!",
    })
    assert response.status_code == 302
    with app.app_context():
        first = User.query.one()
        assert first.email == "first@example.com"
        assert first.role == "user"
        assert first.password_hash != "StrongPass1!"

    duplicate = client.post("/register", data={
        "name": "Duplicate User",
        "email": "FIRST@example.com",
        "password": "StrongPass1!",
        "confirm_password": "StrongPass1!",
    })
    assert duplicate.status_code == 200
    assert b"An account with this email already exists." in duplicate.data

    response = client.post("/register", data={
        "name": "Admin User",
        "email": "admin@example.com",
        "password": "StrongPass1!",
        "confirm_password": "StrongPass1!",
        "account_type": "admin",
        "admin_code": "0808",
    })
    assert response.status_code == 302
    assert "/login?mode=admin" in response.headers["Location"]
    with app.app_context():
        admin = User.query.filter_by(email="admin@example.com").one()
        assert admin.role == "admin"
        assert admin.password_hash != "StrongPass1!"
        assert User.query.count() == 2


def test_user_and_admin_sign_in_modes_enforce_roles_and_keep_admin_dashboard(app, client):
    user_sign_in_page = client.get("/login")
    admin_sign_in_page = client.get("/login?mode=admin")
    assert b"Sign in as User" in user_sign_in_page.data
    assert b"Sign in as Admin" in user_sign_in_page.data
    assert b"Sign In as Admin" in admin_sign_in_page.data

    user_id = add_user(app, email="user@example.com")
    admin_id = add_user(app, name="Admin User", email="admin@example.com", role="admin")

    user_client = app.test_client()
    user_admin_login = user_client.post("/login?mode=admin", data={
        "account_type": "admin",
        "email": "user@example.com",
        "password": "StrongPass1!",
    })
    assert user_admin_login.status_code == 200
    assert b"You are not authorized to access the admin account." in user_admin_login.data

    admin_client = app.test_client()
    admin_user_login = admin_client.post("/login?mode=user", data={
        "account_type": "user",
        "email": "admin@example.com",
        "password": "StrongPass1!",
    })
    assert admin_user_login.status_code == 200
    assert b"Admin accounts must use Admin Sign In." in admin_user_login.data

    user_login = user_client.post("/login?mode=user", data={
        "account_type": "user",
        "email": "user@example.com",
        "password": "StrongPass1!",
    })
    assert user_login.status_code == 302
    assert user_login.headers["Location"].endswith("/dashboard")
    assert user_client.get("/admin/dashboard").status_code == 403

    admin_login = admin_client.post("/login?mode=admin", data={
        "account_type": "admin",
        "email": "admin@example.com",
        "password": "StrongPass1!",
    })
    assert admin_login.status_code == 302
    assert admin_login.headers["Location"].endswith("/admin/dashboard")
    dashboard = admin_client.get("/admin/dashboard")
    assert dashboard.status_code == 200
    assert b"System overview" in dashboard.data
    with app.app_context():
        assert db.session.get(User, user_id).role == "user"
        assert db.session.get(User, admin_id).role == "admin"


def test_login_generic_failure_lockout_and_audit(app, client):
    user_id = add_user(app)
    for _attempt in range(5):
        response = client.post("/login", data={"email": "user@example.com", "password": "wrong"})
        assert response.status_code == 200
        assert b"Invalid email or password" in response.data
    with app.app_context():
        user = db.session.get(User, user_id)
        assert user.failed_attempts == 5
        assert user.locked_until is not None
        assert LoginLog.query.filter_by(success=False).count() == 5

    response = client.post("/login", data={"email": "missing@example.com", "password": "wrong"})
    assert response.status_code == 200
    assert b"Invalid email or password" in response.data
    with app.app_context():
        assert LoginLog.query.count() == 6
        assert LoginLog.query.filter_by(email_entered="missing@example.com", user_id=None).count() == 1


def test_existing_account_can_sign_in_without_reverification(app, client):
    add_user(app)

    response = client.post("/login", data={
        "email": "user@example.com",
        "password": "StrongPass1!",
    })

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/dashboard")


def test_admin_guard_and_self_protection(app, client):
    admin_id = add_user(app, name="Admin User", email="admin@example.com", role="admin")
    user_id = add_user(app, name="Regular User")
    authenticate(client, user_id)
    assert client.get("/admin/dashboard").status_code == 403

    authenticate(client, admin_id)
    response = client.post(f"/admin/users/{admin_id}/role", data={"role": "user"})
    assert response.status_code == 302
    response = client.post(f"/admin/users/{admin_id}/delete")
    assert response.status_code == 302
    with app.app_context():
        assert db.session.get(User, admin_id).role == "admin"
        assert db.session.get(User, admin_id) is not None
        assert AdminActivityLog.query.filter(AdminActivityLog.action.like("blocked self-%")).count() == 2


@pytest.mark.parametrize(
    ("probability", "risk"),
    [(0.2999, "Low Risk"), (0.3, "Medium Risk"), (0.5999, "Medium Risk"), (0.6, "High Risk")],
)
def test_prediction_validation_persistence_and_risk_thresholds(app, client, monkeypatch, tmp_path, probability, risk):
    import pandas as pd

    admin_id = add_user(app, name="Prediction Admin", email="prediction-admin@example.com", role="admin")
    user_id = add_user(app)
    data_path = tmp_path / "company.csv"
    pd.DataFrame({
        "CustomerID": ["C001", "C002", "C003"],
        "Age": [20, 35, 50],
        "Gender": ["Female", "Male", "Female"],
        "Churn": ["No", "Yes", "No"],
    }).to_csv(data_path, index=False)
    with app.app_context():
        company = Company(name="SchemaCo", slug="schemaco", industry="Telecom")
        db.session.add(company)
        db.session.flush()
        dataset = CompanyDataset(
            company_id=company.id,
            filename="company.csv",
            row_count=3,
            schema_json={
                "columns": ["CustomerID", "Age", "Gender", "Churn"],
                "target_column": "Churn",
                "positive_label": "Yes",
                "id_columns": ["CustomerID"],
            },
            data_path=str(data_path),
        )
        db.session.add(dataset)
        db.session.flush()
        model_record = TrainedModel(
            company_id=company.id,
            dataset_id=dataset.id,
            algorithm="StubModel",
            model_path="stub.joblib",
            is_active=True,
        )
        db.session.add_all([
            model_record,
        ])
        db.session.commit()
        company_id = company.id
        model_id = model_record.id

    authenticate(client, admin_id)
    access_response = client.post(
        f"/admin/users/{user_id}/access",
        data={"company_ids": str(company_id)},
    )
    assert access_response.status_code == 302

    with client.session_transaction() as session:
        session["active_company_id"] = str(company_id)
    authenticate(client, user_id)
    monkeypatch.setattr(main_routes, "predict_one", lambda _company, _values, _schema: {
        "prediction": "Yes",
        "probability": probability,
        "risk_level": risk,
        "risk_factors": ["Test risk factor"],
        "suggested_actions": ["Test retention action"],
    })

    prediction_page = client.get("/prediction")
    assert prediction_page.status_code == 200
    assert b'name="record_id"' in prediction_page.data
    assert b'name="feature_0"' not in prediction_page.data
    assert b"C001" in prediction_page.data

    invalid = client.post("/predict", data={"record_id": "C999"})
    assert invalid.status_code == 400
    assert b"Data is wrong" in invalid.data

    arbitrary_values = client.post("/predict", data={"record_id": "C001", "feature_0": "999"})
    assert arbitrary_values.status_code == 400
    assert b"submitted fields do not match this prediction mode" in arbitrary_values.data

    response = client.post("/predict", data={"record_id": "C002"})
    assert response.status_code == 200
    assert risk.encode() in response.data
    with app.app_context():
        saved = Prediction.query.one()
        assert saved.user_id == user_id
        assert saved.company_id == company_id
        assert saved.model_id == model_id
        assert saved.input_json == {"Age": 35, "Gender": "Male", "CustomerID": "C002"}
        assert saved.age == 35
        assert saved.probability == pytest.approx(probability)
        assert saved.risk_level == risk

    manual_page = client.get("/prediction?mode=manual")
    assert manual_page.status_code == 200
    assert b'name="feature_0"' in manual_page.data
    assert b'name="feature_1"' in manual_page.data

    unseen_combination = client.post("/predict", data={
        "mode": "manual", "feature_0": "35", "feature_1": "Female",
    })
    assert unseen_combination.status_code == 400
    assert b"must match one complete row" in unseen_combination.data

    manual_response = client.post("/predict", data={
        "mode": "manual", "feature_0": "35", "feature_1": "Male",
    })
    assert manual_response.status_code == 200
    with app.app_context():
        saved = Prediction.query.order_by(Prediction.id.desc()).first()
        assert saved.input_json == {"Age": 35, "Gender": "Male", "CustomerID": "C002"}


def test_csrf_rejects_unprotected_form_post():
    from churn_app import create_app

    app = create_app({
        "TESTING": True,
        "SECRET_KEY": "pytest-secret-key",
        "SQLALCHEMY_DATABASE_URI": "sqlite://",
        "WTF_CSRF_ENABLED": True,
    })
    response = app.test_client().post("/register", data={"name": "No Token"})
    assert response.status_code == 400


def test_dashboard_links_to_separate_feature_pages(app, client, tmp_path):
    import pandas as pd

    user_id = add_user(app)
    data_path = tmp_path / "dashboard-company.csv"
    pd.DataFrame({"Age": [20, 40], "Plan": ["Basic", "Plus"], "Churn": ["No", "Yes"]}).to_csv(data_path, index=False)
    with app.app_context():
        company = Company(name="DashboardCo", slug="dashboardco", industry="SaaS")
        db.session.add(company)
        db.session.flush()
        db.session.add_all([
            CompanyDataset(
                company_id=company.id,
                filename="dashboard-company.csv",
                row_count=2,
                schema_json={"columns": ["Age", "Plan", "Churn"], "target_column": "Churn"},
                data_path=str(data_path),
            ),
            UserCompanyAccess(user_id=user_id, company_id=company.id),
        ])
        db.session.commit()
    authenticate(client, user_id)

    dashboard = client.get("/dashboard")
    prediction_page = client.get("/prediction")

    assert dashboard.status_code == 200
    assert b"Prediction history" in dashboard.data
    assert b"Bulk CSV" in dashboard.data
    assert b"action=\"/predict\"" not in dashboard.data
    assert prediction_page.status_code == 200
    assert b"action=\"/predict?mode=id\"" in prediction_page.data


def test_prediction_requires_a_company_workspace(app, client):
    user_id = add_user(app)
    authenticate(client, user_id)

    response = client.post("/predict", data={})

    assert response.status_code == 400
    assert b"Choose a company workspace" in response.data


def test_history_renders_company_prediction_without_legacy_telco_fields(app, client):
    user_id = add_user(app)
    with app.app_context():
        company = Company(name="HistoryBank", slug="historybank", industry="Banking")
        db.session.add(company)
        db.session.flush()
        prediction = Prediction(
            user_id=user_id,
            company_id=company.id,
            prediction="No",
            probability=0.12,
            risk_level="Low Risk",
            risk_factors=[],
            suggested_actions=[],
            input_json={"CustomerID": "BK000001", "CreditScore": 650, "Geography": "France"},
        )
        db.session.add_all([prediction, UserCompanyAccess(user_id=user_id, company_id=company.id)])
        db.session.commit()
        company_id = company.id
    with client.session_transaction() as session:
        session["active_company_id"] = str(company_id)
    authenticate(client, user_id)

    response = client.get("/history")

    assert response.status_code == 200
    assert b"BK000001" in response.data
    assert b"CreditScore" in response.data
    with app.app_context():
        prediction_id = Prediction.query.one().id
    report = client.get(f"/prediction/{prediction_id}/report.pdf")
    assert report.status_code == 200
    assert report.mimetype == "application/pdf"


def test_formula_injection_cells_escaped():
    import io
    import openpyxl
    from churn_app.services.export_service import build_csv, build_xlsx, sanitize_cell

    assert sanitize_cell("=1+1") == "'=1+1"
    assert sanitize_cell("+cmd|' /C calc'!A0") == "'+cmd|' /C calc'!A0"
    assert sanitize_cell("-2+3") == "'-2+3"
    assert sanitize_cell("@SUM(1,2)") == "'@SUM(1,2)"
    assert sanitize_cell("Normal Text") == "Normal Text"
    assert sanitize_cell(42) == 42
    assert sanitize_cell(-3.14) == -3.14

    columns = ["Name", "Payload"]
    rows = [{"Name": "Test", "Payload": "=cmd|' /C calc'!A0"}]

    csv_data = build_csv(columns, rows)
    assert b"'=cmd|' /C calc'!A0" in csv_data

    xlsx_data = build_xlsx(columns, rows)
    wb = openpyxl.load_workbook(io.BytesIO(xlsx_data))
    assert wb.active["B2"].value == "'=cmd|' /C calc'!A0"


def test_prediction_handles_missing_model_probability(app, client, monkeypatch, tmp_path):
    import pandas as pd

    user_id = add_user(app)
    data_path = tmp_path / "probability-company.csv"
    pd.DataFrame({
        "CustomerID": ["C001", "C002"],
        "Age": [30, 40],
        "Gender": ["Female", "Male"],
        "Churn": ["No", "Yes"],
    }).to_csv(data_path, index=False)
    with app.app_context():
        company = Company(name="ProbabilityCo", slug="probabilityco", industry="Retail")
        db.session.add(company)
        db.session.flush()
        dataset = CompanyDataset(
            company_id=company.id,
            filename="probability-company.csv",
            row_count=2,
            schema_json={"columns": ["CustomerID", "Age", "Gender", "Churn"], "target_column": "Churn"},
            data_path=str(data_path),
        )
        db.session.add(dataset)
        db.session.flush()
        db.session.add_all([
            TrainedModel(company_id=company.id, dataset_id=dataset.id, model_path="unused.joblib", is_active=True),
            UserCompanyAccess(user_id=user_id, company_id=company.id),
        ])
        db.session.commit()
        company_id = company.id
    with client.session_transaction() as session:
        session["active_company_id"] = str(company_id)
    authenticate(client, user_id)
    monkeypatch.setattr(main_routes, "predict_one", lambda *_args: {
        "prediction": "No",
        "probability": None,
        "risk_level": "Low Risk",
        "risk_factors": [],
        "suggested_actions": [],
    })

    response = client.post("/predict", data={"record_id": "C001"})

    assert response.status_code == 400
    assert b"did not return a valid probability" in response.data
    with app.app_context():
        assert Prediction.query.count() == 0


def test_admin_dashboard_has_shortcuts_to_every_admin_module(app, client):
    admin_id = add_user(app, role="admin")
    authenticate(client, admin_id)

    response = client.get("/dashboard")

    assert response.status_code == 200
    for path in (
        "/admin/dashboard",
        "/admin/companies",
        "/admin/users",
        "/admin/predictions",
        "/admin/login-logs",
        "/admin/activity-logs",
        "/admin/model-info",
        "/admin/company-comparison",
    ):
        assert path.encode() in response.data


def test_company_comparison_is_admin_only(app, client):
    user_id = add_user(app)
    authenticate(client, user_id)
    assert client.get("/admin/company-comparison").status_code == 403

    admin_id = add_user(app, name="Workspace Admin", email="admin@example.com", role="admin")
    authenticate(client, admin_id)
    response = client.get("/admin/company-comparison")
    assert response.status_code == 200
    assert b"Company comparison" in response.data


def test_admin_can_grant_and_revoke_company_access_for_users(app, client):
    admin_id = add_user(app, name="Access Admin", email="access-admin@example.com", role="admin")
    user_id = add_user(app, name="Assigned User")
    with app.app_context():
        company = Company(name="AssignedCo", slug="assignedco", industry="Retail")
        inactive_company = Company(
            name="InactiveCo", slug="inactiveco", industry="Retail", is_active=False
        )
        db.session.add_all([company, inactive_company])
        db.session.commit()
        company_id = company.id
        inactive_company_id = inactive_company.id

    authenticate(client, admin_id)
    page = client.get(f"/admin/users/{user_id}/access")
    assert page.status_code == 200
    assert b"AssignedCo" in page.data
    assert b"InactiveCo" not in page.data

    invalid = client.post(
        f"/admin/users/{user_id}/access",
        data={"company_ids": str(inactive_company_id)},
    )
    assert invalid.status_code == 200
    assert b"Invalid company selection" in invalid.data
    with app.app_context():
        assert UserCompanyAccess.query.filter_by(user_id=user_id).count() == 0

    response = client.post(
        f"/admin/users/{user_id}/access",
        data={"company_ids": [str(company_id)]},
    )
    assert response.status_code == 302
    with app.app_context():
        assert UserCompanyAccess.query.filter_by(
            user_id=user_id, company_id=company_id
        ).count() == 1
        assert AdminActivityLog.query.filter_by(action="granted company access").count() == 1

    authenticate(client, user_id)
    assert client.post(
        "/company/select",
        data={"company_id": str(company_id)},
    ).status_code == 302
    assert client.post(
        "/company/select",
        data={"company_id": str(inactive_company_id)},
    ).status_code == 403

    authenticate(client, admin_id)
    client.post(f"/admin/users/{user_id}/access", data={})
    with app.app_context():
        assert UserCompanyAccess.query.filter_by(user_id=user_id).count() == 0
        assert AdminActivityLog.query.filter_by(action="revoked company access").count() == 1

    authenticate(client, user_id)
    assert client.post(
        "/company/select",
        data={"company_id": str(company_id)},
    ).status_code == 403


def test_users_cannot_manage_company_access(app, client):
    user_id = add_user(app)
    target_id = add_user(app, name="Another User", email="another@example.com")
    authenticate(client, user_id)

    assert client.get(f"/admin/users/{target_id}/access").status_code == 403


def test_company_dashboard_card_opens_selected_workspace(app, client):
    user_id = add_user(app)
    with app.app_context():
        company = Company(name="Northstar", slug="northstar", industry="Telecom")
        db.session.add(company)
        db.session.flush()
        db.session.add(UserCompanyAccess(user_id=user_id, company_id=company.id))
        db.session.commit()
        company_id = company.id
    authenticate(client, user_id)

    dashboard = client.get("/dashboard")
    response = client.post("/company/select", data={
        "company_id": str(company_id),
        "destination": "prediction",
    })

    assert b"Northstar" in dashboard.data
    assert b"Open workspace" in dashboard.data
    assert response.status_code == 302
    assert response.headers["Location"].endswith("/prediction")


def test_company_models_can_store_dataset_and_model_metadata(app):
    with app.app_context():
        admin = User(
            name="Admin",
            email="admin@example.com",
            password_hash="hash",
            role="admin",
        )
        db.session.add(admin)
        db.session.commit()

        company = Company(
            name="TeleConnect",
            slug="teleconnect",
            industry="Telecom",
            description="Customer churn platform",
            logo_color="#123456",
            created_by=admin.id,
            is_active=True,
        )
        db.session.add(company)
        db.session.commit()

        dataset = CompanyDataset(
            company_id=company.id,
            filename="teleconnect.csv",
            row_count=250,
            uploaded_by=admin.id,
            is_active=True,
            schema_json={"columns": ["Age", "Gender"], "target_column": "Churn", "positive_label": "Yes"},
            data_path="data/companies/teleconnect.csv",
        )
        db.session.add(dataset)
        db.session.commit()

        model_record = TrainedModel(
            company_id=company.id,
            dataset_id=dataset.id,
            algorithm="RandomForest",
            version=1,
            metrics_json={"accuracy": 0.82, "f1": 0.77, "roc_auc": 0.89},
            feature_importance_json={"Age": 0.42},
            model_path="models/teleconnect/v1.joblib",
            is_active=True,
            threshold_medium=0.35,
            threshold_high=0.65,
        )
        db.session.add(model_record)
        db.session.commit()

        access = UserCompanyAccess(user_id=admin.id, company_id=company.id)
        db.session.add(access)
        db.session.commit()

        prediction = Prediction(
            user_id=admin.id,
            company_id=company.id,
            model_id=model_record.id,
            age=35,
            gender="Male",
            tenure=12,
            monthly_charges=55.5,
            total_charges=660,
            contract_type="Month-to-Month",
            internet_service="Fiber",
            tech_support="No",
            prediction="Yes",
            probability=0.74,
            risk_level="High Risk",
            risk_factors=["Short tenure"],
            suggested_actions=["Call customer"],
            input_json={"Age": 35, "Gender": "Male"},
        )
        db.session.add(prediction)
        db.session.commit()

        assert Company.query.count() == 1
        assert CompanyDataset.query.filter_by(company_id=company.id).count() == 1
        assert TrainedModel.query.filter_by(company_id=company.id).count() == 1
        assert UserCompanyAccess.query.filter_by(user_id=admin.id, company_id=company.id).count() == 1
        assert Prediction.query.filter_by(company_id=company.id, model_id=model_record.id).count() == 1


def test_schema_and_training_services_work_on_synthetic_data(app, tmp_path):
    import pandas as pd

    with app.app_context():
        admin = User(
            name="Schema Admin",
            email="schema@example.com",
            password_hash="hash",
            role="admin",
        )
        db.session.add(admin)
        db.session.commit()

        company = Company(
            name="TestCompany",
            slug="testcompany",
            industry="Telecom",
            description="Synthetic test",
            logo_color="#111111",
            created_by=admin.id,
            is_active=True,
        )
        db.session.add(company)
        db.session.commit()

        data_path = tmp_path / "synthetic_company.csv"
        frame = pd.DataFrame({
            "CustomerID": [f"C{i:03d}" for i in range(1, 121)],
            "Age": [25 + (i % 40) for i in range(120)],
            "Gender": ["Male" if i % 2 == 0 else "Female" for i in range(120)],
            "Tenure": [6 + (i % 30) for i in range(120)],
            "MonthlyCharges": [50 + (i % 40) for i in range(120)],
            "ContractType": ["Month-to-Month" if i % 2 else "One Year" for i in range(120)],
            "TechSupport": ["No" if i % 3 else "Yes" for i in range(120)],
            "Churn": ["Yes" if (i % 5 == 0 or i % 7 == 0) else "No" for i in range(120)],
        })
        frame.to_csv(data_path, index=False)

        dataset = CompanyDataset(
            company_id=company.id,
            filename="synthetic_company.csv",
            row_count=len(frame),
            uploaded_by=admin.id,
            is_active=True,
            schema_json={
                "columns": list(frame.columns),
                "target_column": "Churn",
                "positive_label": "Yes",
                "id_columns": ["CustomerID"],
                "column_types": {
                    "Age": "numeric",
                    "Gender": "categorical",
                    "Tenure": "numeric",
                    "MonthlyCharges": "numeric",
                    "ContractType": "categorical",
                    "TechSupport": "categorical",
                },
                "allowed_categories": {
                    "Gender": ["Female", "Male"],
                    "ContractType": ["Month-to-Month", "One Year"],
                    "TechSupport": ["No", "Yes"],
                },
                "numeric_ranges": {
                    "Age": {"min": 18, "max": 80},
                    "Tenure": {"min": 0, "max": 72},
                    "MonthlyCharges": {"min": 0, "max": 200},
                },
            },
            data_path=str(data_path),
        )
        db.session.add(dataset)
        db.session.commit()

        schema = infer_schema(frame, "Churn")
        quality = build_data_quality_report(frame, "Churn")
        assert schema["target_column"] == "Churn"
        assert quality["rows"] == 120
        assert "warnings" in quality

        model_record = train_company_model(company.id, dataset.id)
        assert model_record.company_id == company.id
        assert model_record.is_active is True
        assert model_record.model_path.endswith(".joblib")


def test_admin_company_routes_and_session_selection(app, client):
    with app.app_context():
        admin = User(name="Admin", email="company-admin@example.com", password_hash="hash", role="admin")
        db.session.add(admin)
        db.session.commit()
        admin_id = admin.id
        company = Company(name="TeleConnect", slug="teleconnect", industry="Telecom", description="Demo", logo_color="#123456", created_by=admin_id, is_active=True)
        db.session.add(company)
        db.session.commit()
        company_id = company.id
        access = UserCompanyAccess(user_id=admin_id, company_id=company_id)
        db.session.add(access)
        db.session.commit()

    authenticate(client, admin_id)
    response = client.get("/admin/companies")
    assert response.status_code == 200
    assert b"TeleConnect" in response.data

    switch = client.post("/company/select", data={"company_id": str(company_id)})
    assert switch.status_code == 302
    with client.session_transaction() as session:
        assert session.get("active_company_id") == str(company_id)


def test_bulk_upload_rejects_file_with_missing_value_in_row(app, client, tmp_path):
    import pandas as pd

    user_id = add_user(app)
    data_path = tmp_path / "bulk-company.csv"
    pd.DataFrame({
        "CustomerID": ["C001", "C002", "C003"],
        "Age": [20, 35, 50],
        "Gender": ["Female", "Male", "Female"],
        "Churn": ["No", "Yes", "No"],
    }).to_csv(data_path, index=False)
    with app.app_context():
        company = Company(name="BulkCo", slug="bulkco", industry="Retail")
        db.session.add(company)
        db.session.flush()
        dataset = CompanyDataset(
            company_id=company.id,
            filename="bulk-company.csv",
            row_count=3,
            schema_json={
                "columns": ["CustomerID", "Age", "Gender", "Churn"],
                "target_column": "Churn",
                "positive_label": "Yes",
                "id_columns": ["CustomerID"],
            },
            data_path=str(data_path),
        )
        db.session.add(dataset)
        db.session.flush()
        db.session.add_all([
            TrainedModel(
                company_id=company.id,
                dataset_id=dataset.id,
                algorithm="StubModel",
                model_path="stub.joblib",
                is_active=True,
            ),
            UserCompanyAccess(user_id=user_id, company_id=company.id),
        ])
        db.session.commit()
        company_id = company.id
    with client.session_transaction() as session:
        session["active_company_id"] = str(company_id)
    authenticate(client, user_id)

    csv_data = (
        "Age,Gender\n"
        "30,Male\n"
        "25,\n"
    )

    response = client.post(
        "/bulk",
        data={"file": (io.BytesIO(csv_data.encode()), "customers.csv")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert b"Row 3" in response.data
    assert b"data is wrong" in response.data.lower()
    assert b"Gender" in response.data


def test_company_csv_download_contains_uploadable_dataset_rows(app, client, monkeypatch, tmp_path):
    import pandas as pd

    user_id = add_user(app)
    data_path = tmp_path / "template-company.csv"
    pd.DataFrame({
        "CustomerID": ["C001", "C002", "C003"],
        "Age": [20, 35, 50],
        "Gender": ["Female", "Male", None],
        "Churn": ["No", "Yes", "No"],
    }).to_csv(data_path, index=False)
    with app.app_context():
        company = Company(name="TemplateCo", slug="templateco", industry="Retail")
        db.session.add(company)
        db.session.flush()
        dataset = CompanyDataset(
            company_id=company.id,
            filename="template-company.csv",
            row_count=3,
            schema_json={
                "columns": ["CustomerID", "Age", "Gender", "Churn"],
                "target_column": "Churn",
                "id_columns": ["CustomerID"],
            },
            data_path=str(data_path),
        )
        db.session.add(dataset)
        db.session.flush()
        db.session.add_all([
            TrainedModel(
                company_id=company.id,
                dataset_id=dataset.id,
                algorithm="StubModel",
                model_path="stub.joblib",
                is_active=True,
            ),
            UserCompanyAccess(user_id=user_id, company_id=company.id),
        ])
        db.session.commit()
        company_id = company.id
    with client.session_transaction() as session:
        session["active_company_id"] = str(company_id)
    authenticate(client, user_id)

    template = client.get("/bulk/template.csv")
    assert template.status_code == 200
    frame = pd.read_csv(io.BytesIO(template.data))
    assert list(frame.columns) == ["Age", "Gender"]
    assert len(frame) == 2
    assert frame.iloc[0].to_dict() == {"Age": 20, "Gender": "Female"}

    monkeypatch.setattr(main_routes, "predict_many", lambda _company, data: data.assign(
        Prediction="No", Probability=0.1, RiskLevel="Low Risk"
    ))
    scored = client.post(
        "/bulk",
        data={"file": (io.BytesIO(template.data), "templateco-prediction-template.csv")},
        content_type="multipart/form-data",
    )
    assert scored.status_code == 200
    assert b"Prediction,Probability,Risk Level" in scored.data
    with app.app_context():
        assert Prediction.query.filter_by(company_id=company_id).count() == 2


