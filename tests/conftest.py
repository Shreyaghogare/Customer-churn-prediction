import pytest

from churn_app import create_app
from churn_app.extensions import db
@pytest.fixture
def app():
    app = create_app({
        "TESTING": True,
        "SECRET_KEY": "pytest-secret-key",
        "ADMIN_REGISTRATION_CODE": "0808",
        "SQLALCHEMY_DATABASE_URI": "sqlite://",
        "WTF_CSRF_ENABLED": False,
    })
    yield app
    with app.app_context():
        db.session.remove()
        db.drop_all()


@pytest.fixture
def client(app):
    return app.test_client()