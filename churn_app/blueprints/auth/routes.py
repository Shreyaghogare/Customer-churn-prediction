import hmac
from datetime import timedelta, timezone

from flask import current_app, flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required, login_user, logout_user
from sqlalchemy.exc import IntegrityError
from werkzeug.security import check_password_hash, generate_password_hash

from ...extensions import db
from ...forms import LoginForm, RegisterForm
from ...models import LoginLog, User, utc_now
from . import bp

LOCKOUT_ATTEMPTS = 5
LOCKOUT_MINUTES = 15


def _locked(user, now):
    if not user.locked_until:
        return False
    locked_until = user.locked_until
    if locked_until.tzinfo is None:
        locked_until = locked_until.replace(tzinfo=timezone.utc)
    return locked_until > now


def _log_attempt(user, email, success):
    db.session.add(LoginLog(
        user_id=user.id if user else None,
        email_entered=(email or "")[:254],
        success=success,
        ip_address=request.remote_addr,
        user_agent=(request.user_agent.string or "")[:500],
    ))


@bp.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("admin.dashboard" if current_user.role == "admin" else "main.dashboard"))
    form = RegisterForm()
    if request.method == "GET":
        requested_mode = request.args.get("mode", "user")
        form.account_type.data = requested_mode if requested_mode in {"user", "admin"} else "user"
    mode = form.account_type.data if form.account_type.data in {"user", "admin"} else "user"
    if form.validate_on_submit():
        email = form.email.data.strip().lower()
        admin_code = (form.admin_code.data or "").strip()
        if mode == "admin":
            configured_admin_code = current_app.config.get("ADMIN_REGISTRATION_CODE") or ""
            if not configured_admin_code or not hmac.compare_digest(admin_code, configured_admin_code):
                form.admin_code.errors.append("Invalid admin secret code. Please enter the correct code.")
                return render_template("register.html", form=form, mode=mode)

        user = User(
            name=form.name.data,
            email=email,
            password_hash=generate_password_hash(form.password.data),
            role=mode,
        )
        db.session.add(user)
        try:
            db.session.commit()
        except IntegrityError:
            db.session.rollback()
            form.email.errors.append("An account with this email already exists.")
            return render_template("register.html", form=form, mode=mode)

        flash("Admin account created. You can now sign in." if mode == "admin" else "Account created. You can now sign in.", "success")
        return redirect(url_for("auth.login", mode=mode))

    return render_template("register.html", form=form, mode=mode)


@bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("admin.dashboard" if current_user.role == "admin" else "main.dashboard"))
    form = LoginForm()
    if request.method == "GET":
        requested_mode = request.args.get("mode", "user")
        form.account_type.data = requested_mode if requested_mode in {"user", "admin"} else "user"
    mode = form.account_type.data if form.account_type.data in {"user", "admin"} else "user"
    if form.validate_on_submit():
        email = form.email.data.strip().lower()
        user = User.query.filter_by(email=email).first()
        now = utc_now()
        valid_credentials = bool(
            user
            and user.is_active
            and not _locked(user, now)
            and check_password_hash(user.password_hash, form.password.data)
        )
        expected_role = mode
        if valid_credentials and user.role != expected_role:
            user.failed_attempts += 1
            if user.failed_attempts >= LOCKOUT_ATTEMPTS:
                user.locked_until = now + timedelta(minutes=LOCKOUT_MINUTES)
            _log_attempt(user, email, False)
            db.session.commit()
            message = (
                "You are not authorized to access the admin account."
                if mode == "admin"
                else "Admin accounts must use Admin Sign In."
            )
            flash(message, "error")
            return render_template("login.html", form=form, mode=mode)

        if valid_credentials:
            user.failed_attempts = 0
            user.locked_until = None
            user.last_login = now
            _log_attempt(user, email, True)
            db.session.commit()
            login_user(user, remember=form.remember_me.data)
            return redirect(url_for("admin.dashboard" if user.role == "admin" else "main.dashboard"))

        if user and user.is_active and not _locked(user, now):
            user.failed_attempts += 1
            if user.failed_attempts >= LOCKOUT_ATTEMPTS:
                user.locked_until = now + timedelta(minutes=LOCKOUT_MINUTES)
        _log_attempt(user, email, False)
        db.session.commit()
        form.password.errors.append("Invalid email or password.")
    elif request.method == "POST":
        email = request.form.get("email", "")[:254]
        user = User.query.filter_by(email=email.strip().lower()).first() if email else None
        _log_attempt(user, email, False)
        db.session.commit()
    return render_template("login.html", form=form, mode=mode)


@bp.route("/logout", methods=["POST"])
@login_required
def logout():
    logout_user()
    flash("You have been signed out.", "success")
    return redirect(url_for("auth.login"))
