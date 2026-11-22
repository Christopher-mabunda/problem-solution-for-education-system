"""
Pytest tests for the EduPrediction Streamlit application.

Expected project structure:
    project/
        app.py
        train_model.py
        src/
            __init__.py
            model.py
            config.py
            data_loader.py
            recommender.py
        models/                 # trained model files (location depends on src.model)
        tests/
            test_app.py

Install test dependencies:
    pip install pytest streamlit pandas numpy scikit-learn matplotlib seaborn

Run from the project root:
    python -m pytest tests/test_app.py -v

These are application-level tests. They use Streamlit AppTest to run the
real app.py, so the project's src modules and trained model must be present.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_FILE = PROJECT_ROOT / "app.py"


@pytest.fixture
def app():
    """Start the actual Streamlit app and return its AppTest instance."""
    if not APP_FILE.exists():
        pytest.fail(f"Could not find app.py at: {APP_FILE}")

    return AppTest.from_file(str(APP_FILE), default_timeout=30).run()


def assert_no_app_exception(at):
    """Fail with the Streamlit exception details if the app crashed."""
    errors = list(at.exception)
    assert not errors, "\n".join(str(error.message) for error in errors)


def test_app_file_exists():
    assert APP_FILE.exists(), f"Missing application file: {APP_FILE}"


def test_app_starts(app):
    """The app should start and show its navigation."""
    assert_no_app_exception(app)
    assert len(app.sidebar.radio) > 0
    assert "EduPrediction" in str(app.markdown)


@pytest.mark.parametrize(
    "page",
    ["Assessment", "Dashboard", "Model Performance", "Chatbot", "About"],
)
def test_navigation_pages_load(page):
    """Each page should render without an uncaught exception."""
    at = AppTest.from_file(str(APP_FILE), default_timeout=30).run()
    assert_no_app_exception(at)

    at.sidebar.radio[0].set_value(page).run()

    assert_no_app_exception(at)


def test_assessment_form_is_displayed(app):
    """The assessment page should show the five main student questions."""
    assert_no_app_exception(app)

    visible_text = " ".join(
        [str(item.label) for item in app.number_input]
        + [str(item.label) for item in app.selectbox]
        + [str(item.label) for item in app.slider]
        + [str(item.label) for item in app.select_slider]
    )

    assert "Age" in visible_text
    assert "Gender" in visible_text
    assert "Weekly study time" in visible_text
    assert "Number of absences" in visible_text
    assert "Second period grade" in visible_text


def test_assessment_submission():
    """
    Submit the assessment using the form's default values.

    The app should either show a prediction or display a clear validation/
    model error. It must not crash with an uncaught exception.
    """
    at = AppTest.from_file(str(APP_FILE), default_timeout=30).run()
    assert_no_app_exception(at)

    submit_buttons = [
        button for button in at.button
        if button.label == "Analyse"
    ]
    assert submit_buttons, "The Analyse button was not found."

    submit_buttons[0].click().run()
    assert_no_app_exception(at)

    rendered_text = " ".join(str(item.value) for item in at.markdown)
    rendered_text += " " + " ".join(str(item.value) for item in at.info)
    rendered_text += " " + " ".join(str(item.value) for item in at.error)

    # A successful assessment should show a risk result or a useful message.
    assert (
        "Low" in rendered_text
        or "Medium" in rendered_text
        or "High" in rendered_text
        or "model" in rendered_text.lower()
    )


def test_dashboard_page():
    """The dashboard page should load without an uncaught exception."""
    at = AppTest.from_file(str(APP_FILE), default_timeout=30).run()
    at.sidebar.radio[0].set_value("Dashboard").run()

    assert_no_app_exception(at)
    assert any(
        "dashboard" in str(item.value).lower()
        for item in at.subheader
    )


def test_model_performance_page():
    """The model performance page should display its heading."""
    at = AppTest.from_file(str(APP_FILE), default_timeout=30).run()
    at.sidebar.radio[0].set_value("Model Performance").run()

    assert_no_app_exception(at)
    assert any(
        "model performance" in str(item.value).lower()
        for item in at.subheader
    )


def test_chatbot_page():
    """The chatbot page should display its heading and input."""
    at = AppTest.from_file(str(APP_FILE), default_timeout=30).run()
    at.sidebar.radio[0].set_value("Chatbot").run()

    assert_no_app_exception(at)
    assert any(
        "chatbot" in str(item.value).lower()
        for item in at.subheader
    )
    assert len(at.chat_input) > 0


def test_about_page():
    """The About page should describe the three risk levels."""
    at = AppTest.from_file(str(APP_FILE), default_timeout=30).run()
    at.sidebar.radio[0].set_value("About").run()

    assert_no_app_exception(at)

    about_text = " ".join(str(item.value) for item in at.markdown)
    assert "EduPrediction" in about_text
    assert "Low" in about_text
    assert "Medium" in about_text
    assert "High" in about_text
