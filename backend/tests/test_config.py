from app.core.config import Settings


def test_settings_load_environment_values() -> None:
    """Verify that settings accept configured application values."""
    settings = Settings(
        app_name="Test Application",
        app_version="1.0.0",
        environment="testing",
        debug=True,
        database_url="postgresql+psycopg://user:password@localhost/test_db",
        redis_url="redis://localhost:6379",
    )

    assert settings.app_name == "Test Application"
    assert settings.app_version == "1.0.0"
    assert settings.environment == "testing"
    assert settings.debug is True
