"""Start the standalone Flask development server."""

from app import app

if __name__ == "__main__":
    settings = app.config["SETTINGS"]
    app.run(host=settings.host, port=settings.port, debug=settings.debug)
