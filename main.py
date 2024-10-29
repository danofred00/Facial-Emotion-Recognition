from packages import backend, frontend

def get_app():
    app = backend.get_app()
    app.mount("/app", frontend.get_static_files('packages/frontend'))

    return app

app = get_app()