import errno, os, pathlib, stat
from src_flask import main
from flask import Flask
from subprocess import PIPE, Popen

def create_app(test_config=None):
    app = Flask(__name__, instance_relative_config=True)
    app.register_blueprint(main.bp)
    app.add_url_rule("/", endpoint="index")
    return app